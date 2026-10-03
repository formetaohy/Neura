use neura_abi::Element;
use neura_gpu::{Backends, GpuRequest, LimitsPolicy};
use neura_graph::{Graph, Shape};
use neura_runtime::{Program, Runtime, RuntimeRequest};
use std::time::Instant;

fn open() -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 4 << 20,
        heap_bytes: 256 << 20,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn open_backend(backends: Backends) -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        gpu: GpuRequest {
            backends,
            limits: LimitsPolicy::Adapter,
            ..Default::default()
        },
        readback_bytes: 4 << 20,
        heap_bytes: 256 << 20,
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn rectifier<'g>(runtime: &'g Runtime, graph: &Graph<'g>, elements: u32) -> Program<'g> {
    let data = graph.input(Shape::vector(elements), Element::Single);
    graph.retain(graph.relu(graph.mul(data, data)));
    let weights = runtime.weights(graph);
    let program = runtime.compile(graph, &weights);
    runtime.write(&program, data, &vec![0.5; elements as usize]);
    program
}

#[test]
fn a_device_time_counts_the_work_and_not_the_round_trip() {
    let runtime = open();
    let graph = Graph::new();
    let heavy_graph = Graph::new();
    let tiny = rectifier(&runtime, &graph, 1024);
    let heavy = rectifier(&runtime, &heavy_graph, 16 << 20);
    let mut tiny_seconds = f64::MAX;
    let mut heavy_seconds = f64::MAX;
    for _ in 0..6 {
        tiny_seconds = tiny_seconds.min(runtime.run(&tiny).seconds());
        heavy_seconds = heavy_seconds.min(runtime.run(&heavy).seconds());
    }
    assert!(
        tiny_seconds > 0.0,
        "the device reported a run of its own as taking {tiny_seconds} seconds",
    );
    assert!(
        heavy_seconds > 8.0 * tiny_seconds,
        "a run over a million times the elements took {heavy_seconds} seconds beside {tiny_seconds}",
    );
    let started = Instant::now();
    let round = runtime.run(&tiny);
    let _ = round.seconds();
    let round_trip = started.elapsed().as_secs_f64();
    assert!(
        tiny_seconds < round_trip,
        "a tiny run took {tiny_seconds} seconds of device time beside a {round_trip} second round trip",
    );
}

#[test]
fn a_measured_program_reports_the_time_of_its_own_runs() {
    let runtime = open();
    let graph = Graph::new();
    let program = rectifier(&runtime, &graph, 16 << 20);
    let run = runtime.run(&program).seconds();
    let measured = runtime.measure(&program);
    assert!(
        measured > 0.0 && measured < 8.0 * run,
        "the device measured {measured} seconds per run beside a single run of {run} seconds",
    );
}

#[test]
fn every_backend_a_machine_offers_times_the_same_work() {
    let mut measured = Vec::new();
    for backends in Backends::PLATFORM {
        let runtime = open_backend(backends);
        let graph = Graph::new();
        let program = rectifier(&runtime, &graph, 16 << 20);
        let mut fastest = f64::MAX;
        for _ in 0..6 {
            fastest = fastest.min(runtime.run(&program).seconds());
        }
        assert!(
            fastest > 0.0,
            "{:?} reported a run of its own as taking {fastest} seconds",
            runtime.context().adapter_info().backend,
        );
        measured.push(fastest);
    }
    let fastest = measured.iter().copied().fold(f64::MAX, f64::min);
    let slowest = measured.iter().copied().fold(0.0, f64::max);
    assert!(
        slowest < 4.0 * fastest,
        "the backends timed the same work at {measured:?} seconds",
    );
}
