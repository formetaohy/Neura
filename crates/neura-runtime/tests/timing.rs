use neura_abi::Element;
use neura_gpu::{Backend, GpuContext, GpuRequest, LimitsPolicy, PREFERENCE};
use neura_graph::{Graph, Shape};
use neura_runtime::{MemoryRequest, Program, Runtime, RuntimeRequest};
use std::time::Instant;

const HEAVY_ELEMENTS: u64 = 16 << 20;
const HEAP_BYTES: u64 = 256 << 20;

fn heap_bytes(backend: Backend) -> u64 {
    let request = GpuRequest {
        backend: Some(backend),
        ..Default::default()
    };
    let context = GpuContext::open(&request).expect("a device serves a heap probe");
    HEAP_BYTES.min(context.limits().max_storage_buffer_binding_size)
}

fn open() -> Runtime {
    Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            readback_bytes: 4 << 20,
            heap_bytes: heap_bytes(PREFERENCE[0]),
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn heavy_elements(runtime: &Runtime) -> u32 {
    u32::try_from((runtime.heap_bytes() / 16).min(HEAVY_ELEMENTS))
        .expect("a heavy workload fits one word")
}

fn open_backend(backend: Backend) -> Runtime {
    Runtime::open(RuntimeRequest {
        gpu: GpuRequest {
            backend: Some(backend),
            limits: LimitsPolicy::Adapter,
            ..Default::default()
        },
        memory: MemoryRequest {
            readback_bytes: 4 << 20,
            heap_bytes: heap_bytes(backend),
            ..Default::default()
        },
    })
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn rectifier(runtime: &Runtime, graph: &Graph, elements: u32) -> Program {
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
    let heavy = rectifier(&runtime, &heavy_graph, heavy_elements(&runtime));
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
    let program = rectifier(&runtime, &graph, heavy_elements(&runtime));
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
    for &backend in PREFERENCE {
        let runtime = open_backend(backend);
        let graph = Graph::new();
        let program = rectifier(&runtime, &graph, heavy_elements(&runtime));
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
