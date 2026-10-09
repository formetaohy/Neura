use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_runtime::{Backends, MemoryRequest, Runtime, RuntimeRequest};
use std::path::{Path, PathBuf};

const ROWS: u32 = 16;
const WIDTH: u32 = 128;
const STEPS: u32 = 4;
const RESIDENT_BYTES: u64 = 5 * (1 << 14);

fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("neura-rate-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("a spill directory");
    path
}

fn open(backends: Backends, memory: MemoryRequest) -> Runtime {
    Runtime::open(RuntimeRequest {
        gpu: neura_runtime::GpuRequest {
            backends,
            ..Default::default()
        },
        memory,
    })
    .unwrap_or_else(|error| panic!("no device runs the rate tests over {backends:?}: {error}"))
}

fn rate_of(step: u32) -> f32 {
    match step {
        0 => 0.2,
        1 => 0.1,
        2 => 0.05,
        _ => 0.025,
    }
}

fn scheduled(
    backends: Backends,
    memory: MemoryRequest,
    rate: impl Fn(u32) -> f32,
) -> (Vec<Vec<f32>>, f32, u64) {
    let runtime = open(backends, memory);
    let graph = Graph::new();
    let init = Init::Uniform {
        low: -0.1,
        high: 0.1,
    };
    let params = [
        graph.parameter(Shape::matrix(WIDTH, WIDTH), init, Element::Single),
        graph.parameter(Shape::matrix(WIDTH, WIDTH), init, Element::Single),
    ];
    let bias = graph.parameter(Shape::matrix(1, WIDTH), Init::Zero, Element::Single);
    let rate_state = graph.named_state(
        "descent.rate",
        Shape::scalar(),
        Init::Constant(rate(0)),
        Element::Single,
    );
    let data = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    let target = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    let hidden = graph.relu(graph.add(graph.matmul(data, params[0]), bias));
    let prediction = graph.matmul(hidden, params[1]);
    let difference = graph.sub(prediction, target);
    let loss = graph.sum(graph.mul(difference, difference));
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let descent = graph.neg(rate_state);
    for parameter in [params[0], params[1], bias] {
        graph.add_into(parameter, graph.mul(gradients.of(parameter), descent));
    }

    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations = (0..(ROWS * WIDTH))
        .map(|index| ((index * 37) % 101) as f32 / 101.0 - 0.5)
        .collect::<Vec<_>>();
    let targets = (0..(ROWS * WIDTH))
        .map(|index| ((index * 53) % 97) as f32 / 97.0 - 0.5)
        .collect::<Vec<_>>();
    let mut held = 0.0;
    for step in 0..STEPS {
        runtime.write(&program, data, &observations);
        runtime.write(&program, target, &targets);
        runtime.write(&program, rate_state, &[rate(step)]);
        runtime.run(&program);
        held = runtime.read(&program, loss)[0];
    }
    assert!(
        program.is_current(),
        "a rate the host writes leaves the program of a streamed store current",
    );
    assert_eq!(
        runtime.built_plans(),
        1,
        "a schedule steers one plan rather than building a plan per step",
    );
    assert_eq!(
        runtime.read(&program, rate_state),
        [rate(STEPS - 1)],
        "a streamed store reads back the rate its host wrote",
    );
    let parameters = [params[0], params[1], bias]
        .iter()
        .map(|parameter| runtime.read(&program, *parameter))
        .collect();
    (parameters, held, weights.readback_pages())
}

fn a_rate_reaches_the_store_the_device_cannot_hold(backends: Backends, spilled: Option<&Path>) {
    let memory = || MemoryRequest {
        resident_weight_bytes: Some(RESIDENT_BYTES),
        weight_spill: spilled.map(Path::to_path_buf),
        ..Default::default()
    };
    let (resident, resident_loss, _) = scheduled(backends, MemoryRequest::default(), rate_of);
    let (streamed, streamed_loss, churn) = scheduled(backends, memory(), rate_of);
    let (constant, _, _) = scheduled(backends, memory(), |_| rate_of(0));
    assert!(
        streamed[0]
            .iter()
            .zip(&constant[0])
            .any(|(scheduled, constant)| (scheduled - constant).abs() > 1e-5),
        "a rate the host writes every step steers a store the device cannot hold, and the rate the store was built with leaves the same numbers",
    );
    assert!(
        churn > 0,
        "a store of {RESIDENT_BYTES} bytes beside a model that overgrows them read {churn} pages back",
    );
    assert_eq!(
        resident.len(),
        streamed.len(),
        "a streamed store carries the parameters of its model",
    );
    for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
        assert_eq!(expected.len(), observed.len());
        for (index, (expected, observed)) in expected.iter().zip(observed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a written rate reached {observed} where the resident store reached {expected} at parameter {at} element {index}",
            );
        }
    }
    assert!(
        (resident_loss - streamed_loss).abs() <= 1e-5,
        "a streamed store held a loss of {streamed_loss} where the resident store held {resident_loss}",
    );
}

#[test]
fn a_written_rate_reaches_a_weight_store_that_pages() {
    for backends in Backends::PLATFORM {
        a_rate_reaches_the_store_the_device_cannot_hold(backends, None);
    }
}

#[test]
fn a_written_rate_reaches_a_weight_store_that_spills() {
    for backends in Backends::PLATFORM {
        let path = directory("spilled");
        a_rate_reaches_the_store_the_device_cannot_hold(backends, Some(&path));
    }
}
