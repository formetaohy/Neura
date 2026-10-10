use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_nn::{Conv2d, Linear, Mlp};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

fn open() -> Runtime {
    Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            readback_bytes: 1 << 16,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn reach(fan_in: u32, gain: f32) -> f32 {
    gain * (3.0 / fan_in as f32).sqrt()
}

#[test]
fn every_layer_of_a_stack_draws_the_numbers_its_own_fan_names() {
    let runtime = open();
    let graph = Graph::new();
    let gain = (2.0f32).sqrt();
    let model = Mlp::new(
        &graph,
        "model",
        &[32, 64, 4],
        Init::Kaiming { gain },
        Element::Single,
    );
    let records = graph.input(Shape::matrix(4, 32), Element::Single);
    graph.retain(model.forward(&graph, records));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, records, &[0.0; 4 * 32]);
    runtime.run(&program);
    let first = runtime.read(&program, model.layers()[0].weight());
    let second = runtime.read(&program, model.layers()[1].weight());
    let (first_reach, second_reach) = (reach(32, gain), reach(64, gain));
    assert!(
        first.iter().all(|value| value.abs() <= first_reach),
        "a Kaiming init of 32 inputs draws no number beyond {first_reach}",
    );
    assert!(
        second.iter().all(|value| value.abs() <= second_reach),
        "a Kaiming init of 64 inputs draws no number beyond {second_reach}",
    );
    assert!(
        second.iter().all(|value| value.abs() <= second_reach),
        "a layer of 64 inputs draws the reach of its own fan",
    );
    assert!(
        first.iter().any(|value| value.abs() > second_reach),
        "a layer of 32 inputs draws numbers a layer of 64 inputs cannot reach",
    );
    assert!(
        first.iter().any(|value| *value != first[0]) && first.iter().all(|value| value.is_finite()),
        "a fan aware init spreads finite numbers across its weights",
    );
    assert_eq!(
        second.len(),
        64 * 4,
        "a layer of 64 inputs into 4 outputs holds one weight per pair",
    );
}

#[test]
fn a_normal_init_lands_in_the_weights_the_device_reads() {
    let runtime = open();
    let graph = Graph::new();
    let layer = Linear::new(
        &graph,
        "layer",
        512,
        8,
        Init::Normal {
            mean: 0.0,
            deviation: 0.02,
        },
        Element::Single,
    );
    let records = graph.input(Shape::matrix(4, 512), Element::Single);
    graph.retain(layer.forward(&graph, records));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, records, &[0.0; 4 * 512]);
    runtime.run(&program);
    let drawn = runtime.read(&program, layer.weight());
    let count = drawn.len() as f32;
    let mean = drawn.iter().sum::<f32>() / count;
    let deviation = (drawn
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / count)
        .sqrt();
    assert!(
        mean.abs() < 0.004,
        "a normal init of mean zero landed at {mean} in the weights",
    );
    assert!(
        (deviation - 0.02).abs() < 0.003,
        "a normal init of deviation 0.02 landed at {deviation} in the weights",
    );
    assert_eq!(
        runtime.read(&program, layer.bias()),
        vec![0.0; 8],
        "a bias stays where its seed left it",
    );
}

#[test]
fn a_convolution_spreads_the_fan_of_its_taps() {
    let runtime = open();
    let graph = Graph::new();
    let gain = 1.0;
    let model = Conv2d::new(
        &graph,
        "model",
        [8, 16],
        1,
        neura_graph::Window::sliding([3, 3]),
        Init::Xavier { gain },
        Element::Single,
    );
    let images = graph.input(Shape::of([1, 8, 4, 4]), Element::Single);
    graph.retain(model.forward(&graph, images));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, images, &[0.0; 8 * 4 * 4]);
    runtime.run(&program);
    let drawn = runtime.read(&program, model.filter());
    let fan_in = 8 * 3 * 3;
    let fan_out = 16 * 3 * 3;
    let bound = gain * (6.0 / (fan_in + fan_out) as f32).sqrt();
    assert_eq!(
        drawn.len(),
        16 * 8 * 3 * 3,
        "a convolution of 8 channels into 16 over a 3x3 window holds one weight per tap",
    );
    assert!(
        drawn.iter().all(|value| value.abs() <= bound),
        "a Xavier init of {fan_in} inputs and {fan_out} outputs draws no number beyond {bound}",
    );
    assert!(
        drawn.iter().any(|value| value.abs() > bound * 0.5),
        "a Xavier init spreads its numbers across the reach of its fan",
    );
}
