use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_nn::{Adam, Linear, Sgd, mse_loss};
use neura_runtime::{Runtime, RuntimeRequest};
use std::time::Instant;

fn open() -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn surface(samples: u32) -> (Vec<f32>, Vec<f32>) {
    let mut inputs = Vec::with_capacity(samples as usize * 2);
    let mut targets = Vec::with_capacity(samples as usize);
    for sample in 0..samples {
        let first = (sample as f32 / samples as f32) * 2.0 - 1.0;
        let second = ((sample * 7) % samples) as f32 / samples as f32 * 2.0 - 1.0;
        inputs.push(first);
        inputs.push(second);
        targets.push((first * second).tanh() + 0.25 * (first + second));
    }
    (inputs, targets)
}

#[test]
fn half_precision_activations_train_a_model() {
    let runtime = open();
    let graph = Graph::new();
    let first = Linear::new(
        &graph,
        2,
        24,
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Half,
    );
    let second = Linear::new(
        &graph,
        24,
        1,
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Half,
    );
    let inputs = graph.input(Shape::matrix(64, 2), Element::Single);
    let targets = graph.input(Shape::matrix(64, 1), Element::Single);
    let hidden = graph.relu(first.forward(&graph, graph.cast(inputs, Element::Half)));
    let prediction = second.forward(&graph, hidden);
    assert_eq!(graph.element(hidden), Element::Half);
    assert_eq!(graph.element(prediction), Element::Half);
    let loss = mse_loss(&graph, prediction, targets);
    let gradients = graph.backward(loss);
    let mut optimizer = Adam::new(&graph, 0.02, 0.9, 0.999, 1e-8);
    optimizer.track_all(&graph, &first.parameters());
    optimizer.track_all(&graph, &second.parameters());
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (inputs_data, targets_data) = surface(64);
    runtime.write(&program, inputs, &inputs_data);
    runtime.write(&program, targets, &targets_data);
    let mut start = None;
    let mut end = 0.0;
    for step in 0..400 {
        runtime.run(&program);
        if step % 40 == 0 || step == 399 {
            end = runtime.read(&program, loss)[0];
            start.get_or_insert(end);
        }
    }
    let start = start.expect("a first loss");
    assert!(
        end < start * 0.5,
        "the loss fell from {start} to {end} while every activation stayed half precision",
    );
}

#[test]
fn a_narrow_layer_halves_the_weights_it_carries() {
    let runtime = open();
    let graph = Graph::new();
    let layer = Linear::new(&graph, 512, 512, Init::Zero, Element::Half);
    let wide = Graph::new();
    let twin = Linear::new(&wide, 512, 512, Init::Zero, Element::Single);
    let input = graph.input(Shape::matrix(8, 512), Element::Half);
    let halved = layer.forward(&graph, input);
    let wide_input = wide.input(Shape::matrix(8, 512), Element::Single);
    let doubled = twin.forward(&wide, wide_input);
    assert_eq!(graph.element(halved), Element::Half);
    assert_eq!(wide.element(doubled), Element::Single);
    let narrow_store = runtime.weights(&graph);
    let wide_store = runtime.weights(&wide);
    assert_eq!(
        narrow_store.bytes() * 2,
        wide_store.bytes(),
        "a half precision layer carries half the bytes of its single precision twin",
    );
    let program = runtime.compile(&graph, &narrow_store);
    runtime.write(&program, input, &vec![0.5; 8 * 512]);
    runtime.run(&program);
    let produced = runtime.read(&program, halved);
    assert_eq!(
        produced.len(),
        8 * 512,
        "a narrow layer writes every row it scores"
    );
}

#[test]
fn a_narrow_step_runs_beside_a_wide_one() {
    let runtime = open();
    let graph = Graph::new();
    let model = Linear::new(
        &graph,
        256,
        256,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Half,
    );
    let input = graph.input(Shape::matrix(256, 256), Element::Half);
    let prediction = model.forward(&graph, input);
    let loss = mse_loss(&graph, prediction, prediction);
    let gradients = graph.backward(loss);
    let optimizer = Sgd::new(&graph, 0.01);
    optimizer.step(&graph, &gradients, &model.parameters());
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, input, &vec![0.5; 256 * 256]);
    for _ in 0..4 {
        runtime.run(&program);
    }
    runtime.context().drain();
    let started = Instant::now();
    for _ in 0..16 {
        runtime.run(&program);
    }
    runtime.context().drain();
    let per_step = started.elapsed().as_secs_f64() * 1e6 / 16.0;
    assert!(
        program.updates_weights(),
        "the optimizer writes the half precision weights it steps over",
    );
    assert!(per_step > 0.0 && per_step.is_finite());
}
