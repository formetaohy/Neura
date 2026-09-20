use neura_nn::{Adam, Linear, Mlp, Sgd, cross_entropy, mse_loss};
use neura_program::{Graph, Init, Shape};
use neura_runtime::{Runtime, RuntimeRequest};

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
fn a_multilayer_perceptron_learns_a_nonlinear_surface() {
    let runtime = open();
    let graph = Graph::new();
    let model = Mlp::new(
        &graph,
        &[2, 24, 24, 1],
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
    );
    let inputs = graph.input(Shape::matrix(64, 2));
    let targets = graph.input(Shape::matrix(64, 1));
    let prediction = model.forward(&graph, inputs);
    let loss = mse_loss(&graph, prediction, targets);
    let gradients = graph.backward(loss);
    let mut optimizer = Adam::new(&graph, 0.02, 0.9, 0.999, 1e-8);
    optimizer.track_all(&graph, &model.parameters());
    optimizer.step(&graph, &gradients);
    let program = runtime.compile(&graph);
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
        end < start * 0.2,
        "the loss fell from {start} to {end} over four hundred steps",
    );
    assert!(end.is_finite() && end >= 0.0, "the loss ended at {end}");
}

#[test]
fn descent_lowers_the_loss_of_a_single_layer() {
    let runtime = open();
    let graph = Graph::new();
    let layer = Linear::new(
        &graph,
        3,
        1,
        Init::Uniform {
            low: -0.3,
            high: 0.3,
        },
    );
    let inputs = graph.input(Shape::matrix(16, 3));
    let targets = graph.input(Shape::matrix(16, 1));
    let loss = mse_loss(&graph, layer.forward(&graph, inputs), targets);
    let gradients = graph.backward(loss);
    let optimizer = Sgd::new(&graph, 0.05);
    optimizer.step(&graph, &gradients, &layer.parameters());
    let program = runtime.compile(&graph);
    let inputs_data = (0..48)
        .map(|index| index as f32 * 0.02 - 0.5)
        .collect::<Vec<_>>();
    runtime.write(&program, inputs, &inputs_data);
    runtime.write(&program, targets, &[1.0; 16]);
    runtime.run(&program);
    let first = runtime.read(&program, loss)[0];
    for _ in 0..20 {
        runtime.run(&program);
    }
    let last = runtime.read(&program, loss)[0];
    assert!(
        last < first,
        "twenty steps moved the loss from {first} to {last}",
    );
}

#[test]
fn a_step_over_a_parameter_without_a_gradient_stops_the_graph() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 4), Init::Zero);
    let other = graph.parameter(Shape::matrix(4, 4), Init::Zero);
    let loss = graph.sum(graph.relu(weight));
    let gradients = graph.backward(loss);
    let optimizer = Sgd::new(&graph, 0.1);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        optimizer.step(&graph, &gradients, &[other]);
    }));
    assert!(
        outcome.is_err(),
        "a step over a parameter the loss never reached was accepted",
    );
}

#[test]
fn moments_track_each_parameter_once() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero);
    let mut optimizer = Adam::new(&graph, 0.1, 0.9, 0.999, 1e-8);
    optimizer.track(&graph, weight);
    assert_eq!(optimizer.moments().len(), 1);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        optimizer.track(&graph, weight);
    }));
    assert!(
        outcome.is_err(),
        "one parameter carried two pairs of moments"
    );
}

fn actions(samples: u32) -> (Vec<f32>, Vec<f32>) {
    let mut observations = Vec::with_capacity(samples as usize * 2);
    let mut targets = Vec::with_capacity(samples as usize * 3);
    for sample in 0..samples {
        let action = sample % 3;
        let angle = action as f32 * core::f32::consts::TAU / 3.0;
        let reach = 0.9 + 0.05 * (sample % 5) as f32;
        observations.push(angle.cos() * reach);
        observations.push(angle.sin() * reach);
        for class in 0..3 {
            targets.push(f32::from(class == action));
        }
    }
    (observations, targets)
}

fn chosen(logits: &[f32]) -> usize {
    let mut best = 0;
    for class in 1..logits.len() {
        if logits[class] > logits[best] {
            best = class;
        }
    }
    best
}

#[test]
fn a_network_learns_the_action_it_was_shown() {
    let runtime = open();
    let graph = Graph::new();
    let model = Mlp::new(
        &graph,
        &[2, 16, 3],
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    );
    let observations = graph.input(Shape::matrix(24, 2));
    let targets = graph.input(Shape::matrix(24, 3));
    let logits = model.forward(&graph, observations);
    graph.retain(logits);
    let loss = cross_entropy(&graph, logits, targets);
    let gradients = graph.backward(loss);
    let mut optimizer = Adam::new(&graph, 0.05, 0.9, 0.999, 1e-8);
    optimizer.track_all(&graph, &model.parameters());
    optimizer.step(&graph, &gradients);
    let program = runtime.compile(&graph);
    let (observation_data, target_data) = actions(24);
    runtime.write(&program, observations, &observation_data);
    runtime.write(&program, targets, &target_data);
    let mut start = None;
    let mut end = 0.0;
    for step in 0..400 {
        runtime.run(&program);
        if step % 50 == 0 || step == 399 {
            end = runtime.read(&program, loss)[0];
            start.get_or_insert(end);
        }
    }
    let start = start.expect("a first loss");
    assert!(
        start.is_finite() && start > 0.0,
        "the loss started at {start}"
    );
    assert!(
        end < start * 0.2,
        "the loss fell from {start} to {end} over four hundred steps",
    );
    let learned = runtime.read(&program, logits);
    for sample in 0..24 {
        assert_eq!(
            chosen(&learned[sample * 3..sample * 3 + 3]),
            sample % 3,
            "sample {sample} was shown action {}",
            sample % 3,
        );
    }
}
