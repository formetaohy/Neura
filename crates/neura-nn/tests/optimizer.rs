use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_nn::{AdamW, Sgd};
use neura_runtime::{Runtime, RuntimeRequest};

fn open() -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn constant<'g>(graph: &Graph<'g>, name: f32) -> Value<'g> {
    graph.fill(Shape::scalar(), name)
}

#[test]
fn a_corrected_adam_takes_its_rate_on_the_first_step() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::scalar(), Init::Zero, Element::Single);
    let loss = graph.sum(graph.mul(weight, constant(&graph, 5.0)));
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, 0.1, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track(&graph, weight);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    let moved = runtime.read(&program, weight)[0];
    assert!(
        (moved + 0.1).abs() < 1e-5,
        "a bias corrected adam moved a weight of gradient 5 to {moved} where its rate asks for -0.1",
    );
}

#[test]
fn adam_matches_a_reference_with_bias_correction_and_decay() {
    let runtime = open();
    let graph = Graph::new();
    let rate = 0.05f32;
    let decay = 0.01f32;
    let weight = graph.parameter(Shape::vector(4), Init::Constant(1.0), Element::Single);
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(4), 0.5)));
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, rate, 0.9, 0.999, 1e-8, decay);
    optimizer.track(&graph, weight);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let steps = 20;
    for _ in 0..steps {
        runtime.run(&program);
    }
    let produced = runtime.read(&program, weight);
    let (mean_decay, variance_decay, floor) = (0.9f32, 0.999f32, 1e-8f32);
    let gradient = 0.5f32;
    let mut reference = [1.0f32; 4];
    let mut mean = [0.0f32; 4];
    let mut variance = [0.0f32; 4];
    for step in 1..=steps {
        for index in 0..4 {
            mean[index] = mean_decay * mean[index] + (1.0 - mean_decay) * gradient;
            variance[index] =
                variance_decay * variance[index] + (1.0 - variance_decay) * gradient * gradient;
            let corrected_mean = mean[index] / (1.0 - mean_decay.powi(step));
            let corrected_variance = variance[index] / (1.0 - variance_decay.powi(step));
            reference[index] -= rate
                * (corrected_mean / (corrected_variance.sqrt() + floor) + decay * reference[index]);
        }
    }
    for index in 0..4 {
        assert!(
            (produced[index] - reference[index]).abs() < 1e-4,
            "the device kept {} where a bias corrected adam with decay says {}",
            produced[index],
            reference[index],
        );
    }
}

#[test]
fn a_weight_decays_at_its_rate_without_a_gradient() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::scalar(), Init::Constant(1.0), Element::Single);
    let loss = graph.sum(graph.mul(weight, constant(&graph, 0.0)));
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, 0.1, 0.9, 0.999, 1e-8, 0.1);
    optimizer.track(&graph, weight);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    let decayed = runtime.read(&program, weight)[0];
    assert!(
        (decayed - 0.99).abs() < 1e-5,
        "a weight of 1 under a decay of 0.1 at a rate of 0.1 landed at {decayed} where 0.99 is the answer",
    );
}

#[test]
fn a_momentum_descent_matches_a_reference() {
    let runtime = open();
    let graph = Graph::new();
    let rate = 0.1f32;
    let momentum_decay = 0.9f32;
    let weight = graph.parameter(Shape::vector(2), Init::Zero, Element::Single);
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(2), 1.0)));
    let gradients = graph.backward(loss);
    let mut optimizer = Sgd::momentum(&graph, rate, momentum_decay, 0.0);
    optimizer.track_all(&graph, &[weight]);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let steps = 20;
    for _ in 0..steps {
        runtime.run(&program);
    }
    let produced = runtime.read(&program, weight);
    let gradient = 1.0f32;
    let mut velocity = [0.0f32; 2];
    let mut reference = [0.0f32; 2];
    for _ in 0..steps {
        for index in 0..2 {
            velocity[index] = momentum_decay * velocity[index] + gradient;
            reference[index] -= rate * velocity[index];
        }
    }
    for index in 0..2 {
        assert!(
            (produced[index] - reference[index]).abs() < 1e-4,
            "the device kept {} where a momentum descent says {}",
            produced[index],
            reference[index],
        );
    }
}

#[test]
fn a_decay_enters_a_momentum_before_its_velocity() {
    let runtime = open();
    let graph = Graph::new();
    let rate = 0.1f32;
    let momentum_decay = 0.9f32;
    let decay = 0.1f32;
    let weight = graph.parameter(Shape::vector(2), Init::Constant(1.0), Element::Single);
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(2), 0.0)));
    let gradients = graph.backward(loss);
    let mut optimizer = Sgd::momentum(&graph, rate, momentum_decay, decay);
    optimizer.track_all(&graph, &[weight]);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let steps = 20;
    for _ in 0..steps {
        runtime.run(&program);
    }
    let produced = runtime.read(&program, weight);
    let mut velocity = [0.0f32; 2];
    let mut reference = [1.0f32; 2];
    for _ in 0..steps {
        for index in 0..2 {
            let gradient = decay * reference[index];
            velocity[index] = momentum_decay * velocity[index] + gradient;
            reference[index] -= rate * velocity[index];
        }
    }
    for index in 0..2 {
        assert!(
            (produced[index] - reference[index]).abs() < 1e-4,
            "the device kept {} where a momentum descent with a decay of {decay} says {}",
            produced[index],
            reference[index],
        );
    }
}

#[test]
fn clipping_scales_a_gradient_set_by_its_global_norm() {
    let runtime = open();
    let graph = Graph::new();
    let first = graph.parameter(Shape::scalar(), Init::Zero, Element::Single);
    let second = graph.parameter(Shape::scalar(), Init::Zero, Element::Single);
    let loss = graph.sum(graph.add(
        graph.mul(first, constant(&graph, 3.0)),
        graph.mul(second, constant(&graph, 4.0)),
    ));
    let gradients = graph.backward(loss);
    let clipped = gradients.clip(&graph, 1.0);
    let mut optimizer = Sgd::new(&graph, 1.0, 0.0);
    optimizer.track_all(&graph, &[first, second]);
    optimizer.step(&graph, &clipped);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    let moved = runtime.read_many(&program, &[first, second]);
    assert!(
        (moved[0][0] + 0.6).abs() < 1e-5,
        "a gradient of 3 in a set of norm 5 clipped to 1 moved to {} where -0.6 is the answer",
        moved[0][0],
    );
    assert!(
        (moved[1][0] + 0.8).abs() < 1e-5,
        "a gradient of 4 in a set of norm 5 clipped to 1 moved to {} where -0.8 is the answer",
        moved[1][0],
    );
}

#[test]
fn a_checkpoint_carries_the_optimizer_clock() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::scalar(), Init::Zero, Element::Single);
    let loss = graph.sum(graph.mul(weight, constant(&graph, 1.0)));
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, 0.1, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track(&graph, weight);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    for _ in 0..5 {
        runtime.run(&program);
    }
    let checkpoint = runtime.checkpoint(&weights);

    let resumed_graph = Graph::new();
    let resumed_weight = resumed_graph.parameter(Shape::scalar(), Init::Zero, Element::Single);
    let resumed_loss =
        resumed_graph.sum(resumed_graph.mul(resumed_weight, constant(&resumed_graph, 1.0)));
    let resumed_gradients = resumed_graph.backward(resumed_loss);
    let mut resumed_optimizer = AdamW::new(&resumed_graph, 0.1, 0.9, 0.999, 1e-8, 0.0);
    resumed_optimizer.track(&resumed_graph, resumed_weight);
    resumed_optimizer.step(&resumed_graph, &resumed_gradients);
    let weights = runtime.load(&resumed_graph, &checkpoint);
    let program = runtime.compile(&resumed_graph, &weights);
    for _ in 0..5 {
        runtime.run(&program);
    }
    let resumed = runtime.read(&program, resumed_weight)[0];

    let straight_graph = Graph::new();
    let straight_weight = straight_graph.parameter(Shape::scalar(), Init::Zero, Element::Single);
    let straight_loss =
        straight_graph.sum(straight_graph.mul(straight_weight, constant(&straight_graph, 1.0)));
    let straight_gradients = straight_graph.backward(straight_loss);
    let mut straight_optimizer = AdamW::new(&straight_graph, 0.1, 0.9, 0.999, 1e-8, 0.0);
    straight_optimizer.track(&straight_graph, straight_weight);
    straight_optimizer.step(&straight_graph, &straight_gradients);
    let weights = runtime.weights(&straight_graph);
    let program = runtime.compile(&straight_graph, &weights);
    for _ in 0..10 {
        runtime.run(&program);
    }
    let straight = runtime.read(&program, straight_weight)[0];
    assert!(
        (resumed - straight).abs() < 1e-5,
        "a resumed run kept {resumed} where ten unbroken steps say {straight}",
    );
}
