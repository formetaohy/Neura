use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_nn::{AdamW, Sgd};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

const STEPS: usize = 4;
const GRADIENT: f32 = 0.5;

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

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{what} came back as {} numbers where {} were expected",
        actual.len(),
        expected.len(),
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what} element {index} came back as {actual} where {expected} was expected",
        );
    }
}

fn rate_of(step: usize) -> f32 {
    match step {
        0 => 0.4,
        1 => 0.2,
        2 => 0.1,
        _ => 0.05,
    }
}

fn adam_reference(rates: &[f32], weight_decay: f32) -> Vec<[f32; 2]> {
    let (mean_decay, variance_decay, floor) = (0.9f32, 0.999f32, 1e-8f32);
    let mut parameter = [1.0f32; 2];
    let mut mean = [0.0f32; 2];
    let mut variance = [0.0f32; 2];
    let mut walked = Vec::new();
    for (step, rate) in rates.iter().enumerate() {
        let mean_scale = 1.0 / (1.0 - mean_decay.powf(step as f32 + 1.0));
        let variance_scale = 1.0 / (1.0 - variance_decay.powf(step as f32 + 1.0));
        for index in 0..2 {
            mean[index] = mean_decay * mean[index] + (1.0 - mean_decay) * GRADIENT;
            variance[index] =
                variance_decay * variance[index] + GRADIENT * GRADIENT * (1.0 - variance_decay);
            let corrected_mean = mean[index] * mean_scale;
            let corrected_variance = variance[index] * variance_scale;
            let scaled = corrected_mean * (1.0 / (corrected_variance.sqrt() + floor));
            let descent = -rate;
            let mut update = scaled * descent;
            if weight_decay != 0.0 {
                update += (parameter[index] * weight_decay) * descent;
            }
            parameter[index] += update;
        }
        walked.push(parameter);
    }
    walked
}

#[test]
fn a_written_rate_steers_a_bias_corrected_adam() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_parameter(
        "weight",
        Shape::vector(2),
        Init::Constant(1.0),
        Element::Single,
    );
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(2), GRADIENT)));
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", rate_of(0), 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&graph, &[weight]);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let mut produced = Vec::new();
    for step in 0..STEPS {
        runtime.write(&program, optimizer.rate(), &[rate_of(step)]);
        runtime.run(&program);
        produced.push(runtime.read(&program, weight));
    }

    let scheduled = adam_reference(&[rate_of(0), rate_of(1), rate_of(2), rate_of(3)], 0.0);
    for (step, expected) in scheduled.iter().enumerate() {
        assert_close(
            &produced[step],
            expected,
            1e-4,
            &format!("the step a written rate of {} steered", rate_of(step)),
        );
    }
    let constant = adam_reference(&[rate_of(0); STEPS], 0.0);
    assert!(
        (produced[STEPS - 1][0] - constant[STEPS - 1][0]).abs() > 1e-2,
        "a schedule steers the descent every step: a written rate of {} left {} where the rate the descent was built with leaves {}",
        rate_of(STEPS - 1),
        produced[STEPS - 1][0],
        constant[STEPS - 1][0],
    );
    assert!(
        program.is_current(),
        "writing a rate keeps the program current"
    );
    assert_eq!(
        runtime.built_plans(),
        1,
        "a schedule steers one plan rather than building a plan per step",
    );
}

#[test]
fn a_descent_schedules_its_own_rate_on_the_device() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_parameter(
        "weight",
        Shape::vector(2),
        Init::Constant(1.0),
        Element::Single,
    );
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(2), GRADIENT)));
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", 0.8, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&graph, &[weight]);
    graph.mul_into(optimizer.rate(), graph.fill(Shape::scalar(), 0.5));
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let mut produced = Vec::new();
    for _ in 0..STEPS {
        runtime.run(&program);
        produced.push(runtime.read(&program, weight));
    }

    let halved = [rate_of(0), rate_of(1), rate_of(2), rate_of(3)];
    let scheduled = adam_reference(&halved, 0.0);
    for (step, expected) in scheduled.iter().enumerate() {
        assert_close(
            &produced[step],
            expected,
            1e-4,
            &format!("the step a device halved the rate to {} for", halved[step]),
        );
    }
    assert_eq!(
        runtime.read(&program, optimizer.rate()),
        [0.8f32 * 0.5f32.powi(STEPS as i32)],
        "a device that halves the rate every step walks it down to the number it read last",
    );
}

#[test]
fn a_written_momentum_steers_a_momentum_descent() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_parameter("weight", Shape::vector(2), Init::Zero, Element::Single);
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(2), 1.0)));
    let gradients = graph.backward(loss);
    let mut optimizer = Sgd::momentum(&graph, "descent", 0.1, 0.9, 0.0);
    optimizer.track_all(&graph, &[weight]);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let momentum_of = |step: usize| match step {
        0 => 0.9,
        1 => 0.5,
        _ => 0.0,
    };
    for step in 0..STEPS {
        runtime.write(
            &program,
            optimizer
                .momentum_decay()
                .expect("a momentum carries its decay"),
            &[momentum_of(step)],
        );
        runtime.run(&program);
    }
    let produced = runtime.read(&program, weight);

    let mut velocity = [0.0f32; 2];
    let mut reference = [0.0f32; 2];
    for step in 0..STEPS {
        for index in 0..2 {
            velocity[index] = momentum_of(step) * velocity[index] + 1.0;
            reference[index] -= 0.1 * velocity[index];
        }
    }
    assert_close(&produced, &reference, 1e-6, "a written momentum");
}

#[test]
fn every_knob_a_descent_reads_is_a_state_the_host_writes() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_parameter(
        "weight",
        Shape::vector(2),
        Init::Constant(1.0),
        Element::Single,
    );
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(2), GRADIENT)));
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", 0.1, 0.9, 0.999, 1e-8, 0.1);
    optimizer.track_all(&graph, &[weight]);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let knobs_of = |step: usize| match step {
        0 => (0.9f32, 0.999f32, 1e-8f32, 0.1f32),
        1 => (0.5, 0.9, 1e-4, 0.05),
        _ => (0.0, 0.5, 1e-3, 0.0),
    };
    for step in 0..STEPS {
        let (mean_decay, variance_decay, floor, weight_decay) = knobs_of(step);
        for (knob, value) in [
            (optimizer.mean_decay(), mean_decay),
            (optimizer.variance_decay(), variance_decay),
            (optimizer.floor(), floor),
            (
                optimizer.weight_decay().expect("a decay carries its state"),
                weight_decay,
            ),
        ] {
            runtime.write(&program, knob, &[value]);
        }
        runtime.run(&program);
    }
    let produced = runtime.read(&program, weight);

    let rate = 0.1f32;
    let mut reference = [1.0f32; 2];
    let mut mean = [0.0f32; 2];
    let mut variance = [0.0f32; 2];
    for (step, (mean_decay, variance_decay, floor, weight_decay)) in
        (0..STEPS).map(knobs_of).enumerate()
    {
        let mean_scale = 1.0 / (1.0 - mean_decay.powf(step as f32 + 1.0));
        let variance_scale = 1.0 / (1.0 - variance_decay.powf(step as f32 + 1.0));
        for index in 0..2 {
            mean[index] = mean_decay * mean[index] + (1.0 - mean_decay) * GRADIENT;
            variance[index] =
                variance_decay * variance[index] + GRADIENT * GRADIENT * (1.0 - variance_decay);
            let corrected_mean = mean[index] * mean_scale;
            let corrected_variance = variance[index] * variance_scale;
            let scaled = corrected_mean * (1.0 / (corrected_variance.sqrt() + floor));
            let descent = -rate;
            let mut update = scaled * descent;
            if weight_decay != 0.0 {
                update += (reference[index] * weight_decay) * descent;
            }
            reference[index] += update;
        }
    }
    assert_close(&produced, &reference, 1e-4, "a descent of written knobs");
}

#[test]
fn a_descent_names_every_knob_it_reads() {
    let graph = Graph::new();
    let weight = graph.named_parameter("weight", Shape::vector(2), Init::Zero, Element::Single);
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(2), 1.0)));
    let _ = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", 0.1, 0.9, 0.999, 1e-8, 0.1);
    optimizer.track(&graph, weight);
    let knobs = [
        ("optimizer.rate", optimizer.rate()),
        ("optimizer.mean_decay", optimizer.mean_decay()),
        ("optimizer.variance_decay", optimizer.variance_decay()),
        ("optimizer.floor", optimizer.floor()),
        (
            "optimizer.weight_decay",
            optimizer.weight_decay().expect("a decay carries its state"),
        ),
    ];
    for (name, knob) in knobs {
        assert_eq!(
            graph.name_of(knob).as_deref(),
            Some(name),
            "a knob of a descent is the training state its name points at",
        );
        assert!(
            graph.shape(knob).is_scalar(),
            "{name} holds one number a host writes",
        );
        assert!(
            !graph.trains(knob),
            "{name} steers a step rather than learning from one",
        );
    }
    assert!(
        refuses(|| {
            AdamW::new(&graph, "optimizer", 0.1, 0.9, 0.999, 1e-8, 0.0);
        }),
        "two descents of one graph share no name",
    );
    let mut momentum = Sgd::momentum(&graph, "momentum", 0.1, 0.9, 0.0);
    momentum.track(&graph, weight);
    assert_eq!(
        graph
            .name_of(momentum.momentum_decay().expect("a momentum"))
            .as_deref(),
        Some("momentum.momentum_decay"),
    );
}

#[test]
fn a_checkpoint_carries_the_rate_a_descent_stopped_at() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_parameter(
        "weight",
        Shape::vector(2),
        Init::Constant(1.0),
        Element::Single,
    );
    let loss = graph.sum(graph.mul(weight, graph.fill(Shape::vector(2), GRADIENT)));
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", 0.4, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&graph, &[weight]);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    for step in 0..2 {
        runtime.write(&program, optimizer.rate(), &[rate_of(step)]);
        runtime.run(&program);
    }
    let checkpoint = runtime.checkpoint(&weights);
    assert!(
        checkpoint.tensor("optimizer.rate").is_some(),
        "a container carries the rate a descent stepped at",
    );

    let resumed_graph = Graph::new();
    let resumed_weight = resumed_graph.named_parameter(
        "weight",
        Shape::vector(2),
        Init::Constant(1.0),
        Element::Single,
    );
    let resumed_loss = resumed_graph.sum(resumed_graph.mul(
        resumed_weight,
        resumed_graph.fill(Shape::vector(2), GRADIENT),
    ));
    let resumed_gradients = resumed_graph.backward(resumed_loss);
    let mut resumed_optimizer = AdamW::new(&resumed_graph, "optimizer", 0.4, 0.9, 0.999, 1e-8, 0.0);
    resumed_optimizer.track_all(&resumed_graph, &[resumed_weight]);
    resumed_optimizer.step(&resumed_graph, &resumed_gradients);
    let resumed_weights = runtime.load(&resumed_graph, &checkpoint);
    let resumed_program = runtime.compile(&resumed_graph, &resumed_weights);
    assert_eq!(
        runtime.read(&resumed_program, resumed_optimizer.rate()),
        [rate_of(1)],
        "a descent resumed from a container steps at the rate it stopped at",
    );

    for step in 2..STEPS {
        runtime.write(&program, optimizer.rate(), &[rate_of(step)]);
        runtime.run(&program);
        runtime.write(&resumed_program, resumed_optimizer.rate(), &[rate_of(step)]);
        runtime.run(&resumed_program);
    }
    assert_close(
        &runtime.read(&resumed_program, resumed_weight),
        &runtime.read(&program, weight),
        1e-6,
        "a descent that resumed from the rate it stopped at",
    );
}
