use neura_program::{Graph, Init, Shape, Value};
use neura_runtime::{Checkpoint, Precision};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

struct Model<'g> {
    observations: Value<'g>,
    prediction: Value<'g>,
}

fn network<'g>(graph: &Graph<'g>, hidden: u32) -> Model<'g> {
    let observations = graph.input(Shape::matrix(8, 4));
    let first = graph.parameter(
        Shape::matrix(4, hidden),
        Init::Uniform {
            low: -0.25,
            high: 0.25,
        },
    );
    let second = graph.parameter(
        Shape::matrix(hidden, 2),
        Init::Uniform {
            low: -0.25,
            high: 0.25,
        },
    );
    let prediction = graph.matmul(graph.relu(graph.matmul(observations, first)), second);
    graph.retain(prediction);
    Model {
        observations,
        prediction,
    }
}

fn batch(seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    let mut next = move || {
        entropy = entropy.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (entropy >> 8) as f32 / 16_777_216.0 - 0.5
    };
    (0..8 * 4).map(|_| next()).collect()
}

#[test]
fn a_checkpoint_outlives_its_runtime() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, 5);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let observations = batch(3);
    runtime.write(&program, model.observations, &observations);
    for _ in 0..40 {
        runtime.run(&program);
    }
    let prediction = runtime.read(&program, model.prediction);
    let checkpoint = runtime.checkpoint(&weights);

    let another = open();
    let rebuilt = Graph::new();
    let model = network(&rebuilt, 5);
    let weights = another.load(&rebuilt, &checkpoint, Precision::Single);
    let program = another.compile(&rebuilt, &weights);
    another.write(&program, model.observations, &observations);
    another.run(&program);
    assert_close(&another.read(&program, model.prediction), &prediction, 1e-6);
}

#[test]
fn a_store_restores_its_trained_parameters() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, 5);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let observations = batch(7);
    runtime.write(&program, model.observations, &observations);
    runtime.run(&program);
    let fresh = runtime.read(&program, model.prediction);
    let initial = runtime.checkpoint(&weights);
    for _ in 0..40 {
        runtime.run(&program);
    }
    let prediction = runtime.read(&program, model.prediction);
    let checkpoint = runtime.checkpoint(&weights);

    runtime.restore(&weights, &initial);
    runtime.run(&program);
    assert_close(&runtime.read(&program, model.prediction), &fresh, 1e-6);
    runtime.restore(&weights, &checkpoint);
    runtime.run(&program);
    assert_close(&runtime.read(&program, model.prediction), &prediction, 1e-6);
}

#[test]
fn a_checkpoint_of_another_region_is_refused() {
    let runtime = open();
    let graph = Graph::new();
    let _model = network(&graph, 5);
    let checkpoint = runtime.checkpoint(&runtime.weights(&graph, Precision::Single));

    let wider = Graph::new();
    let _wider = network(&wider, 6);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.load(&wider, &checkpoint, Precision::Single);
    }))
    .expect_err("a store loads only a checkpoint of the parameters its graph declares");

    let truncated = &checkpoint.bytes()[..checkpoint.bytes().len() - 4];
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Checkpoint::decode(truncated);
    }))
    .expect_err("a checkpoint decodes only bytes its header accounts for");
}

#[test]
fn a_half_store_round_trips_bit_for_bit() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, 5);
    let weights = runtime.weights(&graph, Precision::Half);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, model.observations, &batch(11));
    runtime.run(&program);
    let before = runtime.read(&program, model.prediction);
    let checkpoint = runtime.checkpoint(&weights);
    assert_eq!(checkpoint.precision(), Precision::Half);

    let reloaded = runtime.load(&graph, &checkpoint, Precision::Half);
    runtime.restore(&reloaded, &checkpoint);
    let program = runtime.compile(&graph, &reloaded);
    runtime.write(&program, model.observations, &batch(11));
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, model.prediction),
        before,
        "a half store pours back the very words it was saved from",
    );
}
