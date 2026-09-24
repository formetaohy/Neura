use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::Precision;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

struct Model<'g> {
    observations: Value<'g>,
    targets: Value<'g>,
    prediction: Value<'g>,
    loss: Value<'g>,
    parameters: [Value<'g>; 2],
}

fn trained<'g>(graph: &Graph<'g>, samples: u32, hidden: u32) -> Model<'g> {
    let observations = graph.input(Shape::matrix(samples, 4));
    let targets = graph.input(Shape::matrix(samples, 2));
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
    let difference = graph.sub(prediction, targets);
    let loss = graph.mul(
        graph.sum(graph.mul(difference, difference)),
        graph.fill(Shape::scalar(), 1.0 / (2 * samples) as f32),
    );
    let gradients = graph.backward(loss);
    let descent = graph.fill(Shape::scalar(), -0.05);
    for parameter in [first, second] {
        graph.add_into(parameter, graph.mul(gradients.of(parameter), descent));
    }
    Model {
        observations,
        targets,
        prediction,
        loss,
        parameters: [first, second],
    }
}

fn batch(samples: u32, seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut entropy = seed | 1;
    let mut next = move || {
        entropy = entropy.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (entropy >> 8) as f32 / 16_777_216.0 - 0.5
    };
    let observations = (0..4 * samples).map(|_| next() * 0.5 + 0.5).collect();
    let targets = (0..2 * samples).map(|_| next()).collect();
    (observations, targets)
}

fn forward(
    observations: &[f32],
    first: &[f32],
    second: &[f32],
    samples: u32,
    hidden: u32,
) -> Vec<f32> {
    let mut middle = vec![0.0f32; (samples * hidden) as usize];
    for row in 0..samples {
        for column in 0..hidden {
            let mut total = 0.0;
            for depth in 0..4 {
                total += observations[(row * 4 + depth) as usize]
                    * first[(depth * hidden + column) as usize];
            }
            middle[(row * hidden + column) as usize] = total.max(0.0);
        }
    }
    let mut out = Vec::with_capacity((samples * 2) as usize);
    for row in 0..samples {
        for column in 0..2 {
            let mut total = 0.0;
            for depth in 0..hidden {
                total +=
                    middle[(row * hidden + depth) as usize] * second[(depth * 2 + column) as usize];
            }
            out.push(total);
        }
    }
    out
}

#[test]
fn a_rebound_store_carries_its_training_across_graphs() {
    let runtime = open();
    let graph = Graph::new();
    let model = trained(&graph, 16, 5);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let (small_observations, small_targets) = batch(16, 3);
    runtime.write(&program, model.observations, &small_observations);
    runtime.write(&program, model.targets, &small_targets);
    for _ in 0..150 {
        runtime.run(&program);
    }
    let trained_first = runtime.read(&program, model.parameters[0]);

    let rebuilt = Graph::new();
    let model = trained(&rebuilt, 64, 5);
    runtime.rebind(&weights, &rebuilt);
    let program = runtime.compile(&rebuilt, &weights);
    assert_eq!(
        runtime.read(&program, model.parameters[0]),
        trained_first,
        "a rebound store hands the rebuilt graph the weights the first graph trained",
    );

    let (large_observations, large_targets) = batch(64, 7);
    let first_now = runtime.read(&program, model.parameters[0]);
    let second_now = runtime.read(&program, model.parameters[1]);
    runtime.write(&program, model.observations, &large_observations);
    runtime.write(&program, model.targets, &large_targets);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, model.prediction),
        &forward(&large_observations, &first_now, &second_now, 64, 5),
        1e-5,
    );
    let before = runtime.read(&program, model.loss)[0];
    for _ in 0..150 {
        runtime.run(&program);
    }
    let after = runtime.read(&program, model.loss)[0];
    assert!(
        after < before && after.is_finite(),
        "training resumes on the rebuilt graph: the loss sits at {before} and ends at {after}",
    );
}

#[test]
fn rebind_rejects_a_foreign_parameter_region() {
    let runtime = open();
    let graph = Graph::new();
    let _model = trained(&graph, 16, 5);
    let weights = runtime.weights(&graph, Precision::Single);
    let foreign = Graph::new();
    let _model = trained(&foreign, 16, 6);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.rebind(&weights, &foreign);
    }))
    .expect_err("a store rebinds only onto a graph that declares the same parameters");
}

#[test]
fn tuning_leaves_the_parameter_store_untouched() {
    let runtime = open();
    let graph = Graph::new();
    let model = trained(&graph, 8, 5);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let (observations, targets) = batch(8, 5);
    runtime.write(&program, model.observations, &observations);
    runtime.write(&program, model.targets, &targets);
    runtime.run(&program);
    let before = runtime.read(&program, model.parameters[0]);

    let tuned = runtime.tune(&graph, &weights);
    let after = runtime.read(&program, model.parameters[0]);
    assert_eq!(
        before, after,
        "the tuning runs measure a scratch store, not the one the model trains on",
    );
    assert!(
        runtime.profiles().contains(&tuned.profile()),
        "a tuned program carries a profile the device offers",
    );
}

#[test]
fn one_kernel_serves_every_batch() {
    let runtime = open();
    assert_eq!(runtime.assembled_kernels(), 0);
    for samples in [8, 32, 96] {
        let graph = Graph::new();
        let model = trained(&graph, samples, 5);
        let weights = runtime.weights(&graph, Precision::Single);
        let program = runtime.compile(&graph, &weights);
        let (observations, targets) = batch(samples, samples);
        runtime.write(&program, model.observations, &observations);
        runtime.write(&program, model.targets, &targets);
        runtime.run(&program);
        assert!(runtime.read(&program, model.loss)[0].is_finite());
    }
    assert_eq!(
        runtime.assembled_kernels(),
        1,
        "every batch of one model runs the device program its task kinds name once",
    );

    let graph = Graph::new();
    let model = trained(&graph, 128, 5);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let (observations, targets) = batch(128, 128);
    runtime.write(&program, model.observations, &observations);
    runtime.write(&program, model.targets, &targets);
    runtime.run(&program);
    assert!(runtime.read(&program, model.loss)[0].is_finite());
    assert_eq!(
        runtime.assembled_kernels(),
        2,
        "a batch whose depth splits assembles the one program its extra task kind names",
    );
}
