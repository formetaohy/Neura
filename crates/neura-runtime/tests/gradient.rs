use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};

#[path = "support/input.rs"]
mod input;
#[path = "support/mod.rs"]
mod support;

use input::random;
use support::{assert_close, open};

const ROWS: u32 = 6;
const DEPTH: u32 = 4;
const COLUMNS: u32 = 3;

#[test]
fn a_gradient_input_receives_the_slope_of_every_row_it_feeds() {
    let runtime = open();
    let graph = Graph::new();
    let observations = graph.gradient_input(Shape::matrix(ROWS, DEPTH), Element::Single);
    let weight = graph.parameter(Shape::matrix(DEPTH, COLUMNS), Init::Zero, Element::Single);
    let loss = graph.sum(graph.matmul(observations, weight));
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(observations));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data = random(ROWS * DEPTH, 17);
    let kernel = random(DEPTH * COLUMNS, 23);
    runtime.write(&program, observations, &data);
    runtime.write(&program, weight, &kernel);
    runtime.run(&program);
    let mut expected = vec![0.0f32; (ROWS * DEPTH) as usize];
    for row in 0..ROWS {
        for depth in 0..DEPTH {
            let mut total = 0.0;
            for column in 0..COLUMNS {
                total += kernel[(depth * COLUMNS + column) as usize];
            }
            expected[(row * DEPTH + depth) as usize] = total;
        }
    }
    assert_close(
        &runtime.read(&program, gradients.of(observations)),
        &expected,
        1e-5,
    );
}

#[test]
fn a_gradient_input_walks_the_slope_the_device_descends() {
    let runtime = open();
    let graph = Graph::new();
    let observations = graph.gradient_input(Shape::matrix(ROWS, DEPTH), Element::Single);
    let target = graph.input(Shape::matrix(ROWS, COLUMNS), Element::Single);
    let weight = graph.parameter(Shape::matrix(DEPTH, COLUMNS), Init::Zero, Element::Single);
    let prediction = graph.matmul(observations, weight);
    let difference = graph.sub(prediction, target);
    let loss = graph.sum(graph.mul(difference, difference));
    let gradients = graph.backward(loss);
    let rate = graph.fill(Shape::scalar(), -1e-4);
    graph.add_into(observations, graph.mul(gradients.of(observations), rate));
    graph.retain(loss);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data = random(ROWS * DEPTH, 29);
    let kernel = random(DEPTH * COLUMNS, 31);
    let target_data = random(ROWS * COLUMNS, 37);
    runtime.write(&program, observations, &data);
    runtime.write(&program, weight, &kernel);
    runtime.write(&program, target, &target_data);
    runtime.run(&program);
    let before = runtime.read(&program, loss)[0];
    runtime.run(&program);
    let after = runtime.read(&program, loss)[0];
    assert!(
        after < before,
        "a step against the slope lowered the loss from {before} to {after}",
    );
    let moved = runtime.read(&program, observations);
    assert_ne!(moved, data, "the descent moved the observation it followed");
}
