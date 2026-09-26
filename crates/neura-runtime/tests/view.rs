use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Window};

#[path = "support/convolution.rs"]
mod convolution;
#[path = "support/reference.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use convolution::conv2d_reference;
use reference::{matmul_reference, random};
use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_convolution_flows_into_a_product_through_a_reshape() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 2, 4, 4]), Element::Single);
    let filter = graph.parameter(Shape::of([3, 2, 2, 2]), Init::Zero, Element::Single);
    let weight = graph.parameter(Shape::matrix(27, 5), Init::Zero, Element::Single);
    let features = graph.conv2d(input, filter, Window::sliding([2, 2]));
    let rows = graph.reshape(features, Shape::matrix(1, 27));
    let logits = graph.matmul(rows, weight);
    graph.retain(logits);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = random(32, 3);
    let filter_values = random(24, 5);
    let weight_values = random(135, 7);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.write(&program, weight, &weight_values);
    runtime.run(&program);
    let features_values = conv2d_reference(
        &input_values,
        &filter_values,
        [1, 2, 4, 4],
        3,
        Window::sliding([2, 2]),
    );
    let expected = matmul_reference(&features_values, &weight_values, 1, 27, 5);
    assert_close(&runtime.read(&program, logits), &expected, 1e-4);
}

#[test]
fn a_view_that_its_storage_addresses_row_by_row_reads_back_as_that_storage() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 4, 4]), Element::Single);
    let doubled = graph.mul(input, input);
    let flattened = graph.reshape(doubled, Shape::matrix(2, 48));
    graph.retain(flattened);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(96, 11);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let expected = values.iter().map(|value| value * value).collect::<Vec<_>>();
    assert_close(&runtime.read(&program, flattened), &expected, 1e-6);
    assert_close(&runtime.read(&program, doubled), &expected, 1e-6);
}

#[test]
fn a_permuted_view_reads_back_through_the_axes_it_names() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 1, 3, 5]), Element::Single);
    let squared = graph.mul(input, input);
    let turned = graph.permute(squared, [0, 1, 3, 2]);
    let rows = graph.sum_rows(turned);
    graph.retain(rows);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(15, 13);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let mut expected = [0.0f32; 5];
    for column in 0..5 {
        for row in 0..3 {
            let value = values[row * 5 + column];
            expected[column] += value * value;
        }
    }
    assert_close(&runtime.read(&program, rows), &expected, 1e-5);
    assert!(
        refuses(|| {
            let _ = runtime.read(&program, turned);
        }),
        "a permuted view was read back as the storage it walks",
    );
}

#[test]
fn a_gradient_walks_back_through_a_reshape() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 2), Init::Zero, Element::Single);
    let observations = graph.input(Shape::matrix(4, 2), Element::Single);
    let loss = graph.sum(graph.mul(
        graph.reshape(weight, Shape::vector(8)),
        graph.reshape(observations, Shape::vector(8)),
    ));
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(weight));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations_values = random(8, 17);
    runtime.write(&program, observations, &observations_values);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, gradients.of(weight)),
        &observations_values,
        1e-6,
    );
}

#[test]
fn a_gradient_walks_back_through_a_permutation() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(2, 5), Init::Zero, Element::Single);
    let observations = graph.input(Shape::matrix(5, 2), Element::Single);
    let loss = graph.sum(graph.mul(graph.permute(weight, [0, 1, 3, 2]), observations));
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(weight));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations_values = random(10, 19);
    runtime.write(&program, observations, &observations_values);
    runtime.run(&program);
    let turned = (0..10)
        .map(|index| {
            let row = index / 5;
            let column = index % 5;
            observations_values[column * 2 + row]
        })
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, gradients.of(weight)), &turned, 1e-6);
}
