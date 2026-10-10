use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Window};

#[path = "support/convolution.rs"]
mod convolution;
#[path = "support/mod.rs"]
mod support;
#[path = "support/transpose.rs"]
mod transpose;

use convolution::conv2d_reference;
use support::{assert_close, open};
use transpose::{conv2d_transpose_reference, conv2d_transpose_weight_grad_reference};

fn samples(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}

#[test]
fn a_convolution_walks_the_window_it_is_handed() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 6, 6]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    let bias = graph.parameter(Shape::of([1, 4, 1, 1]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1]);
    let convolved = graph.conv2d(input, filter, window);
    let (rows, columns, channels) = (3usize, 3usize, 4usize);
    assert_eq!(convolved.shape(), Shape::of([2, 4, 3, 3]));
    let activated = graph.relu(graph.add(convolved, bias));
    graph.retain(activated);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = samples(216, 5);
    let filter_values = samples(108, 11);
    let bias_values = samples(4, 17);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.write(&program, bias, &bias_values);
    runtime.run(&program);
    let expected = conv2d_reference(&input_values, &filter_values, [2, 3, 6, 6], 4, window)
        .iter()
        .enumerate()
        .map(|(index, value)| (value + bias_values[index / (rows * columns) % channels]).max(0.0))
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, activated), &expected, 1e-4);
}

#[test]
fn a_convolution_leaves_the_row_its_window_never_reaches() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([3, 2, 7, 6]), Element::Single);
    let filter = graph.parameter(Shape::of([3, 2, 3, 2]), Init::Zero, Element::Single);
    let window = Window::new([3, 2], [3, 3], [0, 0]);
    let convolved = graph.conv2d(input, filter, window);
    assert_eq!(convolved.shape(), Shape::of([3, 3, 2, 2]));
    graph.retain(convolved);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = samples(252, 23);
    let filter_values = samples(36, 29);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.run(&program);
    let expected = conv2d_reference(&input_values, &filter_values, [3, 2, 7, 6], 3, window);
    assert_close(&runtime.read(&program, convolved), &expected, 1e-4);
}

#[test]
fn a_depthwise_convolution_reads_one_channel_at_a_time() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 4, 6, 6]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 1, 3, 3]), Init::Zero, Element::Single);
    let window = Window::sliding([3, 3]);
    let convolved = graph.conv2d(input, filter, window);
    assert_eq!(convolved.shape(), Shape::of([2, 4, 4, 4]));
    graph.retain(convolved);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = samples(288, 43);
    let filter_values = samples(36, 47);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.run(&program);
    let expected = conv2d_reference(&input_values, &filter_values, [2, 4, 6, 6], 4, window);
    assert_close(&runtime.read(&program, convolved), &expected, 1e-4);
}

#[test]
fn a_grouped_convolution_cuts_the_channels_it_reads() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 4, 7, 7]), Element::Single);
    let filter = graph.parameter(Shape::of([6, 2, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1]);
    let convolved = graph.conv2d(input, filter, window);
    assert_eq!(convolved.shape(), Shape::of([2, 6, 4, 4]));
    graph.retain(convolved);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = samples(392, 71);
    let filter_values = samples(108, 73);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.run(&program);
    let expected = conv2d_reference(&input_values, &filter_values, [2, 4, 7, 7], 6, window);
    assert_close(&runtime.read(&program, convolved), &expected, 1e-4);
}

#[test]
fn two_windows_ride_one_tape() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 8, 8]), Element::Single);
    let narrow = graph.parameter(Shape::of([2, 3, 3, 3]), Init::Zero, Element::Single);
    let wide = graph.parameter(Shape::of([2, 3, 5, 5]), Init::Zero, Element::Single);
    let sliding = Window::sliding([3, 3]);
    let padded = Window::new([5, 5], [1, 1], [1, 1]);
    let first = graph.conv2d(input, narrow, sliding);
    let second = graph.conv2d(input, wide, padded);
    assert_eq!(first.shape(), Shape::of([2, 2, 6, 6]));
    assert_eq!(second.shape(), Shape::of([2, 2, 6, 6]));
    let combined = graph.add(first, second);
    graph.retain(combined);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(runtime.declared_kernels(), 1);
    let input_values = samples(384, 31);
    let narrow_values = samples(54, 37);
    let wide_values = samples(150, 41);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, narrow, &narrow_values);
    runtime.write(&program, wide, &wide_values);
    runtime.run(&program);
    let mut expected = conv2d_reference(&input_values, &narrow_values, [2, 3, 8, 8], 2, sliding);
    let wider = conv2d_reference(&input_values, &wide_values, [2, 3, 8, 8], 2, padded);
    for (value, addition) in expected.iter_mut().zip(wider) {
        *value += addition;
    }
    assert_close(&runtime.read(&program, combined), &expected, 1e-4);
}

#[test]
fn a_transposed_convolution_scatters_the_window_it_is_handed() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 6, 6]), Element::Single);
    let filter = graph.parameter(Shape::of([3, 4, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1]);
    let scattered = graph.conv2d_transpose(input, filter, window, 1);
    assert_eq!(scattered.shape(), Shape::of([2, 4, 11, 11]));
    graph.retain(scattered);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = samples(216, 5);
    let filter_values = samples(108, 11);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.run(&program);
    let expected =
        conv2d_transpose_reference(&input_values, &filter_values, [2, 3, 6, 6], 1, window);
    assert_close(&runtime.read(&program, scattered), &expected, 1e-4);
}

#[test]
fn a_grouped_transposed_convolution_writes_each_group_its_own_channels() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 4, 5, 5]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1]);
    let scattered = graph.conv2d_transpose(input, filter, window, 2);
    assert_eq!(scattered.shape(), Shape::of([1, 6, 9, 9]));
    graph.retain(scattered);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = samples(100, 73);
    let filter_values = samples(108, 79);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.run(&program);
    let expected =
        conv2d_transpose_reference(&input_values, &filter_values, [1, 4, 5, 5], 2, window);
    assert_close(&runtime.read(&program, scattered), &expected, 1e-4);
}

#[test]
fn a_depthwise_transposed_convolution_writes_one_channel_at_a_time() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 4, 6, 6]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 1, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [1, 1], [1, 1]);
    let scattered = graph.conv2d_transpose(input, filter, window, 4);
    assert_eq!(scattered.shape(), Shape::of([2, 4, 6, 6]));
    graph.retain(scattered);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = samples(288, 83);
    let filter_values = samples(36, 89);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.run(&program);
    let expected =
        conv2d_transpose_reference(&input_values, &filter_values, [2, 4, 6, 6], 4, window);
    assert_close(&runtime.read(&program, scattered), &expected, 1e-4);
}

#[test]
fn a_transposed_convolution_walks_its_gradient_back_into_the_input_and_filter() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.gradient_input(Shape::of([1, 3, 4, 4]), Element::Single);
    let filter = graph.parameter(Shape::of([3, 2, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1]);
    let scattered = graph.conv2d_transpose(input, filter, window, 1);
    assert_eq!(scattered.shape(), Shape::of([1, 2, 7, 7]));
    let upstream = graph.parameter(Shape::of([1, 2, 7, 7]), Init::Zero, Element::Single);
    let loss = graph.sum(graph.mul(scattered, upstream));
    let gradients = graph.backward(loss);
    let input_grad = gradients.of(input);
    let filter_grad = gradients.of(filter);
    graph.retain(input_grad);
    graph.retain(filter_grad);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = samples(48, 97);
    let filter_values = samples(54, 101);
    let upstream_values = samples(98, 103);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.write(&program, upstream, &upstream_values);
    runtime.run(&program);
    let expected_input =
        conv2d_reference(&upstream_values, &filter_values, [1, 2, 7, 7], 3, window);
    let expected_filter = conv2d_transpose_weight_grad_reference(
        &input_values,
        [1, 3, 4, 4],
        &upstream_values,
        [1, 2, 7, 7],
        1,
        window,
    );
    assert_close(&runtime.read(&program, input_grad), &expected_input, 1e-4);
    assert_close(&runtime.read(&program, filter_grad), &expected_filter, 1e-4);
}
