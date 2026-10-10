use neura_abi::{Element, Kind};
use neura_graph::{Graph, Init, Shape, Window};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn refuses(action: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_transposed_convolution_writes_the_channels_and_size_its_filter_undoes() {
    let graph: Graph<'static> = Graph::new();
    let input = graph.input(Shape::of([2, 4, 5, 5]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1]);
    let scattered = graph.conv2d_transpose(input, filter, window, 2);
    assert_eq!(
        graph.shape(scattered),
        Shape::of([2, 6, 9, 9]),
        "a transposed convolution writes the channels its groups name and the size its stride, taps and padding undo",
    );
    let tasks = graph.snapshot().tasks().to_vec();
    assert_eq!(tasks.len(), 1, "a transposed convolution is one task");
    assert_eq!(tasks[0].kind, Kind::Conv2dTranspose);
    assert_eq!(tasks[0].inputs[0], filter.id());
    assert_eq!(tasks[0].inputs[1], input.id());
    assert_eq!(tasks[0].out, scattered.id());
    assert_eq!(tasks[0].window, window);
}

#[test]
fn a_free_batch_of_a_transposed_convolution_walks_the_extent_of_its_input() {
    let graph: Graph<'static> = Graph::new();
    let batch = graph.free(2);
    let input = graph.input(
        Shape::of([2, 4, 5, 5]).freed(&[(0, batch)]),
        Element::Single,
    );
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    let scattered = graph.conv2d_transpose(input, filter, Window::new([3, 3], [1, 1], [1, 1]), 1);
    assert_eq!(graph.shape(scattered).free(0), Some(batch.slot()));
    assert_eq!(graph.shape(scattered).dims(), [2, 3, 5, 5]);
}

#[test]
fn a_transposed_convolution_refuses_the_shapes_it_cannot_scatter() {
    let graph: Graph<'static> = Graph::new();
    let input = graph.input(Shape::of([1, 4, 5, 5]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.conv2d_transpose(input, filter, Window::sliding([3, 3]), 0);
        }),
        "a transposed convolution of no group scatters no channel",
    );
    assert!(
        refuses(|| {
            let _ = graph.conv2d_transpose(input, filter, Window::sliding([3, 3]), 8);
        }),
        "a transposed convolution of eight groups scatters four channels",
    );
    let channels = graph.parameter(Shape::of([3, 3, 3, 3]), Init::Zero, Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.conv2d_transpose(input, channels, Window::sliding([3, 3]), 1);
        }),
        "a transposed convolution scatters the channels of the filter it is handed",
    );
    let taps = graph.parameter(Shape::of([4, 3, 2, 2]), Init::Zero, Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.conv2d_transpose(input, taps, Window::sliding([3, 3]), 1);
        }),
        "a transposed convolution walks the taps of the window it is handed",
    );
    let padded = Window::new([3, 3], [1, 1], [2, 2]);
    let narrow = graph.input(Shape::of([1, 4, 1, 1]), Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.conv2d_transpose(narrow, filter, padded, 1);
        }),
        "a transposed convolution padded beyond its size scatters into no position",
    );
    let free = graph.input(
        Shape::of([4, 1, 2, 5]).freed(&[(1, graph.free(1))]),
        Element::Single,
    );
    assert!(
        refuses(|| {
            let _ = graph.conv2d_transpose(input, free, Window::sliding([2, 5]), 1);
        }),
        "a transposed convolution scatters no channel a binding rules",
    );
}
