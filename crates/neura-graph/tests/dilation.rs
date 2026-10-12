use neura_abi::{Element, Kind};
use neura_graph::{Graph, Init, Pool, Shape, Window};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn refuses(action: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_dilated_window_reads_every_dilationth_tap_of_its_span() {
    let window = Window::new([3, 5], [2, 3], [1, 2], [4, 2]);
    assert_eq!(window.dilation_rows(), 4);
    assert_eq!(window.dilation_columns(), 2);
    assert_eq!(window.row_span(), 9);
    assert_eq!(window.column_span(), 9);
    let sliding = Window::dilated([3, 3], [5, 5]);
    assert_eq!(sliding.stride_rows(), 1);
    assert_eq!(sliding.pad_rows(), 0);
    assert_eq!(sliding.row_span(), 11);
    assert_eq!(Window::sliding([3, 3]).row_span(), 3);
    assert!(refuses(|| {
        let _ = Window::new([3, 3], [1, 1], [0, 0], [0, 2]);
    }));
    assert!(refuses(|| {
        let _ = Window::dilated([3, 3], [1, 0]);
    }));
}

#[test]
fn a_dilated_convolution_writes_the_size_its_span_walks() {
    let graph: Graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 9, 9]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1], [2, 2]);
    let convolved = graph.conv2d(input, filter, window);
    assert_eq!(
        graph.shape(convolved),
        Shape::of([2, 4, 4, 4]),
        "a convolution padded by one walks a span of five over nine rows in steps of two",
    );
    let tasks = graph.snapshot().tasks().to_vec();
    assert_eq!(tasks.len(), 1, "a convolution is one task");
    assert_eq!(tasks[0].kind, Kind::Conv2d);
    assert_eq!(tasks[0].window, window);
}

#[test]
fn a_dilated_transposed_convolution_undoes_the_span_it_scattered() {
    let graph: Graph = Graph::new();
    let input = graph.input(Shape::of([1, 3, 4, 4]), Element::Single);
    let filter = graph.parameter(Shape::of([3, 2, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1], [2, 2]);
    let scattered = graph.conv2d_transpose(input, filter, window, 1);
    assert_eq!(graph.shape(scattered), Shape::of([1, 2, 9, 9]));
    let convolved = graph.conv2d(scattered, filter, window);
    assert_eq!(graph.shape(convolved).dims(), [1, 3, 4, 4]);
}

#[test]
fn a_dilated_pool_writes_the_size_its_span_walks() {
    let graph: Graph = Graph::new();
    let input = graph.input(Shape::of([1, 2, 8, 8]), Element::Single);
    let window = Window::dilated([2, 2], [3, 3]);
    let pooled = graph.pool2d(input, window, Pool::Max);
    assert_eq!(graph.shape(pooled), Shape::of([1, 2, 5, 5]));
    let average = graph.pool2d(input, window, Pool::Mean);
    assert_eq!(graph.shape(average), Shape::of([1, 2, 5, 5]));
}

#[test]
fn a_dilated_window_refuses_the_sizes_its_span_outruns() {
    let graph: Graph = Graph::new();
    let filter = graph.parameter(Shape::of([2, 2, 3, 3]), Init::Zero, Element::Single);
    let narrow = graph.input(Shape::of([1, 2, 4, 6]), Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.conv2d(narrow, filter, Window::dilated([3, 3], [2, 1]));
        }),
        "a span of five rows over four rows reaches no position",
    );
    let taps = graph.parameter(Shape::of([2, 2, 5, 5]), Init::Zero, Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.conv2d(narrow, taps, Window::dilated([3, 3], [2, 1]));
        }),
        "a dilated window walks the taps it is handed, not the span it covers",
    );
}
