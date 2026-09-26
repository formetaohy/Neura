use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn strides<'g>(graph: &Graph<'g>, value: Value<'g>) -> [u32; 4] {
    graph.snapshot().values()[value.id() as usize].strides
}

#[test]
fn a_reshape_walks_the_numbers_of_its_storage_row_by_row() {
    let graph = Graph::new();
    let matrix = graph.parameter(Shape::matrix(3, 5), Init::Zero, Element::Single);
    let flattened = graph.reshape(matrix, Shape::vector(15));
    assert_eq!(flattened.shape(), Shape::vector(15));
    assert_eq!(strides(&graph, flattened), [0, 0, 0, 1]);
    let folded = graph.reshape(flattened, Shape::of([3, 1, 5, 1]));
    assert_eq!(folded.shape(), Shape::of([3, 1, 5, 1]));
    assert_eq!(strides(&graph, folded), [5, 0, 1, 0]);
}

#[test]
fn a_reshape_holds_the_numbers_of_the_tensor_it_reads() {
    let graph = Graph::new();
    let matrix = graph.parameter(Shape::matrix(3, 5), Init::Zero, Element::Single);
    assert!(refuses(|| {
        let _ = graph.reshape(matrix, Shape::vector(16));
    }));
}

#[test]
fn a_reshape_reads_a_tensor_its_storage_lays_out_row_by_row() {
    let graph = Graph::new();
    let matrix = graph.parameter(Shape::matrix(3, 5), Init::Zero, Element::Single);
    let turned = graph.permute(matrix, [0, 1, 3, 2]);
    assert!(refuses(|| {
        let _ = graph.reshape(turned, Shape::vector(15));
    }));
}

#[test]
fn a_permutation_walks_each_axis_of_its_source_once() {
    let graph = Graph::new();
    let tensor = graph.parameter(Shape::of([2, 3, 4, 5]), Init::Zero, Element::Single);
    assert!(refuses(|| {
        let _ = graph.permute(tensor, [0, 1, 2, 2]);
    }));
    assert!(refuses(|| {
        let _ = graph.permute(tensor, [0, 1, 2, 4]);
    }));
    let turned = graph.permute(tensor, [0, 1, 3, 2]);
    assert_eq!(turned.shape(), Shape::of([2, 3, 5, 4]));
    assert_eq!(strides(&graph, turned), [60, 20, 1, 5]);
    let turned_again = graph.permute(turned, [0, 1, 3, 2]);
    assert_eq!(turned_again.shape(), tensor.shape());
    assert_eq!(strides(&graph, turned_again), strides(&graph, tensor));
}
