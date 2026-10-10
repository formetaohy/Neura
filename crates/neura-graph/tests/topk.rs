use neura_abi::{CANDIDATES, Element, Kind, NO_VALUE};
use neura_graph::{Graph, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_top_k_asks_for_the_candidates_it_keeps() {
    let graph = Graph::new();
    let value = graph.input(Shape::matrix(4, 8), Element::Single);
    let (kept_values, kept_indices) = graph.top_k(value, 3);
    assert_eq!(graph.shape(kept_values), Shape::matrix(4, 3));
    assert_eq!(graph.shape(kept_indices), Shape::matrix(4, 3));
    assert_eq!(graph.element(kept_values), Element::Single);
    assert_eq!(graph.element(kept_indices), Element::Single);
    assert!(
        !graph.trains(kept_values) && !graph.trains(kept_indices),
        "a selection hands back the numbers a row holds, and an index learns nothing",
    );
    let snapshot = graph.snapshot();
    let task = snapshot
        .tasks()
        .iter()
        .find(|task| task.kind == Kind::TopK)
        .expect("a top k is one task");
    assert_eq!(task.inputs[0], value.id());
    assert_eq!(task.inputs[1], NO_VALUE);
    assert_eq!(task.out, kept_values.id());
    assert_eq!(task.extra, kept_indices.id());
    assert_eq!(task.keep, 3);
}

#[test]
fn a_top_k_refuses_the_candidates_it_cannot_keep() {
    let graph = Graph::new();
    let value = graph.input(Shape::matrix(4, 8), Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.top_k(value, 0);
        }),
        "a top k of no candidates reads a row and keeps none",
    );
    assert!(
        refuses(|| {
            let _ = graph.top_k(value, CANDIDATES + 1);
        }),
        "a top k beyond the candidates a choice kernel holds is refused",
    );
    assert!(
        refuses(|| {
            let _ = graph.top_k(value, 9);
        }),
        "a top k of more candidates than a row holds is refused",
    );
    assert!(
        refuses(|| {
            let _ = graph.top_k(graph.permute(value, [0, 1, 3, 2]), 2);
        }),
        "a top k folds a row of a tensor stored row by row, and a permuted view holds no such row",
    );
    let free = graph.free(8);
    let dynamic = graph.input(Shape::of([1, 1, 4, 8]).freed(&[(3, free)]), Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.top_k(dynamic, 2);
        }),
        "a top k weighs every number a row holds, and a free width walks a length no row closes",
    );
}

#[test]
fn a_top_k_carries_no_gradient() {
    let graph = Graph::new();
    let value = graph.parameter(Shape::vector(8), neura_graph::Init::Zero, Element::Single);
    let (kept_values, _) = graph.top_k(value, 2);
    let loss = graph.sum(kept_values);
    assert!(
        refuses(|| {
            let _ = graph.backward(loss);
        }),
        "a selection hands back the numbers it read, and no gradient walks back to the row it read",
    );
}
