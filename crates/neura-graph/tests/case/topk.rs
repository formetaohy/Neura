use neura_abi::{CANDIDATES, Element, Kind, NO_VALUE};
use neura_graph::{Graph, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn refusal_message(action: impl FnOnce()) -> String {
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action))
        .expect_err("the refusal a readback carries is reported");
    refused
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "a refusal without a message".to_owned())
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
    let empty = refusal_message(|| {
        let _ = graph.top_k(value, 0);
    });
    assert!(
        empty.contains("a top k keeps between one and"),
        "a top k of no candidates reads a row and keeps none: {empty}",
    );
    let beyond = refusal_message(|| {
        let _ = graph.top_k(value, CANDIDATES + 1);
    });
    assert!(
        beyond.contains("a top k keeps between one and"),
        "a top k beyond the candidates a choice kernel holds is refused: {beyond}",
    );
    let wider = refusal_message(|| {
        let _ = graph.top_k(value, 9);
    });
    assert!(
        wider.contains("reads a row of 8 numbers"),
        "a top k of more candidates than a row holds is refused: {wider}",
    );
    let view = graph.permute(value, [0, 1, 3, 2]);
    assert!(
        refuses(|| {
            let _ = graph.top_k(view, 2);
        }),
        "a top k of a view walks rows no storage lays out",
    );
    let free = graph.free(8);
    let dynamic = graph.input(Shape::of([1, 1, 4, 8]).freed(&[(3, free)]), Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.top_k(dynamic, 2);
        }),
        "a top k of a row of no fixed width is refused",
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
