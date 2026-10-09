use neura_abi::{Element, Kind, NO_VALUE};
use neura_graph::{Graph, Init, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_sample_asks_for_a_seed_and_two_parameters() {
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(4, 8), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let keep = graph.parameter(Shape::scalar(), Init::Constant(4.0), Element::Single);
    let cumulative = graph.parameter(Shape::scalar(), Init::Constant(0.9), Element::Single);
    let draw = graph.sample(logits, seed, keep, cumulative);
    assert_eq!(graph.shape(draw), Shape::matrix(4, 1));
    assert_eq!(graph.element(draw), Element::Single);
    assert!(
        !graph.trains(draw),
        "a sample draws the index of a row, and an index learns nothing",
    );
    let snapshot = graph.snapshot();
    let tasks = snapshot.tasks();
    let task = tasks
        .iter()
        .find(|task| task.kind == Kind::Sample)
        .expect("a sample is one task");
    assert_eq!(task.inputs[0], logits.id());
    assert_eq!(task.inputs[1], seed.id());
    assert_eq!(task.inputs[2], keep.id());
    assert_eq!(task.inputs[3], cumulative.id());
    assert_eq!(task.inputs[4], NO_VALUE);
    assert_eq!(task.inputs[5], NO_VALUE);
    assert_eq!(task.out, draw.id());
}

#[test]
fn a_sample_refuses_the_values_it_cannot_read() {
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(4, 8), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let keep = graph.input(Shape::scalar(), Element::Single);
    let cumulative = graph.input(Shape::scalar(), Element::Single);
    let vector = graph.input(Shape::vector(4), Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.sample(logits, vector, keep, cumulative);
        }),
        "a sample takes one seed, not four numbers",
    );
    assert!(
        refuses(|| {
            let _ = graph.sample(logits, seed, vector, cumulative);
        }),
        "a sample keeps one candidate count, not four numbers",
    );
    assert!(
        refuses(|| {
            let _ = graph.sample(logits, seed, keep, vector);
        }),
        "a sample draws one cumulative mass, not four numbers",
    );
    assert!(
        refuses(|| {
            let _ = graph.sample(graph.permute(logits, [0, 1, 3, 2]), seed, keep, cumulative);
        }),
        "a sample folds a row of a tensor stored row by row, and a view holds no such row",
    );
}
