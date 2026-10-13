use neura_abi::{Element, Kind};
use neura_graph::{Graph, Init, Shape};
use neura_pointwise as op;

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_comparison_turns_two_tensors_into_a_mask_of_ones_and_zeros() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    let mask = graph.greater(left, right);
    assert_eq!(graph.shape(mask), Shape::matrix(2, 3));
    assert_eq!(graph.element(mask), Element::Single);
    let snapshot = graph.snapshot();
    let tasks = snapshot.tasks();
    assert_eq!(
        tasks.len(),
        1,
        "a comparison weighs two tensors in one task"
    );
    assert_eq!(tasks[0].kind, Kind::Binary);
    assert_eq!(tasks[0].op, op::GREATER);
    assert_eq!(tasks[0].inputs[0], left.id());
    assert_eq!(tasks[0].inputs[1], right.id());
}

#[test]
fn every_comparison_is_a_binary_op_of_its_own() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let comparisons = [
        (op::GREATER, graph.greater(left, right)),
        (op::GREATER_EQUAL, graph.greater_equal(left, right)),
        (op::LESS, graph.less(left, right)),
        (op::LESS_EQUAL, graph.less_equal(left, right)),
        (op::EQUAL, graph.equal(left, right)),
        (op::NOT_EQUAL, graph.not_equal(left, right)),
    ];
    let snapshot = graph.snapshot();
    let tasks = snapshot.tasks();
    assert_eq!(tasks.len(), comparisons.len());
    for (position, (code, _)) in comparisons.iter().enumerate() {
        assert_eq!(tasks[position].kind, Kind::Binary);
        assert_eq!(tasks[position].op, *code);
    }
    for (code, mask) in comparisons {
        assert_eq!(op::kind(code), Kind::Binary);
        assert_eq!(graph.element(mask), Element::Single);
        assert_eq!(graph.shape(mask), Shape::vector(4));
    }
}

#[test]
fn a_mask_carries_no_gradient_of_its_own() {
    let graph = Graph::new();
    let value = graph.gradient_input(Shape::vector(4), Element::Single);
    let limit = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let mask = graph.greater(value, limit);
    let loss = graph.sum(graph.mul(value, mask));
    let gradients = graph.backward(loss);
    assert_eq!(gradients.of(value).shape(), value.shape());
    assert!(
        refuses(|| {
            let _ = gradients.of(limit);
        }),
        "a comparison weighs a branch of the graph and descends from nothing",
    );
}

#[test]
fn a_selection_asks_three_tensors_of_one_mask() {
    let graph = Graph::new();
    let condition = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let accept = graph.parameter(Shape::matrix(2, 4), Init::Zero, Element::Single);
    let reject = graph.parameter(Shape::matrix(2, 4), Init::Zero, Element::Single);
    let picked = graph.select(condition, accept, reject);
    assert_eq!(graph.shape(picked), Shape::matrix(2, 4));
    let snapshot = graph.snapshot();
    let tasks = snapshot.tasks();
    assert_eq!(tasks.len(), 1);
    assert_eq!(tasks[0].kind, Kind::Select);
    assert_eq!(tasks[0].inputs[0], condition.id());
    assert_eq!(tasks[0].inputs[1], accept.id());
    assert_eq!(tasks[0].inputs[2], reject.id());
}

#[test]
fn a_selection_carries_the_gradient_of_the_branch_it_picked() {
    let graph = Graph::new();
    let condition = graph.greater(
        graph.parameter(Shape::vector(4), Init::Constant(1.0), Element::Single),
        graph.parameter(Shape::vector(4), Init::Constant(0.0), Element::Single),
    );
    let accept = graph.parameter(Shape::matrix(2, 4), Init::Constant(1.0), Element::Single);
    let reject = graph.parameter(Shape::matrix(2, 4), Init::Constant(-1.0), Element::Single);
    let loss = graph.sum(graph.select(condition, accept, reject));
    let gradients = graph.backward(loss);
    assert_eq!(gradients.of(accept).shape(), accept.shape());
    assert_eq!(gradients.of(reject).shape(), reject.shape());
}

#[test]
fn a_condition_learns_through_no_selection() {
    let graph = Graph::new();
    let condition = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let accept = graph.parameter(Shape::matrix(2, 4), Init::Constant(1.0), Element::Single);
    let reject = graph.parameter(Shape::matrix(2, 4), Init::Constant(-1.0), Element::Single);
    let loss = graph.sum(graph.select(condition, accept, reject));
    let gradients = graph.backward(loss);
    assert!(
        refuses(|| {
            let _ = gradients.of(condition);
        }),
        "a condition names a branch and takes no gradient of its own",
    );
}

#[test]
fn a_selection_combines_the_shapes_it_picks_between() {
    let graph = Graph::new();
    let condition = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let accept = graph.parameter(Shape::of([2, 1, 3, 4]), Init::Zero, Element::Single);
    let reject = graph.parameter(Shape::of([1, 5, 1, 4]), Init::Zero, Element::Single);
    let picked = graph.select(condition, accept, reject);
    assert_eq!(graph.shape(picked), Shape::of([2, 5, 3, 4]));
    assert!(
        refuses(|| {
            let other = graph.parameter(Shape::matrix(5, 4), Init::Zero, Element::Single);
            let _ = graph.select(condition, accept, other);
        }),
        "a selection combines every shape it picks between",
    );
}

#[test]
fn a_selection_keeps_what_the_mask_names_and_drops_the_rest() {
    let graph = Graph::new();
    let value = graph.gradient_input(Shape::vector(4), Element::Single);
    let mask = graph.greater(value, graph.fill(Shape::scalar(), 0.0));
    let kept = graph.select(mask, value, graph.fill(Shape::vector(4), 0.0));
    let loss = graph.sum(kept);
    let gradients = graph.backward(loss);
    assert_eq!(gradients.of(value).shape(), value.shape());
    let snapshot = graph.snapshot();
    let tasks = snapshot.tasks();
    assert!(
        tasks.iter().any(|task| task.kind == Kind::Select),
        "a masked term picks between the value and nothing",
    );
}
