use neura_abi::{Element, Kind};
use neura_graph::{Graph, Init, Residency, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn residency(graph: &Graph<'_>, id: u32) -> Residency {
    graph.snapshot().values()[id as usize].residency
}

#[test]
fn a_gradient_input_carries_the_gradient_that_reaches_it() {
    let graph = Graph::new();
    let observations = graph.gradient_input(Shape::matrix(2, 3), Element::Single);
    let weight = graph.parameter(Shape::matrix(3, 4), Init::Constant(1.0), Element::Single);
    let loss = graph.sum(graph.matmul(observations, weight));
    let gradients = graph.backward(loss);
    assert!(graph.trains(observations));
    assert_eq!(
        residency(&graph, observations.id()),
        Residency::Input,
        "a gradient input reads from the tensors store like every other input",
    );
    assert_eq!(gradients.of(observations).shape(), observations.shape());
    assert_eq!(gradients.of(weight).shape(), weight.shape());
}

#[test]
fn an_input_that_carries_no_gradient_stays_out_of_the_tape() {
    let graph = Graph::new();
    let observations = graph.input(Shape::matrix(2, 3), Element::Single);
    let weight = graph.parameter(Shape::matrix(3, 4), Init::Constant(1.0), Element::Single);
    let loss = graph.sum(graph.matmul(observations, weight));
    let gradients = graph.backward(loss);
    assert!(!graph.trains(observations));
    assert!(
        refuses(|| {
            let _ = gradients.of(observations);
        }),
        "an input a loss does not reach carries no gradient",
    );
    assert_eq!(gradients.of(weight).shape(), weight.shape());
}

#[test]
fn only_a_parameter_freezes() {
    let graph = Graph::new();
    let observations = graph.gradient_input(Shape::matrix(2, 3), Element::Single);
    assert!(
        refuses(|| graph.freeze(&[observations])),
        "a frozen tensor holds a model parameter, and a gradient input is an observation",
    );
}

#[test]
fn a_gradient_input_learns_through_an_update_in_place() {
    let graph = Graph::new();
    let observations = graph.gradient_input(Shape::matrix(2, 3), Element::Single);
    let weight = graph.parameter(Shape::matrix(3, 4), Init::Constant(1.0), Element::Single);
    let loss = graph.sum(graph.matmul(observations, weight));
    let gradients = graph.backward(loss);
    let rate = graph.fill(Shape::scalar(), -0.5);
    graph.add_into(observations, graph.mul(gradients.of(observations), rate));
    assert_eq!(
        residency(&graph, observations.id()),
        Residency::Input,
        "an update in place leaves the leaf where it was declared",
    );
}

#[test]
fn a_gradient_folds_a_broadcast_before_it_lands_in_the_layout_of_a_view() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::of([2, 3, 1, 4]), Init::Zero, Element::Single);
    let bias = graph.input(Shape::of([3, 2, 5, 4]), Element::Single);
    let turned = graph.permute(weight, [1, 0, 2, 3]);
    let loss = graph.sum(graph.mul(turned, bias));
    let gradient = graph.backward(loss).of(weight);
    assert_eq!(gradient.shape(), weight.shape());
    let snapshot = graph.snapshot();
    let layout = snapshot
        .tasks()
        .iter()
        .find(|task| task.kind == Kind::Layout)
        .expect("a gradient of a permuted view lands through a layout task");
    let contribution = snapshot.values()[layout.inputs[0] as usize].shape;
    let view = snapshot.values()[layout.inputs[1] as usize].shape;
    assert_eq!(
        contribution, view,
        "a layout task lays out the gradient of the very view it names, and the axes a view broadcast are folded before the layout reads it",
    );
    let folded = snapshot
        .tasks()
        .iter()
        .find(|task| task.out == layout.inputs[0])
        .expect("the contribution a layout task reads comes from the task that wrote it");
    assert_eq!(
        folded.kind,
        Kind::SumAxis,
        "the gradient of a view that spread over the axes around it holds the sum of every element it spread",
    );
}
