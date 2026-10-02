use neura_abi::Element;
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
