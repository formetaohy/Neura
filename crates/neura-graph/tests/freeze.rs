use neura_abi::Element;
use neura_graph::{Graph, Init, Residency, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn residency(graph: &Graph<'_>, id: u32) -> Residency {
    graph.snapshot().values()[id as usize].residency
}

#[test]
fn a_frozen_parameter_stops_the_gradient_that_reaches_it() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(3, 4), Init::Constant(1.0), Element::Single);
    graph.freeze(&[weight]);
    assert!(!graph.trains(weight));
    assert_eq!(
        residency(&graph, weight.id()),
        Residency::Parameter,
        "a frozen tensor keeps the weight store its parameter lives in",
    );
    let observations = graph.input(Shape::matrix(2, 3), Element::Single);
    let free = graph.parameter(Shape::matrix(2, 4), Init::Constant(0.0), Element::Single);
    let loss = graph.sum(graph.add(graph.matmul(observations, weight), free));
    let gradients = graph.backward(loss);
    assert_eq!(gradients.of(free).shape(), free.shape());
    assert!(
        refuses(|| {
            let _ = gradients.of(weight);
        }),
        "a frozen parameter carries no gradient an optimizer could follow",
    );
}

#[test]
fn a_frozen_parameter_leaves_the_operands_beside_it_learning() {
    let graph = Graph::new();
    let frozen = graph.parameter(Shape::matrix(3, 4), Init::Constant(2.0), Element::Single);
    let learning = graph.parameter(Shape::matrix(4, 5), Init::Constant(0.5), Element::Single);
    graph.freeze(&[frozen]);
    let loss = graph.sum(graph.matmul(frozen, learning));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(learning);
    assert_eq!(gradient.shape(), learning.shape());
    assert!(graph.trains(learning));
    assert!(refuses(|| {
        let _ = gradients.of(frozen);
    }));
}

#[test]
fn a_detached_tensor_stops_the_gradient_that_wraps_it() {
    let graph = Graph::new();
    let learning = graph.parameter(Shape::matrix(3, 4), Init::Constant(0.5), Element::Single);
    let teacher = graph.parameter(Shape::matrix(3, 4), Init::Constant(0.5), Element::Single);
    let observations = graph.input(Shape::matrix(2, 3), Element::Single);
    let student = graph.matmul(observations, learning);
    let mut loss = graph.sum(student);
    loss = graph.add(
        loss,
        graph.sum(graph.detach(graph.matmul(observations, teacher))),
    );
    let gradients = graph.backward(loss);
    assert!(graph.trains(learning));
    let detached = graph.detach(teacher);
    assert!(!graph.trains(detached));
    assert_eq!(
        residency(&graph, detached.id()),
        Residency::View,
        "a detached tensor is a view that carries no gradient of its own",
    );
    assert!(refuses(|| {
        let _ = gradients.of(teacher);
    }));
}

#[test]
fn a_frozen_parameter_meets_the_tasks_that_follow_it() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(3, 4), Init::Constant(1.0), Element::Single);
    let observations = graph.input(Shape::matrix(2, 3), Element::Single);
    let product = graph.matmul(observations, weight);
    assert!(
        refuses(|| {
            graph.freeze(&[weight]);
        }),
        "a parameter is frozen before the tasks that read it are authored",
    );
    let detached = graph.detach(product);
    assert!(!graph.trains(detached));
}

#[test]
fn only_a_parameter_freezes() {
    let graph = Graph::new();
    let observations = graph.input(Shape::matrix(2, 3), Element::Single);
    let hidden = graph.resident(Shape::matrix(2, 3), Element::Single);
    assert!(refuses(|| {
        graph.freeze(&[observations]);
    }));
    assert!(refuses(|| {
        graph.freeze(&[hidden]);
    }));
    assert!(refuses(|| {
        graph.freeze(&[]);
    }));
}
