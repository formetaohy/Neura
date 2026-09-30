use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};

#[test]
fn a_revision_stays_current_while_its_graph_stands_still() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 4), Init::Zero, Element::Single);
    let revision = graph.revision();
    assert!(revision.is_current());
    assert_eq!(revision.stamp(), graph.stamp());
    let _ = graph.snapshot();
    assert!(revision.is_current(), "reading a graph does not move it",);
    let _ = graph.shape(weight);
    assert!(revision.is_current());
}

#[test]
fn a_graph_that_moves_on_retires_every_revision_it_handed_out() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 4), Init::Zero, Element::Single);
    let first = graph.revision();
    let second = graph.revision();
    assert_eq!(first.stamp(), second.stamp());
    graph.retain(weight);
    assert!(!first.is_current());
    assert!(!second.is_current());
    let third = graph.revision();
    assert!(third.is_current());
    assert!(third.stamp() != first.stamp());
    assert_eq!(third.stamp(), graph.stamp());
}

#[test]
fn a_graph_retires_its_revisions_wherever_it_moves() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 4), Init::Zero, Element::Single);
    let input = graph.input(Shape::matrix(4, 4), Element::Single);
    let product = graph.matmul(input, weight);
    graph.retain(product);

    let authored = graph.revision();
    let filled = graph.fill(Shape::vector(4), 1.0);
    assert!(!authored.is_current(), "a new task moves the graph");

    let retained = graph.revision();
    graph.retain(filled);
    assert!(!retained.is_current(), "a retain moves the graph");

    let differentiated = graph.revision();
    let loss = graph.sum(graph.matmul(input, weight));
    graph.backward(loss);
    assert!(
        !differentiated.is_current(),
        "a differentiation moves the graph",
    );

    let written = graph.revision();
    let zeroes = graph.fill(Shape::matrix(4, 4), 0.0);
    graph.copy_into(weight, zeroes);
    assert!(!written.is_current(), "an update in place moves the graph");

    let standing = graph.revision();
    assert!(standing.is_current());
    assert_eq!(standing.stamp(), graph.stamp());
}

#[test]
fn a_revision_of_one_graph_does_not_see_another_graph_move() {
    let first = Graph::new();
    let second = Graph::new();
    let revision = first.revision();
    second.fill(Shape::vector(4), 1.0);
    assert!(revision.is_current());
    assert!(revision.stamp() != second.stamp());
    first.fill(Shape::vector(4), 2.0);
    assert!(!revision.is_current());
}

#[test]
fn a_graph_drops_the_revisions_nobody_holds() {
    let graph = Graph::new();
    graph.revision();
    graph.fill(Shape::vector(4), 1.0);
    let standing = graph.revision();
    graph.fill(Shape::vector(4), 2.0);
    assert!(!standing.is_current());
}
