use neura_abi::{Element, Kind};
use neura_graph::{Graph, Shape};

#[test]
fn a_snapshot_keeps_the_graph_as_it_was_authored() {
    let graph = Graph::new();
    graph.input(Shape::vector(4), Element::Single);
    let before = graph.snapshot();
    graph.fill(Shape::vector(4), 2.0);
    let after = graph.snapshot();

    assert_eq!(before.values().len(), 1);
    assert!(before.tasks().is_empty());
    assert_eq!(after.values().len(), 2);
    assert_eq!(after.tasks().len(), 1);
    assert_eq!(after.tasks()[0].kind, Kind::Fill);
}

#[test]
fn a_stamp_names_the_snapshot_a_graph_carries() {
    let authored = || {
        let graph = Graph::new();
        let value = graph.parameter(Shape::vector(4), neura_graph::Init::Zero, Element::Single);
        let out = graph.relu(value);
        graph.retain(out);
        (graph, value, out)
    };
    let (graph, _, _) = authored();
    let stamp = graph.stamp();
    assert_eq!(
        stamp,
        graph.stamp(),
        "a graph that only reads itself carries one stamp",
    );
    let other = authored().0;
    assert!(stamp != other.stamp(), "two graphs carry two stamps");
    assert!(
        stamp != Graph::new().stamp(),
        "an empty graph carries no other graph's stamp",
    );
    let (mutating, value, out) = authored();
    let before = mutating.stamp();
    mutating.relu(out);
    assert!(
        before != mutating.stamp(),
        "a task the graph takes on turns its stamp",
    );
    let before = mutating.stamp();
    mutating.retain(out);
    assert!(
        before != mutating.stamp(),
        "a tensor the graph pins turns its stamp",
    );
    let before = mutating.stamp();
    mutating.add_into(value, value);
    assert!(
        before != mutating.stamp(),
        "a tensor the graph updates in place turns its stamp",
    );
    assert_eq!(value.shape(), out.shape());
}
