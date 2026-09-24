use neura_abi::Kind;
use neura_graph::{Graph, Shape};

#[test]
fn a_snapshot_keeps_the_graph_as_it_was_authored() {
    let graph = Graph::new();
    graph.input(Shape::vector(4));
    let before = graph.snapshot();
    graph.fill(Shape::vector(4), 2.0);
    let after = graph.snapshot();

    assert_eq!(before.values().len(), 1);
    assert!(before.tasks().is_empty());
    assert_eq!(after.values().len(), 2);
    assert_eq!(after.tasks().len(), 1);
    assert_eq!(after.tasks()[0].kind, Kind::Fill);
}
