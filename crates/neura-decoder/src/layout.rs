use neura_graph::{Graph, Shape, Value};

pub fn heads(graph: &Graph, value: Value, count: u32, width: u32) -> Value {
    let parts = (0..count)
        .map(|head| graph.slice(value, 3, head * width, width))
        .collect::<Vec<Value>>();
    graph.concat(&parts, 0)
}

pub(crate) fn base(graph: &Graph, heads: u32, batch: u32, stride: u32) -> Value {
    assert!(
        heads > 0 && batch > 0 && stride > 0,
        "a seat of {batch} planes of {heads} heads over {stride} slots names no slot",
    );
    let head = (0..heads)
        .map(|at| graph.fill(Shape::scalar(), (at * batch * stride) as f32))
        .collect::<Vec<Value>>();
    let plane = (0..batch)
        .map(|at| graph.fill(Shape::scalar(), (at * stride) as f32))
        .collect::<Vec<Value>>();
    graph.add(graph.concat(&head, 0), graph.concat(&plane, 1))
}

pub(crate) fn rows(graph: &Graph, rows: u32) -> Value {
    assert!(rows > 0, "a seat walks no row of {rows}");
    let rows = (0..rows)
        .map(|at| graph.fill(Shape::scalar(), at as f32))
        .collect::<Vec<Value>>();
    graph.concat(&rows, 2)
}

pub(crate) fn ring(graph: &Graph, window: Value, rows: Value, cursor: Value) -> Value {
    let position = graph.add(cursor, rows);
    let turns = graph.floor(graph.div(position, window));
    graph.sub(position, graph.mul(turns, window))
}
