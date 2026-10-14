use neura_graph::{Graph, Shape, Value};

pub fn heads(graph: &Graph, value: Value, count: u32, width: u32) -> Value {
    let parts = (0..count)
        .map(|head| graph.slice(value, 3, head * width, width))
        .collect::<Vec<Value>>();
    graph.concat(&parts, 0)
}

pub fn seat(graph: &Graph, heads: u32, capacity: u32, rows: u32) -> Value {
    assert!(
        heads > 0 && capacity > 0 && rows > 0,
        "a seat of {rows} rows over {heads} heads of {capacity} slots names no slot",
    );
    let head = (0..heads)
        .map(|at| graph.fill(Shape::scalar(), (at * capacity) as f32))
        .collect::<Vec<Value>>();
    let row = (0..rows)
        .map(|at| graph.fill(Shape::scalar(), at as f32))
        .collect::<Vec<Value>>();
    graph.add(graph.concat(&head, 0), graph.concat(&row, 2))
}
