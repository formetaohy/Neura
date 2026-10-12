use neura_graph::{Graph, Shape, Value};

pub fn heads(graph: &Graph, value: Value, count: u32, width: u32) -> Value {
    let parts = (0..count)
        .map(|head| graph.slice(value, 3, head * width, width))
        .collect::<Vec<Value>>();
    graph.concat(&parts, 0)
}

pub fn seat(graph: &Graph, position: Value, count: u32, capacity: u32) -> Value {
    let rows = (0..count)
        .map(|head| {
            graph.add(
                position,
                graph.fill(Shape::scalar(), (head * capacity) as f32),
            )
        })
        .collect::<Vec<Value>>();
    graph.concat(&rows, 0)
}
