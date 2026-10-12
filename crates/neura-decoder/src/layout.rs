use neura_graph::{Graph, Shape, Value};

pub fn heads<'g>(graph: &Graph<'g>, value: Value<'g>, count: u32, width: u32) -> Value<'g> {
    let parts = (0..count)
        .map(|head| graph.slice(value, 3, head * width, width))
        .collect::<Vec<Value<'g>>>();
    graph.concat(&parts, 0)
}

pub fn seat<'g>(graph: &Graph<'g>, position: Value<'g>, count: u32, capacity: u32) -> Value<'g> {
    let rows = (0..count)
        .map(|head| {
            graph.add(
                position,
                graph.fill(Shape::scalar(), (head * capacity) as f32),
            )
        })
        .collect::<Vec<Value<'g>>>();
    graph.concat(&rows, 0)
}
