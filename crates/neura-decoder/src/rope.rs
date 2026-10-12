use neura_graph::{Graph, Value};

pub fn partial(graph: &Graph, value: Value, position: Value, base: Value, width: u32) -> Value {
    let written = graph.shape(value).dims()[3];
    let front = graph.slice(value, 3, 0, width);
    let turned = graph.rope(front, Some(position), Some(base));
    if written == width {
        return turned;
    }
    let rest = graph.slice(value, 3, width, written - width);
    graph.concat(&[turned, rest], 3)
}
