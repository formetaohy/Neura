use neura_graph::{Graph, Value};

pub(crate) fn interleaved(
    graph: &Graph,
    value: Value,
    count: u32,
    width: u32,
    offset: u32,
) -> Value {
    let parts = (0..count)
        .map(|head| graph.slice(value, 3, head * width * 2 + offset, width))
        .collect::<Vec<Value>>();
    graph.concat(&parts, 0)
}
