use neura_graph::{Graph, Value};

pub(crate) fn interleaved<'g>(
    graph: &Graph<'g>,
    value: Value<'g>,
    count: u32,
    width: u32,
    offset: u32,
) -> Value<'g> {
    let parts = (0..count)
        .map(|head| graph.slice(value, 3, head * width * 2 + offset, width))
        .collect::<Vec<Value<'g>>>();
    graph.concat(&parts, 0)
}
