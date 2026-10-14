use neura_graph::{Graph, Shape, Value};

pub fn rms(graph: &Graph, value: Value, weight: Value, floor: Value) -> Value {
    let squares = graph.mul(value, value);
    let mean = graph.mean_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(mean, floor)));
    graph.mul(graph.mul(value, scale), weight)
}

pub fn l2(graph: &Graph, value: Value, floor: Value) -> Value {
    let squares = graph.mul(value, value);
    let total = graph.sum_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(total, floor)));
    graph.mul(value, scale)
}

pub fn softplus(graph: &Graph, value: Value) -> Value {
    let threshold = graph.fill(Shape::scalar(), 20.0);
    let raised = graph.log(graph.add(graph.fill(Shape::scalar(), 1.0), graph.exp(value)));
    graph.select(graph.greater(value, threshold), value, raised)
}

pub fn joined(graph: &Graph, value: Value, count: u32) -> Value {
    let shape = graph.shape(value);
    assert_eq!(
        shape.dims()[0],
        count,
        "a join lays the {count} heads of a chunk beside each other, and {:?} holds {} planes",
        shape.dims(),
        shape.dims()[0],
    );
    let parts = (0..count)
        .map(|head| graph.slice(value, 0, head, 1))
        .collect::<Vec<Value>>();
    graph.concat(&parts, 3)
}
