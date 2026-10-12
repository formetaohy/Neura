use neura_graph::{Graph, Shape, Value};

pub fn rms<'g>(
    graph: &Graph<'g>,
    value: Value<'g>,
    weight: Value<'g>,
    floor: Value<'g>,
) -> Value<'g> {
    let squares = graph.mul(value, value);
    let mean = graph.mean_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(mean, floor)));
    graph.mul(graph.mul(value, scale), weight)
}

pub fn l2<'g>(graph: &Graph<'g>, value: Value<'g>, floor: Value<'g>) -> Value<'g> {
    let squares = graph.mul(value, value);
    let total = graph.sum_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(total, floor)));
    graph.mul(value, scale)
}

pub fn softplus<'g>(graph: &Graph<'g>, value: Value<'g>) -> Value<'g> {
    let threshold = graph.fill(Shape::scalar(), 20.0);
    let raised = graph.log(graph.add(graph.fill(Shape::scalar(), 1.0), graph.exp(value)));
    graph.select(graph.greater(value, threshold), value, raised)
}

pub fn flatten<'g>(graph: &Graph<'g>, value: Value<'g>) -> Value<'g> {
    let elements = graph.shape(value).elements();
    graph.reshape(value, Shape::of([1, 1, 1, elements]))
}
