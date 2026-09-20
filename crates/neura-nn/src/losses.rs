use neura_program::{Graph, Shape, Value};

pub fn mse_loss(graph: &Graph, prediction: Value, target: Value) -> Value {
    let shape = graph.shape(prediction);
    assert_eq!(
        shape,
        graph.shape(target),
        "a squared error needs a prediction and a target of the same shape",
    );
    let difference = graph.add(
        prediction,
        graph.mul(target, graph.fill(Shape::scalar(), -1.0)),
    );
    let squared = graph.mul(difference, difference);
    let total = graph.sum(squared);
    graph.mul(
        total,
        graph.fill(Shape::scalar(), 1.0 / shape.elements() as f32),
    )
}
