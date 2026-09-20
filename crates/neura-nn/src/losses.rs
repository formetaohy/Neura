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

pub fn cross_entropy(graph: &Graph, logits: Value, target: Value) -> Value {
    let shape = graph.shape(logits);
    assert_eq!(
        shape,
        graph.shape(target),
        "a cross entropy weighs every class of a row by the target it holds for that class",
    );
    let rows = shape.rows();
    let log_probability = graph.log_softmax(logits);
    let weighted = graph.mul(target, log_probability);
    graph.mul(
        graph.sum(weighted),
        graph.fill(Shape::scalar(), -1.0 / rows as f32),
    )
}
