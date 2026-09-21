use neura_program::{Graph, Shape, Value};

pub fn mse_loss<'g>(graph: &Graph<'g>, prediction: Value<'g>, target: Value<'g>) -> Value<'g> {
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

pub fn cross_entropy<'g>(graph: &Graph<'g>, logits: Value<'g>, target: Value<'g>) -> Value<'g> {
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

pub fn policy_loss<'g>(
    graph: &Graph<'g>,
    logits: Value<'g>,
    action: Value<'g>,
    advantage: Value<'g>,
) -> Value<'g> {
    let shape = graph.shape(logits);
    let classes = shape.dims()[3];
    let rows = shape.rows();
    assert_eq!(
        graph.shape(action).elements(),
        rows,
        "a policy weighs one taken action per row it scores",
    );
    assert_eq!(
        graph.shape(action).dims()[3],
        1,
        "a taken action is the class its row chose, and {:?} holds a row of them",
        graph.shape(action).dims(),
    );
    let weights = graph.shape(advantage);
    assert!(
        weights.is_scalar() || (weights.dims()[3] == 1 && weights.elements() == rows),
        "an advantage weighs one action per row, and {:?} holds {} weights for {rows} rows",
        weights.dims(),
        weights.elements(),
    );
    let taken = graph.mul(graph.one_hot(action, classes), graph.log_softmax(logits));
    graph.mul(
        graph.sum(graph.mul(advantage, taken)),
        graph.fill(Shape::scalar(), -1.0 / rows as f32),
    )
}
