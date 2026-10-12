use neura_graph::{Graph, Shape, Value};

fn mean_over(graph: &Graph, value: Value, axes: &[u32]) -> Value {
    axes.iter()
        .fold(value, |folded, axis| graph.mean_axis(folded, *axis))
}

fn mean(graph: &Graph, value: Value) -> Value {
    mean_over(graph, value, &[3, 2, 1, 0])
}

fn mean_rows(graph: &Graph, value: Value) -> Value {
    mean_over(graph, value, &[2, 1, 0])
}

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
    mean(graph, graph.mul(difference, difference))
}

pub fn cross_entropy(graph: &Graph, logits: Value, target: Value) -> Value {
    let shape = graph.shape(logits);
    assert_eq!(
        shape,
        graph.shape(target),
        "a cross entropy weighs every class of a row by the target it holds for that class",
    );
    let log_probability = graph.log_softmax(logits);
    let weighted = graph.mul(target, log_probability);
    graph.neg(mean_rows(graph, graph.sum_rows(weighted)))
}

pub fn policy_loss(graph: &Graph, logits: Value, action: Value, advantage: Value) -> Value {
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
    graph.neg(mean_rows(
        graph,
        graph.sum_rows(graph.mul(advantage, taken)),
    ))
}
