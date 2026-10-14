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

pub fn cross_entropy(graph: &Graph, logits: Value, classes: Value) -> Value {
    let shape = graph.shape(logits);
    assert_eq!(
        graph.shape(classes).dims(),
        [shape.dims()[0], shape.dims()[1], shape.dims()[2], 1],
        "a cross entropy weighs the class every row chose, and {:?} holds {} numbers per row of {:?}",
        graph.shape(classes).dims(),
        graph.shape(classes).dims()[3],
        shape.dims(),
    );
    graph.neg(mean_rows(graph, graph.class_log_softmax(logits, classes)))
}

pub fn policy_loss(graph: &Graph, logits: Value, action: Value, advantage: Value) -> Value {
    let shape = graph.shape(logits);
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
    let taken = graph.class_log_softmax(logits, action);
    graph.neg(mean_rows(graph, graph.mul(advantage, taken)))
}
