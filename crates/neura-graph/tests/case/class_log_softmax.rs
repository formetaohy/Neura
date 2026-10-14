use neura_abi::{Element, Kind, NO_VALUE};
use neura_graph::{Graph, Shape};

fn refusal_message(action: impl FnOnce()) -> String {
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action))
        .expect_err("the refusal a graph carries is reported");
    refused
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "a refusal without a message".to_owned())
}

#[test]
fn a_class_log_softmax_weighs_one_class_of_every_row() {
    let graph = Graph::new();
    let logits = graph.gradient_input(Shape::matrix(4, 3), Element::Single);
    let classes = graph.input(Shape::matrix(4, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    assert_eq!(graph.shape(chosen), Shape::matrix(4, 1));
    assert_eq!(graph.element(chosen), Element::Single);
    let snapshot = graph.snapshot();
    assert!(
        snapshot.values()[chosen.id() as usize].requires_grad,
        "a class log softmax carries the gradient of every logit it weighs",
    );
    let task = snapshot
        .tasks()
        .iter()
        .find(|task| task.kind == Kind::ClassLogSoftmax)
        .expect("a class log softmax is one task");
    assert_eq!(task.inputs[0], logits.id());
    assert_eq!(task.inputs[1], classes.id());
    assert_eq!(task.out, chosen.id());
    assert_ne!(
        task.extra, NO_VALUE,
        "a class log softmax carries its row normalizer"
    );
    let statistic = &snapshot.values()[task.extra as usize];
    assert_eq!(statistic.shape, Shape::matrix(4, 1));
    assert_eq!(statistic.element, Element::Single);
}

#[test]
fn a_class_log_softmax_gradient_weighs_the_class_the_logits_chose() {
    let graph = Graph::new();
    let logits = graph.parameter(
        Shape::matrix(4, 3),
        neura_graph::Init::Zero,
        Element::Single,
    );
    let classes = graph.input(Shape::matrix(4, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    let loss = graph.sum(chosen);
    let gradients = graph.backward(loss);
    let gradient = gradients.of(logits);
    assert_eq!(graph.shape(gradient), Shape::matrix(4, 3));
    let snapshot = graph.snapshot();
    let task = snapshot
        .tasks()
        .iter()
        .find(|task| task.kind == Kind::ClassLogSoftmaxGrad)
        .expect("a class log softmax gradient is one task");
    assert_eq!(task.inputs[0], logits.id());
    assert_eq!(task.inputs[1], classes.id());
    assert_ne!(
        task.inputs[3], NO_VALUE,
        "a gradient reads the normalizer its forward published",
    );
    assert_eq!(
        snapshot.values()[task.inputs[2] as usize].shape,
        Shape::matrix(4, 1)
    );
    assert_eq!(
        snapshot.values()[task.out as usize].shape,
        Shape::matrix(4, 3)
    );
}

#[test]
fn a_class_log_softmax_asks_for_one_class_of_every_row() {
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(4, 3), Element::Single);
    let list = graph.input(Shape::matrix(4, 1), Element::Single);
    let wide = graph.input(Shape::matrix(4, 2), Element::Single);
    let rows = graph.input(Shape::matrix(3, 1), Element::Single);
    let viewed = graph.input(Shape::matrix(3, 4), Element::Single);

    let listless = refusal_message(|| {
        let _ = graph.class_log_softmax(logits, wide);
    });
    assert!(
        listless.contains("an index list holds one index per row"),
        "a class log softmax reads one class of every row: {listless}",
    );
    let short = refusal_message(|| {
        let _ = graph.class_log_softmax(logits, rows);
    });
    assert!(
        short.contains("a class log softmax weighs one class of every row it folds"),
        "a class log softmax weighs one class for every row of its logits: {short}",
    );
    let turned = refusal_message(|| {
        let _ = graph.class_log_softmax(graph.permute(viewed, [0, 1, 3, 2]), list);
    });
    assert!(
        turned.contains("a class log softmax folds a row of a tensor stored row by row"),
        "a class log softmax reads every class of a row at one stride: {turned}",
    );
    let free = graph.free(3);
    let bounded = graph.input(Shape::matrix(4, 3).freed(&[(3, free)]), Element::Single);
    let unbounded = refusal_message(|| {
        let _ = graph.class_log_softmax(bounded, list);
    });
    assert!(
        unbounded.contains("a class log softmax weighs every class a row holds"),
        "a class log softmax weighs a row whose classes are known: {unbounded}",
    );
}
