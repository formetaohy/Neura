use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};

use std::panic::{AssertUnwindSafe, catch_unwind};

fn refuses(action: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(action)).is_err()
}

#[test]
fn an_op_refuses_a_tensor_of_another_graph() {
    let first = Graph::new();
    let second = Graph::new();
    let value = first.input(Shape::vector(3), Element::Single);
    assert!(
        refuses(|| {
            second.relu(value);
        }),
        "a tensor walks the graph that declared it, and an op of another graph refuses it",
    );
}

#[test]
fn a_gradient_set_refuses_a_tensor_of_another_graph() {
    let first = Graph::new();
    let weight = first.parameter(Shape::scalar(), Init::Constant(1.0), Element::Single);
    let gradients = first.backward(first.sum(weight));
    let second = Graph::new();
    let stranger = second.input(Shape::scalar(), Element::Single);
    assert_eq!(
        stranger.id(),
        weight.id(),
        "both graphs declare their first tensor in the same slot",
    );
    assert!(
        refuses(|| {
            let _ = gradients.of(stranger);
        }),
        "a gradient reaches a tensor of the graph that differentiated it, and a slot of another graph holds another tensor",
    );
}
