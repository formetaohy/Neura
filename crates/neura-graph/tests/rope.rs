use neura_abi::{Element, Kind};
use neura_graph::{Graph, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_rope_task_carries_the_position_every_row_starts_from() {
    let graph = Graph::new();
    let value = graph.input(Shape::of([2, 1, 4, 6]), Element::Single);
    let cursor = graph.input(Shape::of([2, 1, 1, 1]), Element::Single);
    let out = graph.rope(value, Some(cursor), 10000.0);
    assert_eq!(graph.shape(out), graph.shape(value));
    assert_eq!(graph.element(out), Element::Single);
    let snapshot = graph.snapshot();
    let task = &snapshot.tasks()[0];
    assert_eq!(task.kind, Kind::Rope);
    assert_eq!(task.out, out.id());
    assert_eq!(task.inputs[0], value.id());
    assert_eq!(task.origin, cursor.id());
    assert_eq!(task.param, 10000.0);
}

#[test]
fn a_rope_refuses_what_it_cannot_rotate() {
    let graph = Graph::new();
    let odd = graph.input(Shape::of([1, 1, 2, 5]), Element::Single);
    assert!(refuses(|| {
        graph.rope(odd, None, 10000.0);
    }));
    let quantized = graph.quantize(graph.input(Shape::of([1, 1, 2, 4]), Element::Single), 0.5);
    assert!(refuses(|| {
        graph.rope(quantized, None, 10000.0);
    }));
    let value = graph.input(Shape::of([2, 1, 4, 6]), Element::Single);
    assert!(refuses(|| {
        graph.rope(value, None, 1.0);
    }));
    let wide = graph.input(Shape::of([3, 1, 1, 2]), Element::Single);
    assert!(refuses(|| {
        graph.rope(value, Some(wide), 10000.0);
    }));
    let tall = graph.input(Shape::of([2, 1, 2, 1]), Element::Single);
    assert!(refuses(|| {
        graph.rope(value, Some(tall), 10000.0);
    }));
}
