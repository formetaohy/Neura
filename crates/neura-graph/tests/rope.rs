use neura_abi::{Element, Kind};
use neura_graph::{AttentionOptions, Graph, Shape, Value};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn packed<'g>(graph: &Graph<'g>, extent: neura_graph::Free, bound: u32, planes: u32) -> Value<'g> {
    graph.input(
        Shape::of([planes, 1, bound, 4]).freed(&[(2, extent)]),
        Element::Single,
    )
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

#[test]
fn a_packed_rope_walks_the_rows_the_axis_places() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let ragged = graph.ragged(16, lengths);
    let value = graph.gradient_input(
        Shape::of([1, 1, 16, 4]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let turned = graph.rope(value, None, 10000.0);
    assert_eq!(graph.shape(turned), graph.shape(value));
    let snapshot = graph.snapshot();
    let task = snapshot
        .tasks()
        .iter()
        .find(|task| task.kind == Kind::Rope)
        .expect("a packed rotation turns the rows the axis packs");
    assert_eq!(task.origin, neura_abi::NO_VALUE);
    assert_eq!(
        task.segments,
        ragged.offsets.id(),
        "a packed rotation walks the offsets the axis closes, and every row of one plane turns from the seat its own offsets name",
    );
    let loss = graph.sum(turned);
    let gradients = graph.backward(loss);
    let gradient = gradients.of(value);
    graph.retain(gradient);
    let snapshot = graph.snapshot();
    let grad = snapshot
        .tasks()
        .iter()
        .find(|task| task.kind == Kind::RopeGrad)
        .expect("a packed rotation hands its rows a gradient");
    assert_eq!(
        grad.segments,
        ragged.offsets.id(),
        "a packed gradient turns the rows back from the same seats the rotation used",
    );
}

#[test]
fn a_packed_rope_holds_no_plane_beside_the_rows_its_axis_packs() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let ragged = graph.ragged(16, lengths);
    let value = packed(&graph, ragged.extent, 16, 2);
    assert!(
        refuses(|| {
            graph.rope(value, None, 10000.0);
        }),
        "a rotation of a packed axis turns the rows of one plane at a time, and two planes stand beside the rows the axis packs",
    );
}

#[test]
fn a_packed_rope_refuses_a_cursor_that_places_every_plane_from_one_seat() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let ragged = graph.ragged(16, lengths);
    let value = packed(&graph, ragged.extent, 16, 1);
    let cursor = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
    assert!(
        refuses(|| {
            graph.rope(value, Some(cursor), 10000.0);
        }),
        "one cursor placed the rows of four planes from one position",
    );
}

#[test]
fn a_packed_rope_keeps_the_seat_of_a_single_plane() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(1), Element::Single);
    let ragged = graph.ragged(16, lengths);
    let value = packed(&graph, ragged.extent, 16, 1);
    let cursor = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
    let turned = graph.rope(value, Some(cursor), 10000.0);
    let snapshot = graph.snapshot();
    let task = snapshot
        .tasks()
        .iter()
        .find(|task| task.kind == Kind::Rope)
        .expect("a single plane hands its rows one seat");
    assert_eq!(task.origin, cursor.id());
    assert_eq!(task.segments, ragged.offsets.id());
    assert_eq!(graph.shape(turned).free(2), Some(ragged.extent.slot()));
}

fn packed_attention<'g>(
    graph: &Graph<'g>,
    query: Value<'g>,
    keys: neura_graph::Ragged<'g>,
    queries: neura_graph::Ragged<'g>,
) -> Value<'g> {
    let cache = packed(graph, keys.extent, 16, 1);
    graph.attention(
        query,
        cache,
        cache,
        AttentionOptions {
            scale: 1.0,
            causal: true,
            origin: None,
            segments: Some(keys.offsets),
            reach: None,
            query_segments: Some(queries.offsets),
        },
    )
}

#[test]
fn a_packed_rope_refuses_the_rows_of_a_query_chunk() {
    let graph = Graph::new();
    let query_lengths = graph.input(Shape::vector(4), Element::Single);
    let key_lengths = graph.input(Shape::vector(4), Element::Single);
    let queries = graph.ragged(16, query_lengths);
    let keys = graph.ragged(16, key_lengths);
    let query = packed(&graph, queries.extent, 16, 1);
    packed_attention(&graph, query, keys, queries);
    assert!(
        refuses(|| {
            graph.rope(query, None, 10000.0);
        }),
        "a rotation of a query chunk turns its rows by their own offsets, and the device places those rows at the end of the key plane their sequence already holds",
    );
}

#[test]
fn a_rope_over_a_chunk_turns_before_the_axis_that_places_it() {
    let graph = Graph::new();
    let query_lengths = graph.input(Shape::vector(4), Element::Single);
    let key_lengths = graph.input(Shape::vector(4), Element::Single);
    let queries = graph.ragged(16, query_lengths);
    let keys = graph.ragged(16, key_lengths);
    let query = packed(&graph, queries.extent, 16, 1);
    let turned = graph.rope(query, None, 10000.0);
    assert!(
        refuses(|| {
            packed_attention(&graph, turned, keys, queries);
        }),
        "a query axis gathered the rows a rotation placed by their own plane's offsets",
    );
}

#[test]
fn a_packed_rope_refuses_a_seat_no_ragged_axis_closes() {
    let graph = Graph::new();
    let bound = graph.free(16);
    let value = graph.input(
        Shape::of([1, 1, 16, 4]).freed(&[(2, bound)]),
        Element::Single,
    );
    let turned = graph.rope(value, None, 10000.0);
    let snapshot = graph.snapshot();
    let task = snapshot
        .tasks()
        .iter()
        .find(|task| task.kind == Kind::Rope)
        .expect("a rope turns the rows of the one plane the free extent holds");
    assert_eq!(
        task.segments,
        neura_abi::NO_VALUE,
        "a free extent that no ragged axis closes places every row of the one plane the tensor holds",
    );
    assert_eq!(graph.shape(turned).free(2), Some(bound.slot()));
}
