use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Shape};

use std::panic::{AssertUnwindSafe, catch_unwind};

const PLANES: u32 = 4;
const BOUND: u32 = 16;
const WIDTH: u32 = 2;

fn refuses(action: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(action)).is_err()
}

fn causal<'g>(segments: neura_graph::Value<'g>) -> AttentionOptions<'g> {
    AttentionOptions {
        scale: 1.0,
        causal: false,
        origin: None,
        segments: Some(segments),
        reach: None,
    }
}

#[test]
fn a_segmented_attention_walks_the_offsets_a_ragged_axis_closes() {
    let graph = Graph::new();
    let count = graph.input(Shape::scalar(), Element::Single);
    let extent = graph.counted(BOUND, count);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, extent)]),
        Element::Single,
    );
    let offsets = graph.input(Shape::vector(PLANES + 1), Element::Single);
    let query = graph.input(Shape::of([1, PLANES, 1, WIDTH]), Element::Single);
    assert!(
        refuses(|| {
            graph.attention(query, cache, cache, causal(offsets));
        }),
        "a table the host writes named the key spans of a device authored packed axis",
    );
}

#[test]
fn a_segmented_attention_walks_the_extent_a_ragged_axis_closes() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let other = graph.input(Shape::vector(PLANES), Element::Single);
    let unrelated = graph.ragged(BOUND, other);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, unrelated.extent)]),
        Element::Single,
    );
    let query = graph.input(Shape::of([1, PLANES, 1, WIDTH]), Element::Single);
    assert!(
        refuses(|| {
            graph.attention(query, cache, cache, causal(ragged.offsets));
        }),
        "the offsets of one ragged axis named the spans of the packed axis another ragged axis closes",
    );
}

#[test]
fn a_segmented_attention_walks_the_planes_a_ragged_axis_closes() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let count = graph.input(Shape::scalar(), Element::Single);
    let live = graph.trim(lengths, 3, count);
    let ragged = graph.ragged(BOUND, live);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(Shape::of([1, PLANES, 1, WIDTH]), Element::Single);
    assert!(
        refuses(|| {
            graph.attention(query, cache, cache, causal(ragged.offsets));
        }),
        "a query of four planes walked the segments a ragged axis closes over a count the device authors",
    );
}

#[test]
fn a_segmented_attention_walks_the_planes_its_ragged_axis_closes() {
    let graph = Graph::new();
    let count = graph.input(Shape::scalar(), Element::Single);
    let planes = graph.counted(PLANES, count);
    let lengths = graph.input(
        Shape::of([1, 1, 1, PLANES]).freed(&[(3, planes)]),
        Element::Single,
    );
    let ragged = graph.ragged(BOUND, lengths);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(
        Shape::of([1, PLANES, 1, WIDTH]).freed(&[(1, planes)]),
        Element::Single,
    );
    let out = graph.attention(query, cache, cache, causal(ragged.offsets));
    assert_eq!(
        graph.shape(out).free(1),
        Some(planes.slot()),
        "the planes of a segmented attention walk the free extent the ragged axis counts",
    );
    assert_eq!(
        graph.shape(ragged.offsets).elements(),
        PLANES + 1,
        "a ragged axis closes one offset per plane and one that ends the last",
    );
}

#[test]
fn a_segmented_attention_weighs_one_segment_of_every_plane_it_walks() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let heads = graph.input(Shape::of([2, 2, 1, WIDTH]), Element::Single);
    let out = graph.attention(heads, cache, cache, causal(ragged.offsets));
    assert_eq!(
        graph.shape(out).dims(),
        [2, 2, 1, WIDTH],
        "the planes of a head group walk the segments a ragged axis closes one after another",
    );
    let query = graph.input(Shape::of([1, 2, 1, WIDTH]), Element::Single);
    assert!(
        refuses(|| {
            graph.attention(query, cache, cache, causal(ragged.offsets));
        }),
        "a query of two planes walked the four segments a ragged axis closes, and the device reads a segment by the very plane the query walks",
    );
}

#[test]
fn a_segmented_attention_walks_the_planes_a_packed_query_holds() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let out = graph.attention(query, cache, cache, causal(ragged.offsets));
    assert_eq!(
        graph.shape(out).dims(),
        [1, 1, BOUND, WIDTH],
        "a packed query packs the rows of every plane into the axis its keys pack",
    );
    assert_eq!(
        graph.shape(out).free(2),
        Some(ragged.extent.slot()),
        "the rows of a packed query walk the very extent the offsets close",
    );
}

#[test]
fn a_cursor_places_no_row_of_a_packed_query() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let cursor = graph.input(Shape::of([1, PLANES, 1, 1]), Element::Single);
    assert!(
        refuses(|| {
            graph.attention(
                query,
                cache,
                cache,
                AttentionOptions {
                    scale: 1.0,
                    causal: true,
                    origin: Some(cursor),
                    segments: Some(ragged.offsets),
                    reach: None,
                },
            );
        }),
        "a cursor placed the rows of a packed query, and the row of a plane is the position the packing already names",
    );
}

#[test]
fn a_packed_query_holds_no_plane_beside_the_rows_it_packs() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(
        Shape::of([2, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.attention(query, cache, cache, causal(ragged.offsets));
        }),
        "a query packed the rows of every plane through two planes beside the axis its offsets close, and the row a plane packs belongs to the plane its offset reaches",
    );
}
