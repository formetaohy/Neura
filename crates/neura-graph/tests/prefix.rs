use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_prefix_sum_walks_the_numbers_a_tensor_holds() {
    let graph = Graph::new();
    let rows = graph.free(16);
    let value = graph.input(Shape::matrix(16, 1).freed(&[(2, rows)]), Element::Single);
    let prefix = graph.prefix_sum(value);
    assert_eq!(graph.shape(prefix.exclusive), graph.shape(value));
    assert_eq!(graph.shape(prefix.exclusive).free(2), Some(rows.slot()));
    assert!(graph.shape(prefix.total).is_scalar());
}

#[test]
fn a_prefix_sum_refuses_a_walk_a_device_cannot_sum() {
    let graph = Graph::new();
    let value = graph.input(Shape::matrix(8, 4), Element::Single);
    let permuted = graph.permute(value, [0, 1, 3, 2]);
    assert!(
        refuses(|| {
            graph.prefix_sum(permuted);
        }),
        "a prefix sum walked a view",
    );
    let half = graph.cast(value, Element::Half);
    assert!(
        refuses(|| {
            graph.prefix_sum(half);
        }),
        "a prefix sum summed narrow storage",
    );
    let wide = graph.input(Shape::matrix((1 << 24) + 1, 1), Element::Single);
    assert!(
        refuses(|| {
            graph.prefix_sum(wide);
        }),
        "a prefix sum walked more numbers than a device sums exactly",
    );
}

#[test]
fn a_prefix_sum_is_no_ragged_axis() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let prefix = graph.prefix_sum(lengths);
    let cache = graph.resident(Shape::of([1, 1, 16, 4]), Element::Single);
    let query = graph.input(Shape::of([1, 4, 1, 4]), Element::Single);
    assert!(
        refuses(|| {
            graph.attention(
                query,
                cache,
                cache,
                AttentionOptions {
                    scale: Some(graph.knob(0.5)),
                    causal: true,
                    origin: None,
                    segments: Some(prefix.exclusive),
                    reach: None,
                    query_segments: None,
                },
            );
        }),
        "a prefix sum served as the offsets a ragged axis closes",
    );
}

#[test]
fn a_compaction_holds_the_rows_a_mask_names() {
    let graph = Graph::new();
    let rows = graph.free(8);
    let mask = graph.input(Shape::matrix(8, 1).freed(&[(2, rows)]), Element::Single);
    let compacted = graph.compact(mask);
    let indices = graph.shape(compacted.indices);
    assert_eq!(indices.dims()[2], 8);
    assert!(indices.dims()[3] == 1);
    assert!(
        indices.free(2).is_some(),
        "a compaction walks the rows its count holds",
    );
    assert_ne!(indices.free(2), Some(rows.slot()));
    assert!(graph.shape(compacted.count).is_scalar());
}

#[test]
fn a_compaction_refuses_a_mask_that_weighs_more_than_a_row() {
    let graph = Graph::new();
    let panel = graph.input(Shape::matrix(8, 2), Element::Single);
    assert!(
        refuses(|| {
            graph.compact(panel);
        }),
        "a compaction weighed two flags per row",
    );
    let width = graph.free(2);
    let dynamic = graph.input(Shape::matrix(8, 2).freed(&[(3, width)]), Element::Single);
    assert!(
        refuses(|| {
            graph.compact(dynamic);
        }),
        "a compaction weighed rows of a free width",
    );
}
