use neura_abi::Element;
use neura_graph::{Graph, Shape};
use std::panic::{AssertUnwindSafe, catch_unwind};

const PLANES: u32 = 3;
const BOUND: u32 = 12;
const DEPTH: u32 = 4;
const COLUMNS: u32 = 5;

fn refuses(action: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_grouped_product_walks_the_rows_a_ragged_axis_packs() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let left = graph.input(
        Shape::of([1, 1, BOUND, DEPTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([PLANES, 1, DEPTH, COLUMNS]),
        neura_graph::Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    graph.freeze(&[weights]);
    let out = graph.grouped_matmul(left, weights, ragged.offsets);
    assert_eq!(
        graph.shape(out).free(2),
        Some(ragged.extent.slot()),
        "the rows of a grouped product walk the free extent the ragged axis closes",
    );
    assert_eq!(
        graph.shape(out).dims(),
        [1, 1, BOUND, COLUMNS],
        "a grouped product weighs every packed row into the columns of one weight plane",
    );
}

#[test]
fn a_grouped_product_walks_the_segments_a_ragged_axis_closes() {
    let graph: Graph<'static> = Graph::new();
    let offsets = graph.input(Shape::vector(PLANES + 1), Element::Single);
    let loose = graph.free(BOUND);
    let left = graph.input(
        Shape::of([1, 1, BOUND, DEPTH]).freed(&[(2, loose)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([PLANES, 1, DEPTH, COLUMNS]),
        neura_graph::Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.grouped_matmul(left, weights, offsets);
        }),
        "a table the host writes named the segments a grouped product walks",
    );
}

#[test]
fn a_grouped_product_walks_the_extent_its_ragged_axis_closes() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let other = graph.input(Shape::vector(PLANES), Element::Single);
    let unrelated = graph.ragged(BOUND, other);
    let left = graph.input(
        Shape::of([1, 1, BOUND, DEPTH]).freed(&[(2, unrelated.extent)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([PLANES, 1, DEPTH, COLUMNS]),
        neura_graph::Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.grouped_matmul(left, weights, ragged.offsets);
        }),
        "the offsets of one ragged axis weighed the packed rows another ragged axis closes",
    );
}

#[test]
fn a_grouped_product_weighs_one_plane_of_every_segment() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let left = graph.input(
        Shape::of([1, 1, BOUND, DEPTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([PLANES + 1, 1, DEPTH, COLUMNS]),
        neura_graph::Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.grouped_matmul(left, weights, ragged.offsets);
        }),
        "weights of one plane per segment weighed a segment the ragged axis closes beside them",
    );
}

#[test]
fn a_grouped_product_weighs_the_depth_of_every_packed_row() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let left = graph.input(
        Shape::of([1, 1, BOUND, DEPTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([PLANES, 1, DEPTH + 1, COLUMNS]),
        neura_graph::Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.grouped_matmul(left, weights, ragged.offsets);
        }),
        "a product of depth {DEPTH} weighed weights of depth {}",
        DEPTH + 1,
    );
}

#[test]
fn a_grouped_product_walks_one_axis_of_packed_rows() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let left = graph.input(
        Shape::of([2, 1, BOUND, DEPTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([PLANES, 1, DEPTH, COLUMNS]),
        neura_graph::Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.grouped_matmul(left, weights, ragged.offsets);
        }),
        "two planes of packed rows walked the segments of one ragged axis",
    );
}

#[test]
fn a_grouped_product_walks_the_rows_its_storage_lays_out() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let packed = graph.input(
        Shape::of([1, 1, DEPTH, BOUND]).freed(&[(3, ragged.extent)]),
        Element::Single,
    );
    let left = graph.permute(packed, [0, 1, 3, 2]);
    let weights = graph.parameter(
        Shape::of([PLANES, 1, DEPTH, COLUMNS]),
        neura_graph::Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.grouped_matmul(left, weights, ragged.offsets);
        }),
        "a view of every other row weighed the segments a ragged axis packs",
    );
}

#[test]
fn a_grouped_product_weighs_the_weights_of_static_planes() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let left = graph.input(
        Shape::of([1, 1, BOUND, DEPTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let live = graph.free(PLANES);
    let weights = graph.input(
        Shape::of([PLANES, 1, DEPTH, COLUMNS]).freed(&[(0, live)]),
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.grouped_matmul(left, weights, ragged.offsets);
        }),
        "weights of a length the binding rules weighed the segments a ragged axis closes",
    );
}

#[test]
fn a_grouped_product_carries_no_gradient() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let left = graph.gradient_input(
        Shape::of([1, 1, BOUND, DEPTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([PLANES, 1, DEPTH, COLUMNS]),
        neura_graph::Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.grouped_matmul(left, weights, ragged.offsets);
        }),
        "a grouped product carried the gradient of the rows a device packs",
    );
}
