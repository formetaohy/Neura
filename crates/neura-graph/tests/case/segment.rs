use neura_abi::{Element, Kind, NO_VALUE};
use neura_graph::{Graph, Shape};

const PLANES: u32 = 4;
const BOUND: u32 = 16;
const TALL: u32 = 1024;
const WIDTH: u32 = 3;

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_per_plane_sum_packs_the_rows_of_every_plane_into_one_axis() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(TALL, lengths);
    let cache = graph.input(
        Shape::of([1, 1, TALL, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let total = graph.segment_sum(cache, ragged);
    assert_eq!(graph.shape(total), Shape::of([1, PLANES, 1, WIDTH]));
    assert_eq!(graph.element(total), Element::Single);
    let snapshot = graph.snapshot();
    let tasks = snapshot.tasks();
    let walked = tasks
        .iter()
        .find(|task| task.kind == Kind::SegmentSum)
        .expect("a per-plane sum walks the rows a ragged axis packs");
    assert_eq!(walked.segments, ragged.offsets.id());
    assert_eq!(snapshot.values()[walked.out as usize].storage, walked.out);
    assert!(
        tasks
            .iter()
            .any(|task| task.kind == Kind::SumAxis && task.inputs[0] == walked.out),
        "a per-plane sum folds the packs every plane walks",
    );
}

#[test]
fn a_per_plane_sum_walks_the_planes_the_lengths_lay_out() {
    let graph = Graph::new();
    let heads = 2u32;
    let batch = 3u32;
    let lengths = graph.input(Shape::of([heads, batch, 1, 1]), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let total = graph.segment_sum(cache, ragged);
    assert_eq!(graph.shape(total), Shape::of([heads, batch, 1, WIDTH]));
}

#[test]
fn a_per_plane_sum_carries_the_planes_a_binding_rules() {
    let graph = Graph::new();
    let live = graph.free(PLANES);
    let lengths = graph.input(Shape::of([1, PLANES]).freed(&[(3, live)]), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let extent = graph.free(BOUND);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, extent)]),
        Element::Single,
    );
    assert!(refuses(|| {
        graph.segment_sum(cache, ragged);
    }));
    let packed = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let total = graph.segment_sum(packed, ragged);
    assert_eq!(graph.shape(total).free(1), Some(live.slot()));
}

#[test]
fn a_per_plane_sum_reads_the_table_a_ragged_axis_published() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let bare = graph.input(Shape::matrix(PLANES + 1, 1), Element::Single);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.segment_sum(
                cache,
                neura_graph::Ragged {
                    extent: ragged.extent,
                    offsets: bare,
                },
            );
        }),
        "a host table named the planes of a packed axis",
    );
    let other = graph.input(Shape::vector(PLANES), Element::Single);
    let unrelated = graph.ragged(BOUND, other);
    assert!(
        refuses(|| {
            graph.segment_sum(
                cache,
                neura_graph::Ragged {
                    extent: unrelated.extent,
                    offsets: ragged.offsets,
                },
            );
        }),
        "the extent of one ragged axis named the axis another ragged axis packs",
    );
    let plain = graph.input(Shape::of([1, 1, BOUND, WIDTH]), Element::Single);
    assert!(
        refuses(|| {
            graph.segment_sum(plain, ragged);
        }),
        "a per-plane sum packed no plane of rows",
    );
    let planed = graph.input(Shape::of([PLANES, 1, BOUND, WIDTH]), Element::Single);
    assert!(
        refuses(|| {
            graph.segment_sum(planed, ragged);
        }),
        "a per-plane sum walked an axis of rows that is not packed",
    );
}

#[test]
fn a_fold_over_the_axis_a_ragged_axis_packs_is_refused() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let cache = graph.input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.sum_axis(cache, 2);
        }),
        "a fold walked the rows of every plane a ragged axis packs",
    );
    let other = graph.segment_sum(cache, ragged);
    graph.sum_axis(other, 1);
}

#[test]
fn a_per_plane_sum_trains_the_rows_of_every_plane() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(TALL, lengths);
    let cache = graph.gradient_input(
        Shape::of([1, 1, TALL, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let total = graph.segment_sum(cache, ragged);
    let weight = graph.parameter(
        Shape::of([1, PLANES, 1, WIDTH]),
        neura_graph::Init::Zero,
        Element::Single,
    );
    let loss = graph.sum(graph.mul(total, weight));
    let collected = graph.backward(loss);
    let gradient = collected.of(cache);
    assert_eq!(graph.shape(gradient), graph.shape(cache));
    let snapshot = graph.snapshot();
    let tasks = snapshot.tasks();
    assert_eq!(
        tasks
            .iter()
            .filter(|task| task.kind == Kind::SegmentSum)
            .count(),
        1,
        "the forward pass walks the planes a ragged axis closes",
    );
    assert_eq!(
        tasks
            .iter()
            .filter(|task| task.kind == Kind::SumAxis)
            .count(),
        2,
        "the forward pass and the backward pass each fold the packs of a plane",
    );
    assert_eq!(
        tasks.iter().filter(|task| task.kind == Kind::Rows).count(),
        1,
        "the backward pass maps every row to the plane it belongs to",
    );
    assert!(
        tasks.iter().any(|task| task.kind == Kind::Gather
            && task.out == gradient.id()
            && task.inputs[1] != NO_VALUE),
        "the backward pass hands every row the sum of its plane",
    );
}

#[test]
fn a_per_plane_sum_hands_one_number_to_every_plane_the_lengths_walk() {
    let graph = Graph::new();
    for (dims, expected) in [
        ([1, 1, PLANES, 1], [1, PLANES, 1, WIDTH]),
        ([1, PLANES, 1, 1], [1, PLANES, 1, WIDTH]),
        ([PLANES, 1, 1, 1], [PLANES, 1, 1, WIDTH]),
        ([1, 1, PLANES, 2], [1, PLANES * 2, 1, WIDTH]),
        ([PLANES, 2, 1, 1], [PLANES, 2, 1, WIDTH]),
    ] {
        let lengths = graph.input(Shape::of(dims), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.input(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let total = graph.segment_sum(cache, ragged);
        assert_eq!(graph.shape(total).dims(), expected, "{dims:?}");
    }
    let live = graph.free(PLANES * 2);
    for (axis, dims) in [(1u32, [1, PLANES * 2, 1, 1]), (2, [1, 1, PLANES * 2, 1])] {
        let lengths = graph.input(Shape::of(dims).freed(&[(axis, live)]), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.input(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let total = graph.segment_sum(cache, ragged);
        assert_eq!(graph.shape(total).dims(), [1, PLANES * 2, 1, WIDTH]);
        assert_eq!(graph.shape(total).free(1), Some(live.slot()));
    }
}

#[test]
fn a_per_plane_sum_refuses_lengths_that_lay_their_planes_over_three_axes() {
    let graph = Graph::new();
    for dims in [[2, 1, 3, 1], [1, 2, 1, 3], [2, 3, 2, 1], [1, 1, 3, 2]] {
        let lengths = graph.input(Shape::of(dims), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.input(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        if dims[0] == 1 && dims[1] == 1 {
            graph.segment_sum(cache, ragged);
            continue;
        }
        assert!(
            refuses(|| {
                graph.segment_sum(cache, ragged);
            }),
            "the planes of {dims:?} lay over three axes",
        );
    }
}
