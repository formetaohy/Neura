use neura_abi::{Element, Kind};
use neura_graph::{AttentionOptions, Graph, Residency, Shape};

use std::panic::{AssertUnwindSafe, catch_unwind};

fn refuses(action: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_device_authored_extent_counts_one_number() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 4, 1]), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, 4, 8]), Element::Single);
    assert!(
        refuses(|| {
            graph.trim(tokens, 2, probe);
        }),
        "a probe of four numbers authored the length of a tensor",
    );
}

#[test]
fn a_trim_walks_the_rows_of_the_tensor_it_owns() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 4, 1]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let tokens = graph.input(Shape::of([1, 1, 4, 8]), Element::Single);
    let live = graph.trim(tokens, 2, count);
    assert!(
        graph.shape(live).free(2).is_some(),
        "a trimmed tensor walks the free extent its count authors",
    );
    assert_eq!(graph.shape(live).dims(), graph.shape(tokens).dims());
    assert_eq!(
        graph.snapshot().values()[live.id() as usize].storage,
        tokens.id(),
        "a trimmed tensor holds no storage of its own",
    );
}

#[test]
fn a_count_the_host_writes_gains_a_task_of_its_own() {
    let graph = Graph::new();
    let count = graph.input(Shape::scalar(), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, 4, 8]), Element::Single);
    let tasks = graph.task_count();
    let live = graph.trim(tokens, 2, count);
    assert!(
        graph.shape(live).free(2).is_some(),
        "a trimmed tensor walks a free extent",
    );
    assert_eq!(
        graph.task_count(),
        tasks + 1,
        "a count no task of the graph writes is materialized into one task the device runs",
    );
    assert_eq!(
        graph
            .snapshot()
            .values()
            .iter()
            .filter(|info| info.residency == Residency::Derived)
            .count(),
        1,
        "the counted tensor of a host written count is a tensor of the plan",
    );
    assert_eq!(
        graph.snapshot().values()[count.id() as usize].residency,
        Residency::Input,
        "the tensor the host writes holds the count it wrote",
    );
}

#[test]
fn a_trainable_tensor_walks_the_extent_a_device_count_authors() {
    let graph = Graph::new();
    let count = graph.input(Shape::scalar(), Element::Single);
    let trainable = graph.gradient_input(Shape::of([1, 1, 4, 8]), Element::Single);
    let live = graph.trim(trainable, 2, count);
    let loss = graph.sum(graph.mul(live, live));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(trainable);
    assert_eq!(
        graph.shape(gradient),
        graph.shape(trainable),
        "the gradient of a trimmed tensor lands in the layout of the tensor that owns its storage",
    );
    assert!(
        graph
            .snapshot()
            .tasks()
            .iter()
            .any(|task| task.kind == Kind::Extend),
        "the gradient of a trimmed tensor is landed by a task that extends it to the bounds it walks",
    );
}

#[test]
fn a_gradient_reaches_a_prefix_only_through_the_layout_trim_hands_out() {
    let graph = Graph::new();
    let count = graph.input(Shape::scalar(), Element::Single);
    let trainable = graph.gradient_input(Shape::of([1, 1, 4, 8]), Element::Single);
    let live = graph.trim(trainable, 2, count);
    let permuted = graph.permute(live, [0, 1, 3, 2]);
    let other = graph.gradient_input(graph.shape(permuted), Element::Single);
    let loss = graph.sum(graph.mul(permuted, other));
    assert!(
        refuses(|| {
            graph.backward(loss);
        }),
        "a gradient reached a view that reorders the prefix a trim hands out",
    );
}

#[test]
fn a_static_operand_stands_beside_a_free_extent_of_its_own_bound() {
    let graph = Graph::new();
    let tokens = graph.free(4);
    let rows = graph.input(
        Shape::of([1, 1, 4, 8]).freed(&[(2, tokens)]),
        Element::Single,
    );
    let position = graph.input(Shape::of([1, 1, 4, 8]), Element::Single);
    let sum = graph.add(rows, position);
    assert_eq!(
        graph.shape(sum).free(2),
        Some(tokens.slot()),
        "a static operand as long as the bound stands beside every step the walk names",
    );
    let columns = graph.free(8);
    let wide = graph.input(
        Shape::of([1, 1, 2, 8]).freed(&[(3, columns)]),
        Element::Single,
    );
    let scale = graph.input(Shape::of([1, 1, 1, 8]), Element::Single);
    let scaled = graph.mul(wide, scale);
    assert_eq!(
        graph.shape(scaled).free(3),
        Some(columns.slot()),
        "a row of static numbers stands beside a walk of the columns it holds",
    );
    assert!(
        refuses(|| {
            let short = graph.input(Shape::of([1, 1, 2, 8]), Element::Single);
            graph.add(rows, short);
        }),
        "a static operand shorter than the bound names no number of the steps the walk may reach",
    );
    let depth = graph.free(8);
    let left = graph.input(
        Shape::of([1, 1, 4, 8]).freed(&[(3, depth)]),
        Element::Single,
    );
    let right = graph.input(Shape::of([1, 1, 8, 4]), Element::Single);
    assert!(
        refuses(|| {
            graph.matmul(left, right);
        }),
        "a product walks the depth of both operands through one count, and a static operand as long as the bound holds numbers of a depth no binding rules",
    );
    let planes = graph.free(4);
    let batched = graph.input(
        Shape::of([4, 1, 4, 8]).freed(&[(0, planes)]),
        Element::Single,
    );
    let shared = graph.input(Shape::of([4, 1, 8, 4]), Element::Single);
    assert!(
        refuses(|| {
            graph.matmul(batched, shared);
        }),
        "a product walks the planes of both operands through one count, and a static operand as long as the bound holds numbers of a plane no binding rules",
    );
    let width = graph.free(8);
    let query = graph.input(
        Shape::of([1, 1, 4, 8]).freed(&[(3, width)]),
        Element::Single,
    );
    let keys = graph.input(Shape::of([1, 1, 4, 8]), Element::Single);
    assert!(
        refuses(|| {
            graph.attention(
                query,
                keys,
                keys,
                AttentionOptions {
                    scale: Some(graph.knob(1.0)),
                    causal: false,
                    origin: None,
                    segments: None,
                    reach: None,
                    query_segments: None,
                },
            );
        }),
        "an attention scores the width of every key against the width of a query, and a static width holds numbers of a walk no binding rules",
    );
}

#[test]
fn a_device_authored_extent_cuts_one_walk() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 4, 1]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let planes = graph.input(Shape::of([2, 1, 4, 8]), Element::Single);
    assert!(
        refuses(|| {
            graph.trim(planes, 2, count);
        }),
        "a device count cut the rows of every plane a batch walks",
    );
    let batch = graph.input(Shape::of([4, 1, 4, 8]), Element::Single);
    let live = graph.trim(batch, 0, count);
    assert!(
        graph.shape(live).free(0).is_some(),
        "a device count cuts the planes a tensor walks",
    );
    assert_eq!(
        graph.shape(live).dims(),
        graph.shape(batch).dims(),
        "a cut plane keeps the bound its graph declares",
    );
}
