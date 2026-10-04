use neura_abi::{Element, Kind};
use neura_graph::{Graph, Residency, Shape};

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
