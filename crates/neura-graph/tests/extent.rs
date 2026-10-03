use neura_abi::Element;
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
fn a_tensor_that_trains_walks_no_device_authored_extent() {
    let graph = Graph::new();
    let count = graph.input(Shape::scalar(), Element::Single);
    let trainable = graph.gradient_input(Shape::of([1, 1, 4, 8]), Element::Single);
    assert!(
        refuses(|| {
            graph.trim(trainable, 2, count);
        }),
        "a tensor whose gradient a device counted extent would shape was trimmed",
    );
}
