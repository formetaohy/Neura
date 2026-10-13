use neura_abi::{EXACT_WALK_LIMIT, Element, Kind};
use neura_graph::{Graph, Shape};

use std::panic::{AssertUnwindSafe, catch_unwind};

fn refuses(action: impl FnOnce()) -> bool {
    catch_unwind(AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_mean_of_a_fixed_axis_weighs_the_number_the_plan_carries() {
    let graph = Graph::new();
    let values = graph.input(Shape::of([1, 1, 4, 8]), Element::Single);
    let mean = graph.mean_axis(values, 2);
    let tasks = graph.snapshot().tasks().to_vec();
    assert!(
        !tasks.iter().any(|task| task.kind == Kind::Length),
        "a mean over axis 2 of a fixed shape weighs the number the plan froze, and a device reads no length",
    );
    assert!(
        tasks
            .iter()
            .any(|task| task.kind == Kind::SumChunk || task.kind == Kind::SumAxis),
        "a mean walks the numbers it weighs",
    );
    assert!(
        graph.shape(mean).free(2).is_none(),
        "a mean hands one number to every position of the axes it leaves",
    );
}

#[test]
fn a_mean_of_a_free_axis_weighs_the_length_its_binding_holds() {
    let graph = Graph::new();
    let extent = graph.free(8);
    let values = graph.input(
        Shape::of([1, 1, 8, 4]).freed(&[(2, extent)]),
        Element::Single,
    );
    let mean = graph.mean_axis(values, 2);
    let tasks = graph.snapshot().tasks().to_vec();
    let lengths = tasks
        .iter()
        .filter(|task| task.kind == Kind::Length)
        .collect::<Vec<_>>();
    assert_eq!(
        lengths.len(),
        1,
        "a mean over a free axis asks the device for the length it walks",
    );
    assert_eq!(lengths[0].inputs[0], values.id());
    assert_eq!(
        lengths[0].slot, 2,
        "the length a mean reads names the axis it folds",
    );
    assert_eq!(
        graph.snapshot().values()[mean.id() as usize].shape.free(2),
        None,
        "a mean folds the axis it weighs",
    );
}

#[test]
fn a_length_of_a_fixed_axis_is_a_number_the_plan_carries() {
    let graph = Graph::new();
    let values = graph.input(Shape::of([1, 1, 4, 8]), Element::Single);
    let length = graph.length(values, 2);
    assert!(
        graph
            .snapshot()
            .tasks()
            .iter()
            .all(|task| task.kind != Kind::Length),
        "a fixed length is one number the plan froze, and the device reads no record",
    );
    assert_eq!(graph.shape(length), Shape::scalar());
}

#[test]
fn a_fold_of_one_number_hands_back_the_number_it_holds() {
    let graph = Graph::new();
    let values = graph.input(Shape::of([1, 1, 4, 1]), Element::Single);
    assert_eq!(
        graph.sum_rows(values).id(),
        values.id(),
        "a fold over the one number a fixed axis holds hands that number back",
    );
    assert_eq!(
        graph.sum_axis(values, 3).id(),
        values.id(),
        "a fold names one number whichever entry point reaches the axis",
    );
    assert_eq!(
        graph.mean_axis(values, 3).id(),
        values.id(),
        "a mean over the one number a fixed axis holds weighs it by one",
    );
}

#[test]
fn a_fold_of_a_free_axis_of_one_number_walks_the_length_its_binding_holds() {
    let graph = Graph::new();
    let single = graph.free(1);
    let values = graph.input(
        Shape::of([1, 1, 4, 1]).freed(&[(3, single)]),
        Element::Single,
    );
    let folded = graph.sum_rows(values);
    let tasks = graph.snapshot().tasks().to_vec();
    let folds = tasks
        .iter()
        .filter(|task| task.kind == Kind::SumAxis)
        .collect::<Vec<_>>();
    assert_eq!(
        folds.len(),
        1,
        "a fold over a free axis of one number walks the length the binding holds",
    );
    assert_eq!(folds[0].inputs[0], values.id());
    assert_eq!(folds[0].slot, 3, "the fold names the axis it walks");
    assert!(
        graph.shape(folded).free(3).is_none(),
        "a fold lands on the axis it reduces",
    );
    let mean = graph.mean_axis(values, 3);
    let tasks = graph.snapshot().tasks().to_vec();
    let lengths = tasks
        .iter()
        .filter(|task| task.kind == Kind::Length)
        .collect::<Vec<_>>();
    assert_eq!(
        lengths.len(),
        1,
        "a mean over a free axis of one number divides by the length the binding holds",
    );
    assert_eq!(lengths[0].slot, 3);
    assert!(
        graph.shape(mean).free(3).is_none(),
        "a mean lands on the axis it weighs",
    );
}

#[test]
fn a_length_beyond_the_numbers_a_device_counts_exactly_is_refused() {
    let graph = Graph::new();
    let extent = graph.free(EXACT_WALK_LIMIT + 1);
    let values = graph.input(
        Shape::of([1, 1, EXACT_WALK_LIMIT + 1, 1]).freed(&[(2, extent)]),
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.length(values, 2);
        }),
        "a device counts the numbers a single precision word holds exactly",
    );
    assert!(
        refuses(|| {
            graph.mean_axis(values, 2);
        }),
        "a mean divides by the length a device counts exactly",
    );
}
