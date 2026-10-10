use neura_abi::Element;
use neura_gpu::PREFERENCE;
use neura_graph::{Graph, Init, Shape};

#[path = "support/backend.rs"]
mod backend;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const WIDTH: u32 = 8;
const ROWS: u32 = 3;

fn observations() -> Vec<f32> {
    vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
}

fn limits() -> Vec<f32> {
    vec![1.0, -1.0, 0.0, 0.0, 2.0, 2.5, 6.0, 5.0]
}

fn chosen() -> Vec<f32> {
    observations()
        .iter()
        .zip(&limits())
        .map(|(value, limit)| f32::from(value > limit))
        .collect()
}

fn positive() -> Vec<f32> {
    observations()
        .iter()
        .map(|value| f32::from(*value > 0.0))
        .collect()
}

fn scales() -> Vec<f32> {
    (0..ROWS * WIDTH)
        .map(|index| 0.25 + 0.125 * index as f32)
        .collect()
}

#[test]
fn a_comparison_turns_data_into_the_mask_of_a_masked_loss() {
    for &backend in PREFERENCE {
        let runtime = backend::open_with(backend);
        let graph = Graph::new();
        let value = graph.gradient_input(Shape::vector(WIDTH), Element::Single);
        let limit = graph.input(Shape::vector(WIDTH), Element::Single);
        let mask = graph.greater(value, limit);
        let loss = graph.sum(graph.mul(value, mask));
        graph.retain(mask);
        graph.retain(loss);
        let gradients = graph.backward(loss);
        let slope = gradients.of(value);
        graph.retain(slope);
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        let (values, bounds) = (observations(), limits());
        runtime.write(&program, value, &values);
        runtime.write(&program, limit, &bounds);
        runtime.run(&program);
        let expected = chosen();
        let total = values
            .iter()
            .zip(&expected)
            .map(|(value, mask)| value * mask)
            .sum::<f32>();
        assert_close(&runtime.read(&program, mask), &expected, 0.0);
        assert_close(&runtime.read(&program, loss), &[total], 1e-6);
        assert_close(&runtime.read(&program, slope), &expected, 0.0);
    }
}

#[test]
fn a_selection_picks_the_branch_the_condition_names_exactly() {
    for &backend in PREFERENCE {
        let runtime = backend::open_with(backend);
        let graph = Graph::new();
        let condition = graph.gradient_input(Shape::vector(WIDTH), Element::Single);
        let zeros = graph.parameter(Shape::matrix(ROWS, WIDTH), Init::Zero, Element::Single);
        let positive = graph.recip(zeros);
        let negative = graph.neg(positive);
        let picked = graph.select(condition, positive, negative);
        graph.retain(picked);
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        let chosen = chosen();
        runtime.write(&program, condition, &chosen);
        runtime.write(&program, zeros, &vec![0.0; (ROWS * WIDTH) as usize]);
        runtime.run(&program);
        let expected = (0..ROWS)
            .flat_map(|_| {
                chosen.iter().map(|picked| {
                    if *picked != 0.0 {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect::<Vec<f32>>();
        assert_eq!(
            runtime.read(&program, picked),
            expected,
            "a selection picks one branch and weighs the other to nothing",
        );
    }
}

#[test]
fn a_selection_weighs_only_the_branch_it_picked() {
    for &backend in PREFERENCE {
        let runtime = backend::open_with(backend);
        let graph = Graph::new();
        let value = graph.gradient_input(Shape::vector(WIDTH), Element::Single);
        let limit = graph.input(Shape::vector(WIDTH), Element::Single);
        let condition = graph.greater(value, limit);
        let accept = graph.parameter(
            Shape::matrix(ROWS, WIDTH),
            Init::Constant(1.0),
            Element::Single,
        );
        let reject = graph.parameter(
            Shape::matrix(ROWS, WIDTH),
            Init::Constant(-1.0),
            Element::Single,
        );
        let scale = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
        let loss = graph.sum(graph.mul(graph.select(condition, accept, reject), scale));
        let gradients = graph.backward(loss);
        let (over_accept, over_reject) = (gradients.of(accept), gradients.of(reject));
        graph.retain(over_accept);
        graph.retain(over_reject);
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        let chosen = chosen();
        let weights = scales();
        runtime.write(&program, value, &observations());
        runtime.write(&program, limit, &limits());
        runtime.write(&program, scale, &weights);
        runtime.run(&program);
        let rows = (0..ROWS)
            .flat_map(|_| chosen.iter().copied())
            .collect::<Vec<f32>>();
        let expected_accept = weights
            .iter()
            .zip(&rows)
            .map(|(weight, picked)| weight * picked)
            .collect::<Vec<f32>>();
        let expected_reject = weights
            .iter()
            .zip(&rows)
            .map(|(weight, picked)| weight * (1.0 - picked))
            .collect::<Vec<f32>>();
        assert_close(&runtime.read(&program, over_accept), &expected_accept, 1e-6);
        assert_close(&runtime.read(&program, over_reject), &expected_reject, 1e-6);
    }
}

#[test]
fn a_device_mask_sends_the_logits_it_leaves_behind_to_nothing() {
    let runtime = open();
    let graph = Graph::new();
    let logits = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let zeros = graph.fill(Shape::of([1, 1, 1, WIDTH]), 0.0);
    let mask = graph.greater(logits, zeros);
    let behind = graph.neg(graph.recip(zeros));
    let probabilities = graph.softmax(graph.select(mask, logits, behind));
    graph.retain(mask);
    graph.retain(probabilities);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let values = observations();
    runtime.write(&program, logits, &values);
    runtime.run(&program);
    let kept = positive();
    let largest = values
        .iter()
        .zip(&kept)
        .filter(|(_, mask)| **mask != 0.0)
        .map(|(value, _)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let total = values
        .iter()
        .zip(&kept)
        .filter(|(_, mask)| **mask != 0.0)
        .map(|(value, _)| (value - largest).exp())
        .sum::<f32>();
    let expected = values
        .iter()
        .zip(&kept)
        .map(|(value, mask)| {
            if *mask != 0.0 {
                (value - largest).exp() / total
            } else {
                0.0
            }
        })
        .collect::<Vec<f32>>();
    assert_close(&runtime.read(&program, mask), &kept, 0.0);
    assert_close(&runtime.read(&program, probabilities), &expected, 1e-6);
}

#[test]
fn a_selection_packs_a_half_result_the_same_way() {
    let runtime = open();
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(WIDTH), Init::Constant(1.0), Element::Half);
    let right = graph.parameter(Shape::vector(WIDTH), Init::Constant(0.0), Element::Half);
    let condition = graph.greater(left, right);
    let accept = graph.parameter(Shape::vector(WIDTH), Init::Constant(1.0), Element::Half);
    let reject = graph.parameter(Shape::vector(WIDTH), Init::Constant(-1.0), Element::Half);
    let picked = graph.select(condition, accept, reject);
    assert_eq!(graph.element(condition), Element::Half);
    assert_eq!(graph.element(picked), Element::Half);
    graph.retain(picked);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let observations = [0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 3.0];
    let limits = [0.0, 0.25, 0.0, 0.5, -2.0, 2.0, -1.0, 4.0];
    runtime.write(&program, left, &observations);
    runtime.write(&program, right, &limits);
    runtime.write(&program, accept, &observations);
    runtime.write(&program, reject, &observations.map(|value| -value));
    runtime.run(&program);
    let expected = observations
        .iter()
        .zip(&limits)
        .map(|(value, limit)| if value > limit { *value } else { -*value })
        .collect::<Vec<f32>>();
    assert_close(&runtime.read(&program, picked), &expected, 0.0);
}
