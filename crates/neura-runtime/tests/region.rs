use neura_abi::Element;
use neura_gpu::Backends;
use neura_graph::{Graph, Init, Shape};

#[path = "support/backend.rs"]
mod backend;
#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn coordinates(flat: u32, dims: [u32; 4]) -> [u32; 4] {
    [
        flat / (dims[1] * dims[2] * dims[3]),
        flat / (dims[2] * dims[3]) % dims[1],
        flat / dims[3] % dims[2],
        flat % dims[3],
    ]
}

fn flat(at: [u32; 4], dims: [u32; 4]) -> u32 {
    at[0] * dims[1] * dims[2] * dims[3] + at[1] * dims[2] * dims[3] + at[2] * dims[3] + at[3]
}

fn volume(dims: [u32; 4]) -> usize {
    dims.iter().product::<u32>() as usize
}

fn fold_reference(values: &[f32], dims: [u32; 4], axis: usize, mean: bool) -> Vec<f32> {
    let mut folded = dims;
    folded[axis] = 1;
    let mut out = vec![0.0f32; volume(folded)];
    for (index, value) in values.iter().enumerate() {
        let mut at = coordinates(index as u32, dims);
        at[axis] = 0;
        out[flat(at, folded) as usize] += value;
    }
    if mean {
        for value in &mut out {
            *value /= dims[axis] as f32;
        }
    }
    out
}

fn concat_reference(first: [u32; 4], parts: &[&[f32]], lengths: &[u32], axis: usize) -> Vec<f32> {
    let mut whole = first;
    whole[axis] = lengths.iter().sum();
    let mut out = vec![0.0f32; volume(whole)];
    let mut start = 0;
    for (part, length) in parts.iter().zip(lengths) {
        let mut source = first;
        source[axis] = *length;
        for (index, value) in part.iter().enumerate() {
            let mut at = coordinates(index as u32, source);
            at[axis] += start;
            out[flat(at, whole) as usize] = *value;
        }
        start += length;
    }
    out
}

#[test]
fn a_fold_carries_a_sum_onto_the_axis_it_names() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 4, 5]), Element::Single);
    let folded = (0..4)
        .map(|axis| graph.sum_axis(input, axis))
        .collect::<Vec<_>>();
    for value in &folded {
        graph.retain(*value);
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(120, 7);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    for (axis, value) in folded.iter().enumerate() {
        assert_close(
            &runtime.read(&program, *value),
            &fold_reference(&values, [2, 3, 4, 5], axis, false),
            1e-5,
        );
    }
}

#[test]
fn a_fold_of_a_singleton_axis_carries_the_tensor_it_names() {
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 3, 1, 5]), Element::Single);
    let folded = graph.sum_axis(input, 2);
    assert_eq!(
        folded.id(),
        input.id(),
        "a fold of a singleton axis carries the tensor it names",
    );
}

#[test]
fn a_mean_carries_the_average_onto_the_axis_it_names() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 4, 5]), Element::Single);
    let averaged = (0..4)
        .map(|axis| graph.mean_axis(input, axis))
        .collect::<Vec<_>>();
    for value in &averaged {
        graph.retain(*value);
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(120, 13);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    for (axis, value) in averaged.iter().enumerate() {
        assert_close(
            &runtime.read(&program, *value),
            &fold_reference(&values, [2, 3, 4, 5], axis, true),
            1e-5,
        );
    }
}

#[test]
fn a_mean_gradient_spreads_evenly_over_the_axis_it_reduced() {
    let runtime = open();
    for axis in 0..4u32 {
        let graph = Graph::new();
        let parameter = graph.parameter(Shape::of([2, 3, 4, 5]), Init::Zero, Element::Single);
        let loss = graph.sum(graph.mean_axis(parameter, axis));
        let gradient = graph.backward(loss).of(parameter);
        graph.retain(gradient);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.run(&program);
        let count = [2.0f32, 3.0, 4.0, 5.0][axis as usize];
        assert_close(
            &runtime.read(&program, gradient),
            &vec![1.0 / count; 120],
            1e-6,
        );
    }
}

#[test]
fn a_broadcast_spreads_over_the_axes_it_expands() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 3, 1, 5]), Element::Single);
    let whole = graph.broadcast_to(input, Shape::of([2, 3, 4, 5]));
    let singleton = graph.broadcast_to(input, Shape::of([1, 3, 1, 5]));
    graph.retain(whole);
    graph.retain(singleton);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(15, 17);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let expected = (0..120)
        .map(|index| {
            let at = coordinates(index, [2, 3, 4, 5]);
            values[flat([0, at[1], 0, at[3]], [1, 3, 1, 5]) as usize]
        })
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, whole), &expected, 1e-6);
    assert_close(&runtime.read(&program, singleton), &values, 1e-6);
}

#[test]
fn a_broadcast_gradient_folds_into_the_tensor_it_spread() {
    let runtime = open();
    let graph = Graph::new();
    let parameter = graph.parameter(Shape::of([1, 3, 1, 5]), Init::Zero, Element::Single);
    let whole = graph.broadcast_to(parameter, Shape::of([2, 3, 4, 5]));
    let loss = graph.sum(whole);
    let gradient = graph.backward(loss).of(parameter);
    graph.retain(gradient);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    assert_close(&runtime.read(&program, gradient), &[8.0; 15], 1e-6);
}

#[test]
fn a_concatenation_lands_its_operands_on_the_axis_it_names() {
    let runtime = open();
    for axis in 0..4usize {
        let graph = Graph::new();
        let mut first_dims = [2u32, 3, 4, 5];
        first_dims[axis] = 2;
        let mut second_dims = first_dims;
        second_dims[axis] = 3;
        let first = graph.input(Shape::of(first_dims), Element::Single);
        let second = graph.input(Shape::of(second_dims), Element::Single);
        let lifted = graph.mul(second, graph.fill(Shape::of(second_dims), 1.0));
        let out = graph.concat(&[first, lifted], axis as u32);
        graph.retain(out);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let first_values = random(first_dims.iter().product(), 3 + axis as u32);
        let second_values = random(second_dims.iter().product(), 11 + axis as u32);
        runtime.write(&program, first, &first_values);
        runtime.write(&program, second, &second_values);
        runtime.run(&program);
        assert_close(
            &runtime.read(&program, out),
            &concat_reference(first_dims, &[&first_values, &second_values], &[2, 3], axis),
            1e-6,
        );
    }
}

#[test]
fn a_concatenation_gradient_returns_each_region() {
    let runtime = open();
    for axis in 0..4usize {
        let graph = Graph::new();
        let mut first_dims = [2u32, 3, 4, 5];
        first_dims[axis] = 2;
        let mut second_dims = first_dims;
        second_dims[axis] = 3;
        let first = graph.parameter(Shape::of(first_dims), Init::Zero, Element::Single);
        let second = graph.parameter(Shape::of(second_dims), Init::Zero, Element::Single);
        let whole = graph.concat(&[first, second], axis as u32);
        let mut whole_dims = first_dims;
        whole_dims[axis] = 5;
        let mask = graph.parameter(Shape::of(whole_dims), Init::Zero, Element::Single);
        let loss = graph.sum(graph.mul(whole, mask));
        let gradients = graph.backward(loss);
        let first_gradient = gradients.of(first);
        let second_gradient = gradients.of(second);
        graph.retain(first_gradient);
        graph.retain(second_gradient);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let mask_values = random(whole_dims.iter().product(), 19 + axis as u32);
        runtime.write(&program, mask, &mask_values);
        runtime.run(&program);
        let mut expected_first = Vec::new();
        let mut expected_second = Vec::new();
        for (index, value) in mask_values.iter().enumerate() {
            let at = coordinates(index as u32, whole_dims);
            if at[axis] < first_dims[axis] {
                expected_first.push(*value);
            } else {
                expected_second.push(*value);
            }
        }
        assert_close(
            &runtime.read(&program, first_gradient),
            &expected_first,
            1e-6,
        );
        assert_close(
            &runtime.read(&program, second_gradient),
            &expected_second,
            1e-6,
        );
    }
}

#[test]
fn a_slice_reads_the_region_it_names() {
    let runtime = open();
    for axis in 0..4usize {
        let graph = Graph::new();
        let input = graph.input(Shape::of([2, 3, 4, 5]), Element::Single);
        let mut dims = [2u32, 3, 4, 5];
        dims[axis] = 2;
        let start = u32::from(axis == 1);
        let out = graph.slice(input, axis as u32, start, dims[axis]);
        graph.retain(out);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let values = random(120, 23 + axis as u32);
        runtime.write(&program, input, &values);
        runtime.run(&program);
        let mut expected = Vec::new();
        for (index, value) in values.iter().enumerate() {
            let at = coordinates(index as u32, [2, 3, 4, 5]);
            if at[axis] >= start && at[axis] < start + dims[axis] {
                expected.push(*value);
            }
        }
        assert_close(&runtime.read(&program, out), &expected, 1e-6);
    }
}

#[test]
fn a_slice_gradient_zeroes_every_element_outside_its_region() {
    let runtime = open();
    let graph = Graph::new();
    let parameter = graph.parameter(Shape::of([2, 3, 4, 5]), Init::Zero, Element::Single);
    let region = graph.slice(parameter, 1, 1, 2);
    let loss = graph.sum(region);
    let gradient = graph.backward(loss).of(parameter);
    graph.retain(gradient);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    let expected = (0..120)
        .map(|index| {
            let at = coordinates(index, [2, 3, 4, 5]);
            f32::from(at[1] >= 1 && at[1] < 3)
        })
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, gradient), &expected, 1e-6);
}

#[test]
fn a_slice_meets_the_concatenation_it_undoes() {
    let runtime = open();
    let graph = Graph::new();
    let whole = graph.input(Shape::matrix(4, 6), Element::Single);
    let left = graph.slice(whole, 3, 0, 4);
    let right = graph.slice(whole, 3, 2, 4);
    let joined = graph.concat(&[left, right], 3);
    graph.retain(joined);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(24, 31);
    runtime.write(&program, whole, &values);
    runtime.run(&program);
    let expected = (0..32)
        .map(|index| {
            let at = coordinates(index, [1, 1, 4, 8]);
            let column = if at[3] < 4 { at[3] } else { at[3] - 2 };
            values[(at[2] * 6 + column) as usize]
        })
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, joined), &expected, 1e-6);
}

#[test]
fn a_concatenation_packs_a_narrow_operand() {
    let runtime = open();
    let graph = Graph::new();
    let first = graph.quantized_parameter(Shape::of([2, 3, 1, 5]), Init::Zero, 0.25);
    let second = graph.quantized_parameter(Shape::of([2, 3, 2, 5]), Init::Zero, 0.25);
    let out = graph.concat(&[first, second], 2);
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let first_values = random(30, 5);
    let second_values = random(60, 6);
    runtime.write(&program, first, &first_values);
    runtime.write(&program, second, &second_values);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, out),
        &concat_reference([2, 3, 1, 5], &[&first_values, &second_values], &[1, 2], 2),
        0.25,
    );
}

#[test]
fn a_narrow_slice_carries_the_words_it_copied() {
    let runtime = open();
    let graph = Graph::new();
    let parameter = graph.quantized_parameter(Shape::of([2, 3, 4, 5]), Init::Zero, 0.25);
    let region = graph.slice(parameter, 2, 1, 2);
    graph.retain(region);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(120, 37);
    runtime.write(&program, parameter, &values);
    runtime.run(&program);
    let mut expected = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let at = coordinates(index as u32, [2, 3, 4, 5]);
        if at[2] >= 1 && at[2] < 3 {
            expected.push(*value);
        }
    }
    assert_close(&runtime.read(&program, region), &expected, 0.25);
}

#[test]
fn a_concatenation_carries_a_gradient_through_its_words() {
    let runtime = open();
    let graph = Graph::new();
    let first = graph.parameter(Shape::of([2, 3, 1, 5]), Init::Zero, Element::Half);
    let second = graph.parameter(Shape::of([2, 3, 2, 5]), Init::Zero, Element::Half);
    let out = graph.concat(&[first, second], 2);
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let first_gradient = gradients.of(first);
    let second_gradient = gradients.of(second);
    graph.retain(first_gradient);
    graph.retain(second_gradient);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    assert_close(&runtime.read(&program, first_gradient), &[1.0; 30], 1e-3);
    assert_close(&runtime.read(&program, second_gradient), &[1.0; 60], 1e-3);
}

#[test]
fn every_backend_runs_a_region_and_an_axis() {
    for backends in Backends::PLATFORM {
        let runtime = backend::open_with(backends);
        let graph = Graph::new();
        let input = graph.input(Shape::of([2, 3, 4, 5]), Element::Single);
        let folded = graph.sum_axis(input, 1);
        let spread = graph.broadcast_to(folded, Shape::of([2, 3, 4, 5]));
        let left = graph.slice(spread, 2, 0, 2);
        let right = graph.slice(spread, 2, 2, 2);
        let joined = graph.concat(&[left, right], 2);
        graph.retain(joined);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let values = random(120, 3);
        runtime.write(&program, input, &values);
        runtime.run(&program);
        let expected = (0..120)
            .map(|index| {
                let at = coordinates(index, [2, 3, 4, 5]);
                (0..3)
                    .map(|channel| {
                        values[flat([at[0], channel, at[2], at[3]], [2, 3, 4, 5]) as usize]
                    })
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_close(&runtime.read(&program, joined), &expected, 1e-5);
    }
}

#[test]
fn a_concatenation_refuses_shapes_that_do_not_meet() {
    let graph = Graph::new();
    let left = graph.input(Shape::matrix(2, 3), Element::Single);
    let right = graph.input(Shape::matrix(3, 3), Element::Single);
    assert!(refuses(|| {
        let _ = graph.concat(&[left, right], 1);
    }));
    assert!(refuses(|| {
        let _ = graph.concat(&[left], 4);
    }));
    assert!(refuses(|| {
        let _ = graph.slice(left, 1, 1, 3);
    }));
    assert!(refuses(|| {
        let _ = graph.slice(left, 1, 2, 0);
    }));
}
