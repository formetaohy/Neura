use neura_abi::Element;
use neura_graph::{Graph, Init, Pool, Shape, Window};

#[path = "support/pooling.rs"]
mod pooling;
#[path = "support/mod.rs"]
mod support;

use pooling::{max_pool2d_gradient, pool2d_reference};
use support::{assert_close, open};

fn samples(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}

#[test]
fn a_max_pool_takes_the_maximum_of_the_window_it_walks() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 6, 6]), Element::Single);
    let window = Window::new([2, 2], [2, 2], [0, 0]);
    let pooled = graph.pool2d(input, window, Pool::Max);
    assert_eq!(pooled.shape(), Shape::of([2, 3, 3, 3]));
    graph.retain(pooled);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = samples(216, 53);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let expected = pool2d_reference(&values, [2, 3, 6, 6], window, true);
    assert_close(&runtime.read(&program, pooled), &expected, 1e-4);
}

#[test]
fn a_mean_pool_averages_the_window_it_pads() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 7, 7]), Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1]);
    let pooled = graph.pool2d(input, window, Pool::Mean);
    assert_eq!(pooled.shape(), Shape::of([2, 3, 4, 4]));
    graph.retain(pooled);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = samples(294, 59);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let expected = pool2d_reference(&values, [2, 3, 7, 7], window, false);
    assert_close(&runtime.read(&program, pooled), &expected, 1e-4);
}

#[test]
fn both_pools_ride_one_tape() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 8, 8]), Element::Single);
    let coarse = Window::new([2, 2], [2, 2], [0, 0]);
    let wide = Window::new([3, 3], [2, 2], [1, 1]);
    let taken = graph.pool2d(input, coarse, Pool::Max);
    let averaged = graph.pool2d(input, wide, Pool::Mean);
    assert_eq!(taken.shape(), Shape::of([2, 3, 4, 4]));
    assert_eq!(averaged.shape(), Shape::of([2, 3, 4, 4]));
    let combined = graph.add(taken, averaged);
    graph.retain(combined);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(runtime.declared_kernels(), 1);
    let values = samples(384, 61);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let mut expected = pool2d_reference(&values, [2, 3, 8, 8], coarse, true);
    let averaged_reference = pool2d_reference(&values, [2, 3, 8, 8], wide, false);
    for (value, addition) in expected.iter_mut().zip(averaged_reference) {
        *value += addition;
    }
    assert_close(&runtime.read(&program, combined), &expected, 1e-4);
}

#[test]
fn a_mean_pooled_gradient_matches_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.parameter(
        Shape::of([1, 3, 6, 6]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let window = Window::new([2, 2], [2, 2], [0, 0]);
    let pooled = graph.pool2d(input, window, Pool::Mean);
    let scale = graph.input(Shape::of([1, 3, 3, 3]), Element::Single);
    let loss = graph.sum(graph.mul(pooled, scale));
    let gradients = graph.backward(loss);
    let slope = gradients.of(input);
    graph.retain(slope);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        scale,
        &(0..27)
            .map(|index| (index as f32 * 0.037).sin() * 0.5)
            .collect::<Vec<_>>(),
    );
    runtime.run(&program);
    let values = runtime.read(&program, input);
    let analytic = runtime.read(&program, slope);
    for element in (0..values.len()).step_by(values.len() / 4) {
        let step = 0.01 * values[element].abs().max(0.1);
        let mut probe = values.clone();
        probe[element] += step;
        runtime.write(&program, input, &probe);
        runtime.run(&program);
        let high = runtime.read(&program, loss)[0];
        probe[element] -= 2.0 * step;
        runtime.write(&program, input, &probe);
        runtime.run(&program);
        let low = runtime.read(&program, loss)[0];
        let numeric = (high - low) / (2.0 * step);
        assert!(
            (numeric - analytic[element]).abs()
                <= 1e-3 + 1e-2 * analytic[element].abs().max(numeric.abs()),
            "element {element}: the tape gives {} where averaging slopes by {numeric}",
            analytic[element],
        );
    }
}

#[test]
fn a_max_pooled_gradient_routes_through_the_number_the_window_took() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.parameter(
        Shape::of([2, 3, 6, 6]),
        Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    let window = Window::new([2, 2], [2, 2], [0, 0]);
    let pooled = graph.pool2d(input, window, Pool::Max);
    let scale = graph.input(Shape::of([2, 3, 3, 3]), Element::Single);
    let loss = graph.sum(graph.mul(pooled, scale));
    let gradients = graph.backward(loss);
    let slope = gradients.of(input);
    graph.retain(slope);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = samples(216, 67);
    let scale_values = (0..54)
        .map(|index| (index as f32 * 0.071).cos() * 0.5)
        .collect::<Vec<_>>();
    runtime.write(&program, input, &values);
    runtime.write(&program, scale, &scale_values);
    runtime.run(&program);
    let expected = max_pool2d_gradient(&values, &scale_values, [2, 3, 6, 6], window);
    assert_close(&runtime.read(&program, slope), &expected, 1e-5);
}
