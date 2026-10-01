use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_precision::{pack, unpack};

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

fn rotate(
    rows: u32,
    width: u32,
    origin: u32,
    base: f32,
    values: &[f32],
    inverse: bool,
) -> Vec<f32> {
    let half = (width / 2) as usize;
    let width = width as usize;
    let mut out = values.to_vec();
    for row in 0..rows as usize {
        for column in 0..width {
            let high = column >= half;
            let channel = if high { column - half } else { column };
            let angle =
                (origin + row as u32) as f32 * base.powf(-2.0 * channel as f32 / width as f32);
            let (sine, cosine) = angle.sin_cos();
            let partner = if high { column - half } else { column + half };
            let value = values[row * width + column];
            let other = values[row * width + partner];
            let sign = if high != inverse { 1.0 } else { -1.0 };
            out[row * width + column] = value * cosine + sign * other * sine;
        }
    }
    out
}

#[test]
fn a_rope_turns_every_row_by_the_position_its_cursor_names() {
    let runtime = open();
    let graph = Graph::new();
    let shape = Shape::of([2, 1, 4, 6]);
    let data = graph.input(shape, Element::Single);
    let cursor = graph.input(Shape::of([2, 1, 1, 1]), Element::Single);
    let turned = graph.rope(data, Some(cursor), 10000.0);
    graph.retain(turned);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(48, 17);
    runtime.write(&program, data, &values);
    runtime.write(&program, cursor, &[1.0, 5.0]);
    runtime.run(&program);
    let produced = runtime.read(&program, turned);
    for (plane, origin) in [1u32, 5].into_iter().enumerate() {
        let span = plane * 24;
        assert_close(
            &produced[span..span + 24],
            &rotate(4, 6, origin, 10000.0, &values[span..span + 24], false),
            1e-3,
        );
    }
}

#[test]
fn a_rope_gradient_undoes_the_turn_it_was_shown() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::of([1, 1, 4, 6]), Init::Zero, Element::Single);
    let loss = graph.sum(graph.rope(weight, None, 10000.0));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(weight);
    graph.retain(gradient);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, gradient),
        &rotate(4, 6, 0, 10000.0, &[1.0; 24], true),
        1e-3,
    );
}

#[test]
fn a_rope_gradient_matches_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::of([1, 1, 3, 4]), Init::Zero, Element::Single);
    let target = graph.input(Shape::of([1, 1, 3, 4]), Element::Single);
    let loss = graph.sum(graph.mul(graph.rope(weight, None, 100.0), target));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(weight);
    graph.retain(loss);
    graph.retain(gradient);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, target, &random(12, 23));
    runtime.run(&program);
    let values = runtime.read(&program, weight);
    let analytic = runtime.read(&program, gradient);
    for element in 0..values.len() {
        let step = 1e-2;
        let mut probe = values.clone();
        probe[element] += step;
        runtime.write(&program, weight, &probe);
        runtime.run(&program);
        let high = runtime.read(&program, loss)[0];
        probe[element] -= 2.0 * step;
        runtime.write(&program, weight, &probe);
        runtime.run(&program);
        let low = runtime.read(&program, loss)[0];
        let numeric = (high - low) / (2.0 * step);
        let slack = 1e-3 + 1e-2 * analytic[element].abs().max(numeric.abs());
        assert!(
            (numeric - analytic[element]).abs() <= slack,
            "element {element} of a rotated tensor: the tape gives {} where the slope is {numeric}",
            analytic[element],
        );
    }
}

#[test]
fn a_cursor_the_device_cannot_walk_stops_the_rope() {
    let runtime = open();
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, 1, 4]), Element::Single);
    let cursor = graph.input(Shape::scalar(), Element::Single);
    let turned = graph.rope(data, Some(cursor), 10000.0);
    graph.retain(turned);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[1.0, 2.0, 3.0, 4.0]);
    for cursor_value in [-1.0, 0.5, 1.0e30] {
        runtime.write(&program, cursor, &[cursor_value]);
        runtime.run(&program);
        assert!(refuses(|| {
            let _ = runtime.read(&program, turned);
        }));
    }
}

#[test]
fn a_rope_carries_the_epilogue_that_folds_into_it() {
    let runtime = open();
    let graph = Graph::new();
    let shape = Shape::of([1, 1, 2, 4]);
    let data = graph.input(shape, Element::Single);
    let scaled = graph.mul(graph.rope(data, None, 100.0), graph.fill(shape, 2.0));
    graph.retain(scaled);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(
        program.task_count(),
        2,
        "a fill and the rope whose epilogue multiplies it ride one tape",
    );
    let values = random(8, 19);
    runtime.write(&program, data, &values);
    runtime.run(&program);
    let expected = rotate(2, 4, 0, 100.0, &values, false)
        .into_iter()
        .map(|value| value * 2.0)
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, scaled), &expected, 1e-3);
}

#[test]
fn a_narrow_rope_packs_the_turn_it_computed() {
    let runtime = open();
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, 2, 4]), Element::Single);
    let halves = graph.cast(data, Element::Half);
    graph.retain(halves);
    let turned = graph.rope(halves, None, 100.0);
    assert_eq!(graph.element(turned), Element::Half);
    graph.retain(turned);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(8, 13);
    runtime.write(&program, data, &values);
    runtime.run(&program);
    let rounded = unpack(Element::Half, 8, &pack(Element::Half, 1.0, &values));
    assert_close(
        &runtime.read(&program, turned),
        &unpack(
            Element::Half,
            8,
            &pack(Element::Half, 1.0, &rotate(2, 4, 0, 100.0, &rounded, false)),
        ),
        1e-4,
    );
}
