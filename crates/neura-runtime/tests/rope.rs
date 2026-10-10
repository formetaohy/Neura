use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
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

fn rotate_planes(lengths: &[f32], width: u32, base: f32, inverse: bool, image: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut start = 0u32;
    for length in lengths {
        let length = *length as u32;
        let at = (start * width) as usize;
        out.extend(rotate(
            length,
            width,
            0,
            base,
            &image[at..at + (length * width) as usize],
            inverse,
        ));
        start += length;
    }
    out
}

struct Packed<'a> {
    runtime: neura_runtime::Runtime,
    graph: Graph<'a>,
    lengths: Value<'a>,
    extent: neura_graph::Free,
    packed: Value<'a>,
    turned: Value<'a>,
}

const PACKED_BOUND: u32 = 8;
const PACKED_WIDTH: u32 = 4;
const PACKED_BASE: f32 = 10000.0;

impl<'a> Packed<'a> {
    fn build() -> Self {
        let graph = Graph::new();
        let lengths = graph.input(Shape::vector(4), Element::Single);
        let ragged = graph.ragged(PACKED_BOUND, lengths);
        let packed = graph.gradient_input(
            Shape::of([1, 1, PACKED_BOUND, PACKED_WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let turned = graph.rope(packed, None, Some(graph.knob(PACKED_BASE)));
        graph.retain(turned);
        Self {
            runtime: open(),
            graph,
            lengths,
            extent: ragged.extent,
            packed,
            turned,
        }
    }

    fn step(&self, program: &neura_runtime::Program, lengths: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let image = random(PACKED_BOUND * PACKED_WIDTH, 71);
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.packed, &image);
        self.runtime.run(program);
        let produced = self.runtime.read(program, self.turned);
        let expected = rotate_planes(lengths, PACKED_WIDTH, PACKED_BASE, false, &image);
        assert_close(&produced, &expected, 1e-4);
        (produced, image)
    }
}

#[test]
fn a_packed_rope_places_every_row_in_the_plane_its_offsets_close() {
    let packed = Packed::build();
    let weights = packed.runtime.weights(&packed.graph);
    let program = packed.runtime.compile(&packed.graph, &weights);
    for lengths in [
        [3.0, 3.0, 2.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [4.0, 2.0, 2.0, 0.0],
    ] {
        packed.step(&program, &lengths);
    }
}

#[test]
fn a_packed_rope_turns_the_rows_of_a_plane_wider_than_a_workgroup() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(3), Element::Single);
    let bound = 512;
    let ragged = graph.ragged(bound, lengths);
    let packed = graph.gradient_input(
        Shape::of([1, 1, bound, PACKED_WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let turned = graph.rope(packed, None, Some(graph.knob(PACKED_BASE)));
    graph.retain(turned);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let walked = [300.0f32, 70.0, 5.0];
    let image = random(bound * PACKED_WIDTH, 73);
    runtime.write(&program, lengths, &walked);
    runtime.write(&program, packed, &image);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, turned),
        &rotate_planes(&walked, PACKED_WIDTH, PACKED_BASE, false, &image),
        1e-4,
    );
}

#[test]
fn a_packed_rope_gradient_turns_the_rows_back_to_the_seat_they_came_from() {
    let packed = Packed::build();
    let graph = &packed.graph;
    let weight = graph.input(
        Shape::of([1, 1, PACKED_BOUND, PACKED_WIDTH]).freed(&[(2, packed.extent)]),
        Element::Single,
    );
    let loss = graph.sum(graph.mul(packed.turned, weight));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(packed.packed);
    graph.retain(gradient);
    let weights = packed.runtime.weights(graph);
    let program = packed.runtime.compile(graph, &weights);
    let lengths = [2.0f32, 1.0, 3.0, 0.0];
    let image = random(PACKED_BOUND * PACKED_WIDTH, 79);
    let incoming = random(PACKED_BOUND * PACKED_WIDTH, 83);
    packed.runtime.write(&program, packed.lengths, &lengths);
    packed.runtime.write(&program, packed.packed, &image);
    packed.runtime.write(&program, weight, &incoming);
    packed.runtime.run(&program);
    assert_close(
        &packed.runtime.read(&program, gradient),
        &rotate_planes(&lengths, PACKED_WIDTH, PACKED_BASE, true, &incoming),
        1e-4,
    );
}

#[test]
fn a_packed_rope_of_one_plane_walks_the_row_its_cursor_names() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(1), Element::Single);
    let ragged = graph.ragged(PACKED_BOUND, lengths);
    let packed = graph.input(
        Shape::of([1, 1, PACKED_BOUND, PACKED_WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let cursor = graph.input(Shape::scalar(), Element::Single);
    let turned = graph.rope(packed, Some(cursor), Some(graph.knob(PACKED_BASE)));
    graph.retain(turned);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let walked = [3.0f32];
    let image = random(PACKED_BOUND * PACKED_WIDTH, 89);
    runtime.write(&program, lengths, &walked);
    runtime.write(&program, packed, &image);
    runtime.write(&program, cursor, &[5.0]);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, turned),
        &rotate(3, PACKED_WIDTH, 5, PACKED_BASE, &image[..12], false),
        1e-4,
    );
}

#[test]
fn a_narrow_packed_rope_packs_the_turn_of_every_plane() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let ragged = graph.ragged(PACKED_BOUND, lengths);
    let data = graph.input(
        Shape::of([1, 1, PACKED_BOUND, PACKED_WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let halves = graph.cast(data, Element::Half);
    let turned = graph.rope(halves, None, Some(graph.knob(PACKED_BASE)));
    assert_eq!(graph.element(turned), Element::Half);
    graph.retain(turned);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let walked = [2.0f32, 1.0, 3.0, 0.0];
    let values = random(PACKED_BOUND * PACKED_WIDTH, 97);
    runtime.write(&program, lengths, &walked);
    runtime.write(&program, data, &values);
    runtime.run(&program);
    let rounded = unpack(
        Element::Half,
        (PACKED_BOUND * PACKED_WIDTH) as usize,
        &pack(Element::Half, 1.0, &values),
    );
    let turned_expected = rotate_planes(&walked, PACKED_WIDTH, PACKED_BASE, false, &rounded);
    let expected = unpack(
        Element::Half,
        turned_expected.len(),
        &pack(Element::Half, 1.0, &turned_expected),
    );
    assert_close(&runtime.read(&program, turned), &expected, 1e-4);
}

#[test]
fn a_packed_rope_leaves_the_planes_a_binding_never_closes() {
    let graph: Graph<'static> = Graph::new();
    let free = graph.free(4);
    let lengths = graph.input(Shape::vector(4).freed(&[(3, free)]), Element::Single);
    let ragged = graph.ragged(PACKED_BOUND, lengths);
    let packed = graph.input(
        Shape::of([1, 1, PACKED_BOUND, PACKED_WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let turned = graph.rope(packed, None, Some(graph.knob(PACKED_BASE)));
    graph.retain(turned);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let image = random(PACKED_BOUND * PACKED_WIDTH, 101);
    runtime.bind(&program, &[4]);
    runtime.write(&program, lengths, &[2.0, 1.0, 3.0, 0.0]);
    runtime.write(&program, packed, &image);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, turned),
        &rotate_planes(
            &[2.0, 1.0, 3.0, 0.0],
            PACKED_WIDTH,
            PACKED_BASE,
            false,
            &image,
        ),
        1e-4,
    );
    runtime.bind(&program, &[2]);
    runtime.write(&program, lengths, &[2.0, 1.0]);
    runtime.write(&program, packed, &image);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, turned),
        &rotate_planes(&[2.0, 1.0], PACKED_WIDTH, PACKED_BASE, false, &image),
        1e-4,
    );
}

#[test]
fn a_packed_rope_carries_the_epilogue_that_folds_into_it() {
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let ragged = graph.ragged(PACKED_BOUND, lengths);
    let shape = Shape::of([1, 1, PACKED_BOUND, PACKED_WIDTH]).freed(&[(2, ragged.extent)]);
    let packed = graph.input(shape, Element::Single);
    let turned = graph.rope(packed, None, Some(graph.knob(PACKED_BASE)));
    let rows = graph.input(
        Shape::of([1, 1, PACKED_BOUND, 1]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let scaled = graph.mul(turned, rows);
    graph.retain(scaled);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let walked = [2.0f32, 1.0, 3.0, 0.0];
    let image = random(PACKED_BOUND * PACKED_WIDTH, 103);
    let bias = random(PACKED_BOUND, 107);
    runtime.write(&program, lengths, &walked);
    runtime.write(&program, packed, &image);
    runtime.write(&program, rows, &bias);
    runtime.run(&program);
    let expected = rotate_planes(&walked, PACKED_WIDTH, PACKED_BASE, false, &image)
        .into_iter()
        .enumerate()
        .map(|(at, value)| value * bias[at / PACKED_WIDTH as usize])
        .collect::<Vec<f32>>();
    assert_close(&runtime.read(&program, scaled), &expected, 1e-4);
}

#[test]
fn a_rope_turns_every_row_by_the_position_its_cursor_names() {
    let runtime = open();
    let graph = Graph::new();
    let shape = Shape::of([2, 1, 4, 6]);
    let data = graph.input(shape, Element::Single);
    let cursor = graph.input(Shape::of([2, 1, 1, 1]), Element::Single);
    let turned = graph.rope(data, Some(cursor), None);
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
    let loss = graph.sum(graph.rope(weight, None, None));
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
    let loss = graph.sum(graph.mul(graph.rope(weight, None, Some(graph.knob(100.0))), target));
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
            "element {element} of a rotated tensor: the plan gives {} where the slope is {numeric}",
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
    let turned = graph.rope(data, Some(cursor), None);
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
    let scaled = graph.mul(
        graph.rope(data, None, Some(graph.knob(100.0))),
        graph.fill(shape, 2.0),
    );
    graph.retain(scaled);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(
        program.task_count(),
        2,
        "a fill and the rope whose epilogue multiplies it ride one plan",
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
    let turned = graph.rope(halves, None, Some(graph.knob(100.0)));
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

const KNOB_ROWS: u32 = 3;
const KNOB_WIDTH: u32 = 4;

#[test]
fn a_rotary_base_the_host_writes_places_each_position_on_the_angle_it_names() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let knob = graph.named_knob("rope.base", 10_000.0);
    let data = graph.parameter(
        Shape::of([1, 1, KNOB_ROWS, KNOB_WIDTH]),
        Init::Zero,
        Element::Single,
    );
    let turned = graph.rope(data, None, Some(knob));
    graph.retain(turned);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(KNOB_ROWS * KNOB_WIDTH, 11);
    runtime.write(&program, data, &values);
    for written in [10_000.0f32, 100.0, 500.0] {
        runtime.write(&program, knob, &[written]);
        runtime.run(&program);
        assert_close(
            &runtime.read(&program, turned),
            &rotate(KNOB_ROWS, KNOB_WIDTH, 0, written, &values, false),
            1e-5,
        );
    }
    assert_eq!(
        runtime.built_plans(),
        1,
        "a program turns every base the host writes",
    );
}

#[test]
fn a_rotation_of_the_standard_base_needs_no_knob() {
    let runtime = open();
    let graph = Graph::new();
    let data = graph.parameter(
        Shape::of([1, 1, KNOB_ROWS, KNOB_WIDTH]),
        Init::Zero,
        Element::Single,
    );
    let turned = graph.rope(data, None, None);
    graph.retain(turned);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(KNOB_ROWS * KNOB_WIDTH, 11);
    runtime.write(&program, data, &values);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, turned),
        &rotate(KNOB_ROWS, KNOB_WIDTH, 0, 10_000.0, &values, false),
        1e-5,
    );
}

#[test]
fn the_device_refuses_a_rotary_base_it_cannot_turn_with() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let knob = graph.named_knob("rope.base", 10_000.0);
    let data = graph.parameter(
        Shape::of([1, 1, KNOB_ROWS, KNOB_WIDTH]),
        Init::Zero,
        Element::Single,
    );
    let turned = graph.rope(data, None, Some(knob));
    graph.retain(turned);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    for written in [1.0f32, 0.5, f32::NAN, f32::INFINITY] {
        runtime.write(&program, knob, &[written]);
        runtime.run(&program);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = runtime.read(&program, turned);
        }))
        .expect_err("a base every position shares is refused");
        let message = refused
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| "a refusal without a message".to_owned());
        assert!(
            message.contains("the device refused the rotary base of the rope task"),
            "{message}",
        );
    }
}
