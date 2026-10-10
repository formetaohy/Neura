use neura_abi::Element;
use neura_graph::{Graph, Shape, Value};
use neura_runtime::Runtime;

#[path = "support/backend.rs"]
mod backend;
#[path = "support/input.rs"]
mod input;
#[path = "support/mod.rs"]
mod support;

use backend::open_with;
use input::random;
use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn refusal_message(action: impl FnOnce()) -> String {
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action))
        .expect_err("the refusal a readback carries is reported");
    refused
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "a refusal without a message".to_owned())
}

fn top_k_reference(values: &[f32], rows: u32, columns: u32, keep: u32) -> (Vec<f32>, Vec<f32>) {
    let mut kept_values = Vec::new();
    let mut kept_indices = Vec::new();
    for row in 0..rows as usize {
        let source = &values[row * columns as usize..(row + 1) * columns as usize];
        let mut order = (0..columns as usize).collect::<Vec<_>>();
        order.sort_by(
            |left, right| match source[*right].partial_cmp(&source[*left]) {
                Some(order) => order.then(left.cmp(right)),
                None => std::cmp::Ordering::Equal,
            },
        );
        for index in &order[..keep as usize] {
            kept_values.push(source[*index]);
            kept_indices.push(*index as f32);
        }
    }
    (kept_values, kept_indices)
}

struct Selecting {
    graph: Graph<'static>,
    value: Value<'static>,
    kept_values: Value<'static>,
    kept_indices: Value<'static>,
}

impl Selecting {
    fn of(rows: u32, columns: u32, keep: u32, element: Element) -> Self {
        let graph = Graph::new();
        let value = graph.input(Shape::matrix(rows, columns), element);
        let (kept_values, kept_indices) = graph.top_k(value, keep);
        graph.retain(kept_values);
        graph.retain(kept_indices);
        Self {
            graph,
            value,
            kept_values,
            kept_indices,
        }
    }

    fn run(
        &self,
        runtime: &Runtime,
        data: &[f32],
        keep: u32,
        rows: u32,
        columns: u32,
    ) -> (Vec<f32>, Vec<f32>) {
        let weights = runtime.weights(&self.graph);
        let program = runtime.compile(&self.graph, &weights);
        runtime.write(&program, self.value, data);
        runtime.run(&program);
        let (expected_values, expected_indices) = top_k_reference(data, rows, columns, keep);
        let mut produced = runtime.read_many(&program, &[self.kept_values, self.kept_indices]);
        assert_close(&produced[0], &expected_values, 1e-6);
        assert_eq!(produced[1], expected_indices);
        (produced.remove(0), produced.remove(0))
    }
}

#[test]
fn a_top_k_hands_back_the_largest_numbers_of_a_row_and_where_they_stood() {
    let runtime = open();
    let (rows, columns, keep) = (40u32, 64u32, 3u32);
    let selecting = Selecting::of(rows, columns, keep, Element::Single);
    let data = random(rows * columns, 11);
    let (values, indices) = selecting.run(&runtime, &data, keep, rows, columns);
    for row in 0..rows as usize {
        let kept = &indices[row * keep as usize..(row + 1) * keep as usize];
        let mut sorted = kept.to_vec();
        sorted.sort_by(|left, right| left.partial_cmp(right).expect("a class"));
        sorted.dedup();
        assert_eq!(sorted.len(), keep as usize, "row {row} kept a class twice");
        for pair in values[row * keep as usize..(row + 1) * keep as usize].windows(2) {
            assert!(
                pair[0] >= pair[1],
                "row {row} walks its candidates out of order: {pair:?}",
            );
        }
    }
}

#[test]
fn a_top_k_keeps_the_smaller_class_of_a_tie() {
    let runtime = open();
    let (rows, columns, keep) = (4u32, 9u32, 4u32);
    let selecting = Selecting::of(rows, columns, keep, Element::Single);
    let mut data = vec![0.0f32; (rows * columns) as usize];
    for row in 0..rows as usize {
        for column in 0..columns as usize {
            data[row * columns as usize + column] = ((column + row) % 3) as f32;
        }
    }
    data[3] = 2.0;
    data[7] = 2.0;
    data[8] = 2.0;
    let (values, indices) = selecting.run(&runtime, &data, keep, rows, columns);
    assert_eq!(
        &indices[..keep as usize],
        [2.0, 3.0, 5.0, 7.0],
        "a tie keeps the smaller class first",
    );
    assert_close(&values[..keep as usize], &[2.0, 2.0, 2.0, 2.0], 1e-6);
}

#[test]
fn a_top_k_of_one_is_the_largest_number_of_every_row() {
    let runtime = open();
    let (rows, columns) = (37u32, 32u32);
    let graph = Graph::new();
    let value = graph.input(Shape::matrix(rows, columns), Element::Single);
    let (kept_values, kept_indices) = graph.top_k(value, 1);
    let largest = graph.argmax(value);
    graph.retain(kept_values);
    graph.retain(kept_indices);
    graph.retain(largest);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data = random(rows * columns, 23);
    runtime.write(&program, value, &data);
    runtime.run(&program);
    let produced = runtime.read_many(&program, &[kept_values, kept_indices, largest]);
    let (expected_values, expected_indices) = top_k_reference(&data, rows, columns, 1);
    assert_close(&produced[0], &expected_values, 1e-6);
    assert_eq!(produced[1], expected_indices);
    assert_eq!(
        produced[1], produced[2],
        "a top k of one keeps the class an argmax names",
    );
}

#[test]
fn a_top_k_of_every_column_sorts_every_row() {
    let runtime = open();
    let (rows, columns) = (6u32, 60u32);
    let selecting = Selecting::of(rows, columns, columns, Element::Single);
    let data = random(rows * columns, 31);
    let (values, _) = selecting.run(&runtime, &data, columns, rows, columns);
    for row in 0..rows as usize {
        let mut expected = data[row * columns as usize..(row + 1) * columns as usize].to_vec();
        expected.sort_by(|left, right| right.partial_cmp(left).expect("a number"));
        assert_close(
            &values[row * columns as usize..(row + 1) * columns as usize],
            &expected,
            1e-6,
        );
    }
}

#[test]
fn a_row_wider_than_the_workgroup_keeps_its_largest_numbers() {
    let runtime = open();
    let (rows, columns, keep) = (5u32, 1000u32, 8u32);
    let selecting = Selecting::of(rows, columns, keep, Element::Single);
    let mut data = random(rows * columns, 43);
    data[3 * columns as usize + 501] = 100.0;
    data[3 * columns as usize + 999] = 100.0;
    let (values, indices) = selecting.run(&runtime, &data, keep, rows, columns);
    let kept = &indices[3 * keep as usize..4 * keep as usize];
    assert_eq!(
        &kept[..2],
        [501.0, 999.0],
        "a wide row keeps the two maxima it holds",
    );
    assert!(
        values[3 * keep as usize..4 * keep as usize]
            .windows(2)
            .all(|pair| pair[0] >= pair[1]),
        "a wide row walks its candidates from the largest on",
    );
}

#[test]
fn a_narrow_row_keeps_the_numbers_it_holds() {
    let runtime = open();
    let (rows, columns, keep) = (12u32, 17u32, 5u32);
    let selecting = Selecting::of(rows, columns, keep, Element::Half);
    let data = random(rows * columns, 47)
        .into_iter()
        .map(|value| (value * 8.0).round() / 8.0)
        .collect::<Vec<_>>();
    let (values, _) = selecting.run(&runtime, &data, keep, rows, columns);
    assert!(
        values.iter().all(|value| (value * 8.0).fract() == 0.0),
        "half precision keeps the numbers it was handed",
    );
}

#[test]
fn every_profile_keeps_the_same_candidates() {
    let runtime = open();
    let (rows, columns, keep) = (64u32, 128u32, 4u32);
    let selecting = Selecting::of(rows, columns, keep, Element::Single);
    let weights = runtime.weights(&selecting.graph);
    let data = random(rows * columns, 53);
    let (expected_values, expected_indices) = top_k_reference(&data, rows, columns, keep);
    for profile in runtime.profiles() {
        let program = runtime.compile_with(&selecting.graph, &weights, profile);
        runtime.write(&program, selecting.value, &data);
        runtime.run(&program);
        let produced =
            runtime.read_many(&program, &[selecting.kept_values, selecting.kept_indices]);
        assert_close(&produced[0], &expected_values, 1e-6);
        assert_eq!(
            produced[1], expected_indices,
            "profile {profile:?} kept another class"
        );
    }
}

#[test]
fn a_top_k_keeps_the_columns_a_binding_holds() {
    let runtime = open();
    let graph = Graph::new();
    let tokens = graph.free(32);
    let value = graph.input(Shape::of([2, 32, 4]).freed(&[(2, tokens)]), Element::Single);
    let (kept_values, kept_indices) = graph.top_k(value, 2);
    graph.retain(kept_values);
    graph.retain(kept_indices);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let row = [0.25f32, -0.5, 0.75, 0.0];
    let data = (0..64).flat_map(|_| row).collect::<Vec<_>>();
    let mut kept = Vec::new();
    for bound in [32u32, 8] {
        runtime.bind(&program, &[bound]);
        runtime.write(&program, value, &data[..bound as usize * 8]);
        runtime.run(&program);
        let mut produced = runtime.read_many(&program, &[kept_values, kept_indices]);
        kept.push((produced.remove(0), produced.remove(0)));
    }
    let (kept_wide, indices_wide) = (&kept[0].0, &kept[0].1);
    let (kept_narrow, indices_narrow) = (&kept[1].0, &kept[1].1);
    assert_eq!(kept_wide.len(), 2 * 32 * 2);
    assert_eq!(kept_narrow.len(), 2 * 8 * 2);
    for plane in 0..2usize {
        for token in 0..8usize {
            let narrow = plane * 8 * 2 + token * 2;
            let wide = plane * 32 * 2 + token * 2;
            assert_close(
                &kept_narrow[narrow..narrow + 2],
                &kept_wide[wide..wide + 2],
                1e-6,
            );
            assert_eq!(
                &indices_narrow[narrow..narrow + 2],
                &indices_wide[wide..wide + 2],
                "plane {plane} kept other classes when the binding named eight tokens",
            );
        }
    }
}

#[test]
fn a_top_k_the_graph_cannot_shape_is_refused() {
    let graph = Graph::new();
    let value = graph.input(Shape::matrix(4, 8), Element::Single);
    let empty = refusal_message(|| {
        let _ = graph.top_k(value, 0);
    });
    assert!(
        empty.contains("a top k keeps between one and"),
        "a top k of no candidates reads a row and keeps none: {empty}",
    );
    let beyond = refusal_message(|| {
        let _ = graph.top_k(value, 65);
    });
    assert!(
        beyond.contains("a top k keeps between one and"),
        "a top k beyond the candidates a choice kernel holds is refused: {beyond}",
    );
    let wider = refusal_message(|| {
        let _ = graph.top_k(value, 9);
    });
    assert!(
        wider.contains("reads a row of 8 numbers"),
        "a top k of more candidates than a row holds is refused: {wider}",
    );
    let view = graph.permute(value, [0, 1, 3, 2]);
    assert!(
        refuses(|| {
            let _ = graph.top_k(view, 2);
        }),
        "a top k of a view walks rows no storage lays out",
    );
    let free = graph.free(8);
    let dynamic = graph.input(Shape::of([1, 1, 4, 8]).freed(&[(3, free)]), Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.top_k(dynamic, 2);
        }),
        "a top k of a row of no fixed width is refused",
    );
}

#[test]
fn a_top_k_gradient_is_refused() {
    let graph = Graph::new();
    let value = graph.parameter(
        Shape::matrix(4, 8),
        neura_graph::Init::Zero,
        Element::Single,
    );
    let (kept_values, _) = graph.top_k(value, 2);
    let loss = graph.sum(kept_values);
    assert!(
        refuses(|| {
            let _ = graph.backward(loss);
        }),
        "a selection carries no gradient",
    );
}

#[test]
fn every_platform_backend_keeps_the_largest_numbers_of_a_row() {
    for &backend in neura_gpu::PREFERENCE {
        let runtime = open_with(backend);
        let (rows, columns, keep) = (17u32, 300u32, 6u32);
        let selecting = Selecting::of(rows, columns, keep, Element::Single);
        let data = random(rows * columns, 59);
        selecting.run(&runtime, &data, keep, rows, columns);
    }
}
