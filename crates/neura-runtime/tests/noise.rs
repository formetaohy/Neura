use neura_abi::Element;
use neura_graph::{Graph, Shape, Value};
use neura_runtime::Backends;

#[path = "support/backend.rs"]
mod backend;
#[path = "support/mod.rs"]
mod support;

use backend::open_with;
use support::{assert_close, open};

const ROWS: u32 = 64;
const COLUMNS: u32 = 64;
const ELEMENTS: usize = (ROWS * COLUMNS) as usize;

fn seed_of<'g>(graph: &Graph<'g>) -> Value<'g> {
    graph.input(Shape::scalar(), Element::Single)
}

fn moments(values: &[f32]) -> (f32, f32) {
    let count = values.len() as f64;
    let mean = values.iter().map(|value| f64::from(*value)).sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| {
            let difference = f64::from(*value) - mean;
            difference * difference
        })
        .sum::<f64>()
        / count;
    (mean as f32, variance as f32)
}

#[test]
fn a_draw_hands_back_the_numbers_the_seed_asks_for() {
    let runtime = open();
    let graph = Graph::new();
    let seed = seed_of(&graph);
    let uniform = graph.uniform(Shape::matrix(ROWS, COLUMNS), seed);
    let normal = graph.normal(Shape::matrix(ROWS, COLUMNS), seed);
    graph.retain(uniform);
    graph.retain(normal);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);

    let mut drawn = Vec::new();
    for bits in [0u32, 1, 7, 0xffff_ffff] {
        runtime.write(&program, seed, &[f32::from_bits(bits)]);
        runtime.run(&program);
        drawn.push((
            runtime.read(&program, uniform),
            runtime.read(&program, normal),
        ));
    }
    for (index, (uniform, normal)) in drawn.iter().enumerate() {
        assert_eq!(uniform.len(), ELEMENTS);
        assert!(
            uniform.iter().all(|value| *value >= 0.0 && *value < 1.0),
            "a uniform draw of seed {index} left the unit it names",
        );
        assert!(
            normal.iter().all(|value| value.is_finite()),
            "a normal draw of seed {index} left the numbers",
        );
        assert!(
            distinct(uniform) > ELEMENTS * 15 / 16,
            "a uniform draw of seed {index} repeats its numbers",
        );
        assert!(
            distinct(normal) > ELEMENTS * 15 / 16,
            "a normal draw of seed {index} repeats its numbers",
        );
    }
    for pair in drawn.windows(2) {
        assert_ne!(
            pair[0].0, pair[1].0,
            "two seeds draw the same numbers of the same tensor",
        );
    }
    runtime.write(&program, seed, &[f32::from_bits(7)]);
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, uniform),
        drawn[2].0,
        "a draw of a seed a run already walked comes back the same way",
    );
}

#[test]
fn a_draw_of_one_seed_is_a_draw_of_the_other() {
    let runtime = open();
    let graph = Graph::new();
    let seed = seed_of(&graph);
    let first = graph.uniform(Shape::matrix(ROWS, COLUMNS), seed);
    let again = graph.uniform(Shape::matrix(ROWS, COLUMNS), seed);
    let other = graph.uniform(
        Shape::matrix(ROWS, COLUMNS),
        graph.fill(Shape::scalar(), f32::from_bits(3)),
    );
    graph.retain(first);
    graph.retain(again);
    graph.retain(other);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, seed, &[f32::from_bits(9)]);
    runtime.run(&program);
    let first = runtime.read(&program, first);
    let again = runtime.read(&program, again);
    let other = runtime.read(&program, other);
    assert_eq!(
        first, again,
        "two draws name the same seed and walk the same tensor",
    );
    let differing = first
        .iter()
        .zip(&other)
        .filter(|(left, right)| left != right)
        .count();
    assert!(
        differing > ELEMENTS * 999 / 1000,
        "two seeds drew the same numbers at {} of {ELEMENTS} places",
        ELEMENTS - differing,
    );
}

#[test]
fn a_uniform_draw_spreads_over_the_unit_it_names() {
    let runtime = open();
    let graph = Graph::new();
    let seed = seed_of(&graph);
    let uniform = graph.uniform(Shape::matrix(ROWS, COLUMNS), seed);
    graph.retain(uniform);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, seed, &[f32::from_bits(0x5eed)]);
    runtime.run(&program);
    let values = runtime.read(&program, uniform);
    let (mean, variance) = moments(&values);
    assert!(
        (mean - 0.5).abs() < 0.02,
        "a uniform draw averages {mean} where the unit averages 0.5",
    );
    assert!(
        (variance - 1.0 / 12.0).abs() < 0.01,
        "a uniform draw spreads {variance} where the unit spreads {}",
        1.0 / 12.0,
    );
    let low = values.iter().copied().fold(f32::INFINITY, f32::min);
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        low < 0.01 && high > 0.99,
        "a draw of {ELEMENTS} numbers reached {low} to {high}"
    );
}

#[test]
fn a_normal_draw_carries_no_moment_of_its_own() {
    let runtime = open();
    let graph = Graph::new();
    let seed = seed_of(&graph);
    let normal = graph.normal(Shape::matrix(ROWS, COLUMNS), seed);
    graph.retain(normal);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, seed, &[f32::from_bits(0x5eed)]);
    runtime.run(&program);
    let values = runtime.read(&program, normal);
    let (mean, variance) = moments(&values);
    assert!(
        mean.abs() < 0.08,
        "a normal draw averages {mean} where the standard normal averages 0",
    );
    assert!(
        (variance - 1.0).abs() < 0.12,
        "a normal draw spreads {variance} where the standard normal spreads 1",
    );
    let high = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    assert!(
        high > 2.0,
        "a draw of {ELEMENTS} normal numbers reached only {high}"
    );
}

#[test]
fn a_dropout_keeps_the_share_of_numbers_it_names() {
    let runtime = open();
    let graph = Graph::new();
    let seed = seed_of(&graph);
    let data = graph.input(Shape::matrix(ROWS, COLUMNS), Element::Single);
    let dropped = graph.dropout(data, 0.75, seed);
    graph.retain(dropped);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &vec![1.0; ELEMENTS]);
    let scale = 1.0f32 / 0.75;
    for bits in [0u32, 3, 11] {
        runtime.write(&program, seed, &[f32::from_bits(bits)]);
        runtime.run(&program);
        let values = runtime.read(&program, dropped);
        let kept = values.iter().filter(|value| **value != 0.0).count();
        let due = ELEMENTS as f32 * 0.75;
        assert!(
            (kept as f32 - due).abs() <= ELEMENTS as f32 * 0.04,
            "seed {bits} kept {kept} of {ELEMENTS} numbers where {due} were due",
        );
        for (index, value) in values.iter().enumerate() {
            assert!(
                *value == 0.0 || *value == scale,
                "element {index} came back as {value} where a dropout keeps {scale} or nothing",
            );
        }
    }
}

#[test]
fn a_dropout_gradient_keeps_the_numbers_its_forward_kept() {
    let runtime = open();
    let graph = Graph::new();
    let seed = seed_of(&graph);
    let data = graph.gradient_input(Shape::matrix(ROWS, COLUMNS), Element::Single);
    let weights_value = graph.input(Shape::matrix(ROWS, COLUMNS), Element::Single);
    let dropped = graph.dropout(data, 0.5, seed);
    let loss = graph.sum(graph.mul(dropped, weights_value));
    let gradients = graph.backward(loss);
    graph.retain(dropped);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data_values = (0..ELEMENTS)
        .map(|index| 0.5 + index as f32 * 0.25)
        .collect::<Vec<_>>();
    let weight_values = (0..ELEMENTS)
        .map(|index| 1.0 + index as f32 * 0.5)
        .collect::<Vec<_>>();
    runtime.write(&program, data, &data_values);
    runtime.write(&program, weights_value, &weight_values);
    runtime.write(&program, seed, &[f32::from_bits(5)]);
    runtime.run(&program);
    let forward = runtime.read(&program, dropped);
    let gradient = runtime.read(&program, gradients.of(data));
    let mut dropped_count = 0usize;
    for index in 0..ELEMENTS {
        let expected = if forward[index] == 0.0 {
            dropped_count += 1;
            0.0
        } else {
            weight_values[index] * 2.0
        };
        assert_eq!(
            gradient[index], expected,
            "element {index} descended by {} where the mask of the forward keeps {expected}",
            gradient[index],
        );
    }
    assert!(
        dropped_count > ELEMENTS / 8 && dropped_count < ELEMENTS * 7 / 8,
        "a dropout of half the numbers dropped {dropped_count} of {ELEMENTS}",
    );
}

#[test]
fn a_draw_walks_the_length_a_binding_holds() {
    let runtime = open();
    let graph = Graph::new();
    let batch = graph.free(16);
    let seed = seed_of(&graph);
    let shape = Shape::of([16, 1, 1, 8]).freed(&[(0, batch)]);
    let draw = graph.uniform(shape, seed);
    graph.retain(draw);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    for rows in [16u32, 5, 1, 0] {
        runtime.bind(&program, &[rows]);
        runtime.write(&program, seed, &[f32::from_bits(2)]);
        runtime.run(&program);
        let values = runtime.read(&program, draw);
        assert_eq!(values.len(), rows as usize * 8);
        assert!(
            values.iter().all(|value| *value >= 0.0 && *value < 1.0),
            "a draw of {rows} rows left the unit it names",
        );
    }
}

#[test]
fn a_drawn_element_keeps_the_number_its_coordinate_names() {
    let runtime = open();
    let graph = Graph::new();
    let tokens = graph.free(8);
    let seed = seed_of(&graph);
    let draw = graph.uniform(Shape::of([2, 8, 4]).freed(&[(2, tokens)]), seed);
    graph.retain(draw);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let mut drawn = Vec::new();
    for tokens in [8u32, 4] {
        runtime.bind(&program, &[tokens]);
        runtime.write(&program, seed, &[f32::from_bits(0x5eed)]);
        runtime.run(&program);
        drawn.push(runtime.read(&program, draw));
    }
    assert_eq!(drawn[0].len(), 2 * 8 * 4);
    assert_eq!(drawn[1].len(), 2 * 4 * 4);
    for plane in 0..2usize {
        assert_eq!(
            &drawn[0][plane * 32..plane * 32 + 16],
            &drawn[1][plane * 16..plane * 16 + 16],
            "the first four tokens of plane {plane} drew other numbers when the binding named four instead of eight",
        );
    }
}

#[test]
fn a_reparameterized_draw_descends_through_its_mean_and_its_spread() {
    let runtime = open();
    let graph = Graph::new();
    let seed = seed_of(&graph);
    let mean = graph.gradient_input(Shape::matrix(ROWS, COLUMNS), Element::Single);
    let spread = graph.gradient_input(Shape::matrix(ROWS, COLUMNS), Element::Single);
    let draw = graph.normal(Shape::matrix(ROWS, COLUMNS), seed);
    let action = graph.add(mean, graph.mul(spread, draw));
    let gradients = graph.backward(graph.sum(action));
    graph.retain(draw);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, mean, &vec![0.0; ELEMENTS]);
    runtime.write(&program, spread, &vec![1.0; ELEMENTS]);
    runtime.write(&program, seed, &[f32::from_bits(0x51)]);
    runtime.run(&program);
    let draw = runtime.read(&program, draw);
    assert_close(
        &runtime.read(&program, gradients.of(mean)),
        &vec![1.0; ELEMENTS],
        0.0,
    );
    assert_close(&runtime.read(&program, gradients.of(spread)), &draw, 0.0);
}

#[test]
fn a_uniform_draw_names_the_same_number_on_every_backend() {
    let mut drawn = Vec::new();
    for backends in Backends::PLATFORM {
        let runtime = open_with(backends);
        let graph = Graph::new();
        let seed = seed_of(&graph);
        let uniform = graph.uniform(Shape::matrix(ROWS, COLUMNS), seed);
        graph.retain(uniform);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, seed, &[f32::from_bits(0x5eed)]);
        runtime.run(&program);
        drawn.push((backends, runtime.read(&program, uniform)));
    }
    for pair in drawn.windows(2) {
        assert_eq!(
            pair[0].1, pair[1].1,
            "one seed wrote different numbers on {:?} and {:?}",
            pair[0].0, pair[1].0,
        );
    }
}

fn distinct(values: &[f32]) -> usize {
    values
        .iter()
        .map(|value| value.to_bits())
        .collect::<std::collections::HashSet<_>>()
        .len()
}
