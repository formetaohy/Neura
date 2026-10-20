use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

const CLASSES: u32 = 32;

fn runtime() -> Runtime {
    Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("a device that runs the tests")
}

fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index} came back as {actual} where {expected} was expected",
        );
    }
}

fn bits(seed: u32) -> f32 {
    f32::from_bits(seed)
}

#[test]
fn a_row_seed_draws_the_numbers_its_own_row_holds() {
    let runtime = runtime();
    let rows = 2u32;
    let logits = random(rows * CLASSES, 71)
        .chunks(CLASSES as usize)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let seeds = [0x2f0f_1a3bu32, 0x91e1_0d57];
    let mut sampled = Vec::new();
    for row in 0..rows {
        let graph = Graph::new();
        let source = graph.input(Shape::of([1, 1, 1, CLASSES]), Element::Single);
        let seed = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
        let keep = graph.fill(Shape::scalar(), 6.0);
        let cumulative = graph.fill(Shape::scalar(), 0.85);
        let chosen = graph.sample(source, seed, keep, cumulative);
        graph.retain(chosen);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, source, &logits[row as usize]);
        runtime.write(&program, seed, &[bits(seeds[row as usize])]);
        runtime.run(&program);
        sampled.push(runtime.read(&program, chosen)[0]);
    }

    let graph = Graph::new();
    let source = graph.input(Shape::of([1, 1, rows, CLASSES]), Element::Single);
    let seed = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
    let keep = graph.fill(Shape::scalar(), 6.0);
    let cumulative = graph.fill(Shape::scalar(), 0.85);
    let chosen = graph.sample(source, seed, keep, cumulative);
    graph.retain(chosen);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = logits.iter().flatten().copied().collect::<Vec<f32>>();
    runtime.write(&program, source, &held);
    runtime.write(
        &program,
        seed,
        &seeds.iter().copied().map(bits).collect::<Vec<f32>>(),
    );
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, chosen),
        sampled,
        "a row keeps the draw its own seed names whatever row it answers in",
    );
}

#[test]
fn a_nucleus_that_keeps_one_candidate_draws_the_argmax() {
    let runtime = runtime();
    let rows = 4u32;
    let logits = random(rows * CLASSES, 73);
    let graph = Graph::new();
    let source = graph.input(Shape::of([1, 1, rows, CLASSES]), Element::Single);
    let seed = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
    let keep = graph.fill(Shape::scalar(), 1.0);
    let cumulative = graph.fill(Shape::scalar(), 1.0);
    let sampled = graph.sample(source, seed, keep, cumulative);
    let largest = graph.argmax(source);
    graph.retain(sampled);
    graph.retain(largest);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, source, &logits);
    let mut seeds = (0..rows).map(|row| 0x3c6e_f372 ^ row).collect::<Vec<u32>>();
    runtime.write(
        &program,
        seed,
        &seeds.iter().copied().map(bits).collect::<Vec<f32>>(),
    );
    runtime.run(&program);
    let expected = runtime.read(&program, largest);
    assert_close(&runtime.read(&program, sampled), &expected, 0.0);
    for value in &mut seeds {
        *value = value.wrapping_add(0x9e37_79b9);
    }
    runtime.write(
        &program,
        seed,
        &seeds.iter().copied().map(bits).collect::<Vec<f32>>(),
    );
    runtime.run(&program);
    assert_close(&runtime.read(&program, sampled), &expected, 0.0);
}

#[test]
fn a_row_seed_draws_the_class_its_own_row_holds() {
    let runtime = runtime();
    let rows = 2u32;
    let logits = random(rows * CLASSES, 79)
        .chunks(CLASSES as usize)
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    let seeds = [0x51e7_3c2du32, 0x87ab_19f5];
    let mut drawn = Vec::new();
    for row in 0..rows {
        let graph = Graph::new();
        let source = graph.input(Shape::of([1, 1, 1, CLASSES]), Element::Single);
        let seed = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
        let chosen = graph.categorical(source, seed);
        graph.retain(chosen);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, source, &logits[row as usize]);
        runtime.write(&program, seed, &[bits(seeds[row as usize])]);
        runtime.run(&program);
        drawn.push(runtime.read(&program, chosen)[0]);
    }

    let graph = Graph::new();
    let source = graph.input(Shape::of([1, 1, rows, CLASSES]), Element::Single);
    let seed = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
    let chosen = graph.categorical(source, seed);
    graph.retain(chosen);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = logits.iter().flatten().copied().collect::<Vec<f32>>();
    runtime.write(&program, source, &held);
    runtime.write(
        &program,
        seed,
        &seeds.iter().copied().map(bits).collect::<Vec<f32>>(),
    );
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, chosen),
        drawn,
        "a row draws the class its own seed names whatever row it answers in",
    );
}
