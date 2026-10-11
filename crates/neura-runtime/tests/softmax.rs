use neura_abi::{Element, Kind};
use neura_graph::{Graph, Init, Shape};
use neura_plan::{DEFAULT_ENCODING_BYTES, Plan};
use neura_precision::{pack, unpack};
use neura_runtime::{MemoryRequest, Program, Runtime, RuntimeRequest};

#[path = "support/shared.rs"]
mod shared;

use shared::runtime;

fn open() -> Runtime {
    runtime(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes: 1 << 30,
            readback_bytes: 4 << 20,
            ..Default::default()
        },
        ..Default::default()
    })
}

fn data(count: usize, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..count)
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn peaked(rows: u32, columns: u32, sections: u32, peak: f32) -> Vec<f32> {
    let mut values = vec![-peak; (rows * columns) as usize];
    let span = columns / sections;
    for row in 0..rows {
        for section in 0..sections {
            let at = row * columns + section * span + (span / 2 + section) % span;
            values[at as usize] = peak - section as f32;
        }
    }
    values
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: lengths differ");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if actual == expected {
            continue;
        }
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: element {index} came back as {actual} where {expected} was expected",
        );
    }
}

fn softmax_reference(input: &[f32], columns: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (row, chunk) in input.chunks(columns as usize).enumerate() {
        let largest = chunk.iter().copied().fold(f32::MIN, f32::max);
        let total: f64 = chunk
            .iter()
            .map(|value| f64::from(value - largest).exp())
            .sum();
        for (column, value) in chunk.iter().enumerate() {
            out[row * columns as usize + column] =
                (f64::from(value - largest).exp() / total) as f32;
        }
    }
    out
}

fn log_softmax_reference(input: &[f32], columns: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (row, chunk) in input.chunks(columns as usize).enumerate() {
        let largest = chunk.iter().copied().fold(f32::MIN, f32::max);
        let total: f64 = chunk
            .iter()
            .map(|value| f64::from(value - largest).exp())
            .sum();
        let normalizer = total.ln();
        for (column, value) in chunk.iter().enumerate() {
            out[row * columns as usize + column] = (f64::from(value - largest) - normalizer) as f32;
        }
    }
    out
}

fn weighted_gradient_reference(input: &[f32], weight: &[f32], columns: u32) -> Vec<f32> {
    let probabilities = softmax_reference(input, columns);
    let mut out = vec![0.0f32; input.len()];
    for row in 0..input.len() / columns as usize {
        let base = row * columns as usize;
        let dot: f64 = (0..columns as usize)
            .map(|column| {
                f64::from(probabilities[base + column]) * f64::from(weight[base + column])
            })
            .sum();
        for column in 0..columns as usize {
            out[base + column] = (f64::from(probabilities[base + column])
                * (f64::from(weight[base + column]) - dot)) as f32;
        }
    }
    out
}

fn kinds(runtime: &Runtime, graph: &Graph, program: &Program) -> Vec<Kind> {
    Plan::of(
        graph,
        runtime.alignment(),
        program.profile(),
        DEFAULT_ENCODING_BYTES,
    )
    .kinds()
    .to_vec()
}

#[test]
fn a_starved_softmax_parts_its_columns_between_the_tasks_of_its_rows() {
    let rows = 1u32;
    let columns = 4096u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let probabilities = graph.softmax(logits);
    graph.retain(probabilities);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let plan = Plan::of(
        &graph,
        runtime.alignment(),
        program.profile(),
        DEFAULT_ENCODING_BYTES,
    );
    assert!(
        plan.kinds().contains(&Kind::SoftmaxFold),
        "one row of {columns} numbers starves the workgroups of the device",
    );
    let values = peaked(rows, columns, 4, 8.0);
    runtime.write(&program, logits, &values);
    runtime.run(&program);
    close(
        &runtime.read(&program, probabilities),
        &softmax_reference(&values, columns),
        1e-6,
        "a sectioned softmax",
    );
}

#[test]
fn a_starved_log_softmax_folds_every_column_section_back() {
    let rows = 2u32;
    let columns = 8192u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let probabilities = graph.log_softmax(logits);
    graph.retain(probabilities);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let plan = Plan::of(
        &graph,
        runtime.alignment(),
        program.profile(),
        DEFAULT_ENCODING_BYTES,
    );
    assert!(plan.kinds().contains(&Kind::SoftmaxFold));
    let values = peaked(rows, columns, 8, 12.0);
    runtime.write(&program, logits, &values);
    runtime.run(&program);
    let produced = runtime.read(&program, probabilities);
    close(
        &produced,
        &log_softmax_reference(&values, columns),
        1e-5,
        "a sectioned log softmax",
    );
    let mass: f32 = produced.iter().map(|value| value.exp()).sum();
    assert!(
        (mass - rows as f32).abs() < 1e-3,
        "a shared log softmax weighs {mass} where every one of {rows} rows weighs one",
    );
}

#[test]
fn a_starved_softmax_gradient_folds_the_columns_of_every_row() {
    let rows = 3u32;
    let columns = 6144u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let weight = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let weighted = graph.mul(graph.softmax(logits), weight);
    let loss = graph.sum(weighted);
    let gradients = graph.backward(loss);
    let gradient = gradients.of(logits);
    graph.retain(loss);
    graph.retain(gradient);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let plan = Plan::of(
        &graph,
        runtime.alignment(),
        program.profile(),
        DEFAULT_ENCODING_BYTES,
    );
    assert!(plan.kinds().contains(&Kind::SoftmaxFold));
    assert!(
        plan.kinds().contains(&Kind::SoftmaxGradFold),
        "a gradient of a sectioned softmax folds its rows back",
    );
    let logits_data = data((rows * columns) as usize, 17);
    let weight_data = data((rows * columns) as usize, 29);
    runtime.write(&program, logits, &logits_data);
    runtime.write(&program, weight, &weight_data);
    runtime.run(&program);
    close(
        &runtime.read(&program, gradient),
        &weighted_gradient_reference(&logits_data, &weight_data, columns),
        1e-5,
        "a sectioned softmax gradient",
    );
}

#[test]
fn a_softmax_the_device_already_fills_walks_whole_rows() {
    let rows = 128u32;
    let columns = 4096u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let probabilities = graph.softmax(logits);
    graph.retain(probabilities);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let kinds = kinds(&runtime, &graph, &program);
    assert!(
        !kinds.contains(&Kind::SoftmaxFold),
        "{rows} rows fill every workgroup of the device without parting their columns",
    );
    let values = data((rows * columns) as usize, 43);
    runtime.write(&program, logits, &values);
    runtime.run(&program);
    close(
        &runtime.read(&program, probabilities),
        &softmax_reference(&values, columns),
        1e-5,
        "a whole row softmax",
    );
}

#[test]
fn a_softmax_narrower_than_the_section_floor_walks_whole_rows() {
    let rows = 1u32;
    let columns = 1024u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let probabilities = graph.softmax(logits);
    graph.retain(probabilities);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let kinds = kinds(&runtime, &graph, &program);
    assert!(
        !kinds.contains(&Kind::SoftmaxFold),
        "a row of {columns} numbers is one workgroup's work",
    );
    let values = data((rows * columns) as usize, 47);
    runtime.write(&program, logits, &values);
    runtime.run(&program);
    close(
        &runtime.read(&program, probabilities),
        &softmax_reference(&values, columns),
        1e-6,
        "a narrow softmax",
    );
}

#[test]
fn a_softmax_of_a_free_width_walks_whole_rows() {
    let graph = Graph::new();
    let width = graph.free(32000);
    let logits = graph.input(
        Shape::of([1, 1, 1, 32000]).freed(&[(3, width)]),
        Element::Single,
    );
    let probabilities = graph.softmax(logits);
    graph.retain(probabilities);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let kinds = kinds(&runtime, &graph, &program);
    assert!(
        !kinds.contains(&Kind::SoftmaxFold),
        "a row whose width a binding rules walks it as one workgroup",
    );
    let values = data(32000, 53);
    runtime.bind(&program, &[32000]);
    runtime.write(&program, logits, &values);
    runtime.run(&program);
    close(
        &runtime.read(&program, probabilities),
        &softmax_reference(&values, 32000),
        1e-5,
        "a bound softmax",
    );
}

#[test]
fn a_starved_softmax_narrows_every_column_section_it_folds() {
    let rows = 1u32;
    let columns = 4096u32;
    let graph = Graph::new();
    let wide = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let logits = graph.cast(wide, Element::Half);
    let probabilities = graph.cast(graph.softmax(logits), Element::Single);
    graph.retain(probabilities);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let plan = Plan::of(
        &graph,
        runtime.alignment(),
        program.profile(),
        DEFAULT_ENCODING_BYTES,
    );
    assert!(plan.kinds().contains(&Kind::SoftmaxFold));
    let values = peaked(rows, columns, 4, 8.0);
    runtime.write(&program, wide, &values);
    runtime.run(&program);
    let narrow = pack(Element::Half, 1.0, &values);
    let held = unpack(Element::Half, values.len(), &narrow);
    let expected = softmax_reference(&held, columns);
    let expected = unpack(
        Element::Half,
        expected.len(),
        &pack(Element::Half, 1.0, &expected),
    );
    close(
        &runtime.read(&program, probabilities),
        &expected,
        1e-3,
        "a narrow sectioned softmax",
    );
}
