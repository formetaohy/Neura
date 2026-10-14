use neura_abi::{Element, Kind};
use neura_graph::{Graph, Init, Shape};
use neura_plan::{DEFAULT_ENCODING_BYTES, Plan};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

use crate::support::shared;

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

fn classes_of(rows: u32, columns: u32, seed: u32) -> Vec<f32> {
    data(rows as usize, seed)
        .into_iter()
        .map(|value| {
            (((value + 0.5) * f32::from(u16::try_from(columns).unwrap_or(u16::MAX))).floor())
                % columns as f32
        })
        .collect()
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

fn class_log_softmax_reference(input: &[f32], classes: &[f32], columns: u32) -> Vec<f32> {
    input
        .chunks(columns as usize)
        .zip(classes)
        .map(|(row, chosen)| {
            let largest = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f64 = row
                .iter()
                .map(|value| f64::from(value - largest).exp())
                .sum();
            (f64::from(row[*chosen as usize] - largest) - total.ln()) as f32
        })
        .collect()
}

fn class_log_softmax_gradient_reference(
    input: &[f32],
    classes: &[f32],
    weight: &[f32],
    columns: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (row, chunk) in input.chunks(columns as usize).enumerate() {
        let base = row * columns as usize;
        let largest = chunk.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total: f64 = chunk
            .iter()
            .map(|value| f64::from(value - largest).exp())
            .sum();
        let chosen = classes[row] as usize;
        for column in 0..columns as usize {
            let probability = f64::from(chunk[column] - largest).exp() / total;
            let target = f64::from(u8::from(column == chosen));
            out[base + column] = (f64::from(weight[row]) * (target - probability)) as f32;
        }
    }
    out
}

fn plan(runtime: &Runtime, graph: &Graph, program: &neura_runtime::Program) -> Plan {
    Plan::of(
        graph,
        runtime.alignment(),
        program.profile(),
        DEFAULT_ENCODING_BYTES,
    )
}

#[test]
fn a_class_log_softmax_weighs_the_class_every_row_chose() {
    let rows = 6u32;
    let columns = 5u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let classes = graph.input(Shape::matrix(rows, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    graph.retain(chosen);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = plan(&runtime, &graph, &program);
    assert!(held.kinds().contains(&Kind::ClassLogSoftmax));
    assert!(
        !held.kinds().contains(&Kind::LogSoftmax),
        "a class log softmax weighs one class of every row instead of writing the whole row",
    );
    let logits_data = data((rows * columns) as usize, 17);
    let class_data = classes_of(rows, columns, 29);
    runtime.write(&program, logits, &logits_data);
    runtime.write(&program, classes, &class_data);
    runtime.run(&program);
    close(
        &runtime.read(&program, chosen),
        &class_log_softmax_reference(&logits_data, &class_data, columns),
        1e-6,
        "a class log softmax",
    );
}

#[test]
fn a_starved_class_log_softmax_folds_every_column_section_back() {
    let rows = 2u32;
    let columns = 8192u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let classes = graph.input(Shape::matrix(rows, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    graph.retain(chosen);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = plan(&runtime, &graph, &program);
    assert!(
        held.kinds().contains(&Kind::ClassLogSoftmaxFold),
        "two rows of {columns} numbers starve the workgroups of the device",
    );
    let logits_data = peaked(rows, columns, 8, 12.0);
    let class_data = vec![3.0f32, columns as f32 - 1.0];
    runtime.write(&program, logits, &logits_data);
    runtime.write(&program, classes, &class_data);
    runtime.run(&program);
    close(
        &runtime.read(&program, chosen),
        &class_log_softmax_reference(&logits_data, &class_data, columns),
        1e-5,
        "a sectioned class log softmax",
    );
}

#[test]
fn a_class_log_softmax_gradient_is_the_class_it_chose_less_the_probability() {
    let rows = 4u32;
    let columns = 6u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let classes = graph.input(Shape::matrix(rows, 1), Element::Single);
    let weight = graph.input(Shape::matrix(rows, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    let loss = graph.sum(graph.mul(chosen, weight));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(logits);
    graph.retain(loss);
    graph.retain(gradient);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = plan(&runtime, &graph, &program);
    assert!(held.kinds().contains(&Kind::ClassLogSoftmax));
    assert!(held.kinds().contains(&Kind::ClassLogSoftmaxGrad));
    let logits_data = data((rows * columns) as usize, 41);
    let class_data = classes_of(rows, columns, 43);
    let weight_data = vec![1.5f32, -0.75, 0.25, 2.0];
    runtime.write(&program, logits, &logits_data);
    runtime.write(&program, classes, &class_data);
    runtime.write(&program, weight, &weight_data);
    runtime.run(&program);
    close(
        &runtime.read(&program, gradient),
        &class_log_softmax_gradient_reference(&logits_data, &class_data, &weight_data, columns),
        1e-5,
        "a class log softmax gradient",
    );
}

#[test]
fn a_class_log_softmax_holds_no_row_of_probabilities_of_its_own() {
    let rows = 512u32;
    let columns = 8192u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let classes = graph.input(Shape::matrix(rows, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    let loss = graph.sum(chosen);
    let gradients = graph.backward(loss);
    let gradient = gradients.of(logits);
    graph.retain(loss);
    graph.retain(gradient);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let logits_bytes = u64::from(rows) * u64::from(columns) * 4;
    assert!(
        program.tensor_bytes() < 3 * logits_bytes,
        "a class log softmax of {rows} rows by {columns} classes holds {} bytes where the one hot target, the log softmax and the weighted loss it replaces hold {logits_bytes} bytes each",
        program.tensor_bytes(),
    );
}

#[test]
fn every_profile_the_device_offers_weighs_the_same_class() {
    let rows = 7u32;
    let columns = 5u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let classes = graph.input(Shape::matrix(rows, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    graph.retain(chosen);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let logits_data = data((rows * columns) as usize, 53);
    let class_data = classes_of(rows, columns, 59);
    let expected = class_log_softmax_reference(&logits_data, &class_data, columns);
    for profile in runtime.profiles() {
        let program = runtime.compile_with(&graph, &weights, profile);
        runtime.write(&program, logits, &logits_data);
        runtime.write(&program, classes, &class_data);
        runtime.run(&program);
        close(
            &runtime.read(&program, chosen),
            &expected,
            1e-5,
            &format!("{:?}", program.profile().workgroup()),
        );
    }
}

#[test]
fn a_class_outside_the_classes_a_row_weighs_is_refused() {
    let rows = 3u32;
    let columns = 4u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let classes = graph.input(Shape::matrix(rows, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    graph.retain(chosen);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    for held in [vec![0.0, 4.0, 2.0], vec![0.0, 1.5, 2.0]] {
        runtime.write(&program, classes, &held);
        runtime.run(&program);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.read(&program, chosen)
        }));
        assert!(
            refused.is_err(),
            "a class of {held:?} names no class of {columns} on every row",
        );
    }
}

#[test]
fn a_sectioned_class_log_softmax_gradient_applies_the_chain_it_carries() {
    let rows = 1u32;
    let columns = 8192u32;
    let graph = Graph::new();
    let logits = graph.gradient_input(Shape::matrix(rows, columns), Element::Single);
    let classes = graph.input(Shape::matrix(rows, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    let loss = graph.sum(chosen);
    let gradients = graph.backward(loss);
    let scaled = graph.mul(
        gradients.of(logits),
        graph.fill(Shape::matrix(rows, columns), 0.5),
    );
    graph.retain(scaled);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = plan(&runtime, &graph, &program);
    let chained = held
        .tasks()
        .as_chunks::<{ std::mem::size_of::<neura_abi::TaskRecord>() }>()
        .0
        .iter()
        .map(|task| bytemuck::pod_read_unaligned::<neura_abi::TaskRecord>(task))
        .any(|task| Kind::of(task.kind) == Kind::ClassLogSoftmaxGrad && task.steps > 0);
    assert!(
        chained,
        "a scale the plan folds into the gradient reaches the section that walks the columns",
    );
    let logits_data = data((rows * columns) as usize, 61);
    let class_data = vec![columns as f32 - 1.0];
    runtime.write(&program, logits, &logits_data);
    runtime.write(&program, classes, &class_data);
    runtime.run(&program);
    let mut expected =
        class_log_softmax_gradient_reference(&logits_data, &class_data, &[1.0f32], columns);
    for value in &mut expected {
        *value *= 0.5;
    }
    close(
        &runtime.read(&program, scaled),
        &expected,
        1e-5,
        "a sectioned class log soft max gradient that carries a chain",
    );
}
