use neura_gpu::{BufferUsages, GpuBuffer, Submission};
use neura_program::{Graph, Init, Shape, Store};
use neura_runtime::{Precision, Runtime, RuntimeRequest};

#[path = "support/references.rs"]
mod references;
#[path = "support/mod.rs"]
mod support;

use references::{matmul_reference, random};
use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn linear(
    graph: &Graph,
    inputs: u32,
    outputs: u32,
) -> (neura_program::Value, neura_program::Value) {
    (
        graph.parameter(
            Shape::matrix(inputs, outputs),
            Init::Uniform {
                low: -0.5,
                high: 0.5,
            },
        ),
        graph.parameter(Shape::vector(outputs), Init::Zero),
    )
}

#[test]
fn one_store_serves_every_shape_of_one_model() {
    let runtime = open();
    let narrow = Graph::new();
    let (weight, bias) = linear(&narrow, 4, 8);
    let data = narrow.input(Shape::matrix(2, 4));
    let out = narrow.relu(narrow.add(narrow.matmul(data, weight), bias));
    narrow.retain(out);
    let weights = runtime.weights(&narrow, Precision::Single);
    let small = runtime.compile(&narrow, &weights);

    let wide = Graph::new();
    let (shared_weight, shared_bias) = linear(&wide, 4, 8);
    let rows = wide.input(Shape::matrix(64, 4));
    let spread = wide.relu(wide.add(wide.matmul(rows, shared_weight), shared_bias));
    let big = runtime.compile(&wide, &weights);

    assert_eq!(
        weights.bytes(),
        (4 * 8 + 8) * 4,
        "the store carries the parameters of the model and nothing else",
    );
    assert_eq!(weights.tensors(), 2);
    assert_eq!(
        small.weights().buffer().allocation(),
        big.weights().buffer().allocation(),
        "two shapes of one model were handed two weight stores",
    );
    assert!(
        big.arena_bytes() > small.arena_bytes(),
        "a wider batch asks for a wider arena",
    );
    assert!(
        small.arena_bytes() >= 16 * 4 && big.arena_bytes() >= 64 * 8 * 4,
        "the arena carries the activations of the shape it plans",
    );

    let values = random(8, 7);
    let mut observed = values.clone();
    observed.extend(random(62 * 4, 11));
    runtime.write(&small, data, &values);
    runtime.write(&big, rows, &observed);
    runtime.run(&small);
    runtime.run(&big);
    let narrow_out = runtime.read(&small, out);
    let wide_out = runtime.read(&big, spread);
    assert_close(&wide_out[..narrow_out.len()], &narrow_out, 1e-5);
}

#[test]
fn a_plan_keeps_its_weights_out_of_the_arena() {
    let graph = Graph::new();
    let (weight, bias) = linear(&graph, 256, 256);
    let data = graph.input(Shape::matrix(1, 256));
    let out = graph.add(graph.matmul(data, weight), bias);
    graph.retain(out);
    let encoding = graph.encode(256, neura_abi::NARROW, Precision::Single);
    assert_eq!(encoding.weights().bytes(), (256 * 256 + 256) * 4);
    assert!(
        encoding.arena_bytes() <= 8 * 1024,
        "the arena holds {} bytes of activations beside a {} kilobyte weight store",
        encoding.arena_bytes(),
        encoding.weights().bytes() / 1024,
    );
}

#[test]
fn half_precision_weights_halve_the_store() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(
        Shape::matrix(8, 16),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    );
    let data = graph.input(Shape::matrix(4, 8));
    let out = graph.matmul(data, weight);
    graph.retain(out);
    let single = runtime.weights(&graph, Precision::Single);
    let half = runtime.weights(&graph, Precision::Half);
    assert_eq!(half.precision(), Precision::Half);
    assert_eq!(half.tensors(), single.tensors());
    assert_eq!(
        half.bytes() * 2,
        single.bytes(),
        "a single half precision tensor holds half the bytes of its single precision twin",
    );

    let program = runtime.compile(&graph, &half);
    let data_values = random(32, 3);
    let weight_values = random(128, 17);
    runtime.write(&program, data, &data_values);
    runtime.write(&program, weight, &weight_values);
    runtime.run(&program);
    assert_close(&runtime.read(&program, weight), &weight_values, 1e-3);
    assert_close(
        &runtime.read(&program, out),
        &matmul_reference(&data_values, &weight_values, 4, 8, 16),
        4e-2,
    );
}

#[test]
fn a_half_weight_store_carries_no_parameter_update() {
    let graph = Graph::new();
    let (weight, _) = linear(&graph, 4, 4);
    let data = graph.input(Shape::matrix(2, 4));
    let loss = graph.sum(graph.matmul(data, weight));
    let gradients = graph.backward(loss);
    let rate = graph.fill(Shape::scalar(), -0.1);
    let step = graph.mul(gradients.of(weight), rate);
    graph.add_into(weight, step);
    let runtime = open();
    let single = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &single);
    assert!(program.updates_weights());
    let half = runtime.weights(&graph, Precision::Half);
    assert!(
        refuses(|| {
            let _ = runtime.compile(&graph, &half);
        }),
        "a half precision store accepted an in-place parameter update",
    );
}

#[test]
fn a_store_serves_no_model_it_was_not_built_for() {
    let graph = Graph::new();
    let (_, _) = linear(&graph, 4, 4);
    let other = Graph::new();
    let (_, _) = linear(&other, 4, 8);
    let runtime = open();
    let weights = runtime.weights(&graph, Precision::Single);
    assert!(
        refuses(|| {
            let _ = runtime.compile(&other, &weights);
        }),
        "two different models were handed one weight store",
    );
}

#[test]
fn a_resident_tensor_survives_every_run() {
    let runtime = open();
    let graph = Graph::new();
    let state = graph.resident(Shape::vector(4));
    let step = graph.input(Shape::vector(4));
    let readout = graph.resident(Shape::vector(4));
    graph.add_into(state, step);
    graph.copy_into(readout, state);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(program.span(state).store, Store::Tensors);
    assert_eq!(program.span(readout).store, Store::Tensors);
    assert_eq!(program.span(step).store, Store::Tensors);
    assert!(
        program.span(readout).offset > program.span(state).offset,
        "two resident tensors were laid out over each other",
    );
    assert!(program.resident_bytes() >= 32);
    assert!(program.tensor_bytes() >= program.resident_bytes());
    runtime.write(&program, step, &[1.0, 2.0, 3.0, 4.0]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, state), &[1.0, 2.0, 3.0, 4.0], 1e-6);
    runtime.run(&program);
    assert_close(&runtime.read(&program, state), &[2.0, 4.0, 6.0, 8.0], 1e-6);
    assert_close(
        &runtime.read(&program, readout),
        &[2.0, 4.0, 6.0, 8.0],
        1e-6,
    );
}

#[test]
fn the_engine_writes_a_resident_tensor_without_the_host() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 12,
        ..Default::default()
    }))
    .expect("device");
    let graph = Graph::new();
    let observation = graph.resident(Shape::vector(4));
    let scale = graph.parameter(Shape::scalar(), Init::Constant(3.0));
    let action = graph.mul(observation, scale);
    graph.retain(action);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);

    let device = runtime.context().device();
    let queue = runtime.context().queue();
    let engine = GpuBuffer::new(
        device,
        "engine frame",
        4 * 4,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    engine.write(queue, bytemuck::cast_slice(&[1.0f32, 2.0, 3.0, 4.0]));
    let mut submission = Submission::new(device, "engine frame");
    submission.copy_buffer_to_buffer(
        engine.buffer(),
        0,
        program.heap().buffer(),
        program.span(observation).offset,
        4 * 4,
    );
    submission.submit(queue);

    runtime.run(&program);
    assert_close(
        &runtime.read(&program, action),
        &[3.0, 6.0, 9.0, 12.0],
        1e-6,
    );
}

#[test]
fn a_program_wider_than_the_heap_is_refused() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 12,
        heap_bytes: 1 << 12,
        ..Default::default()
    }))
    .expect("device");
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4096));
    graph.relu(data);
    let weights = runtime.weights(&graph, Precision::Single);
    assert!(
        refuses(|| {
            let _ = runtime.compile(&graph, &weights);
        }),
        "a tape whose tensors outrun the device heap was compiled",
    );
}

#[test]
fn a_dropped_program_returns_its_tensors_to_the_heap() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 20,
        heap_bytes: 1 << 20,
        ..Default::default()
    }))
    .expect("device");
    let graph = Graph::new();
    let (weight, _) = linear(&graph, 256, 256);
    let data = graph.input(Shape::matrix(64, 256));
    let out = graph.relu(graph.matmul(data, weight));
    graph.retain(out);
    let weights = runtime.weights(&graph, Precision::Single);
    let first = runtime.compile(&graph, &weights);
    let span = first.span(out);
    runtime.write(&first, data, &random(64 * 256, 41));
    runtime.run(&first);
    drop(first);
    let second = runtime.compile(&graph, &weights);
    assert_eq!(second.span(out), span);
    assert_close(&runtime.read(&second, out), &[0.0; 64 * 256], 1e-9);
    assert!(
        second.tensor_bytes() + weights.bytes() <= runtime.heap_bytes(),
        "{} bytes of tensors and {} bytes of weights outrun the {} byte heap",
        second.tensor_bytes(),
        weights.bytes(),
        runtime.heap_bytes(),
    );
}

#[test]
fn a_dropped_store_returns_its_words_to_the_heap() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 12,
        heap_bytes: 64 << 10,
        ..Default::default()
    }))
    .expect("device");
    let graph = Graph::new();
    let (weight, _) = linear(&graph, 96, 96);
    assert_eq!(weight.shape(), Shape::matrix(96, 96));
    for _ in 0..8 {
        let weights = runtime.weights(&graph, Precision::Single);
        assert_eq!(weights.tensors(), 2);
        assert!(
            weights.bytes() < runtime.heap_bytes(),
            "a store of {} bytes does not fit beside itself in the {} byte heap",
            weights.bytes(),
            runtime.heap_bytes(),
        );
    }
    let weights = runtime.weights(&graph, Precision::Single);
    assert_eq!(weights.tensors(), 2);
}

#[test]
fn a_program_keeps_the_store_it_was_built_with() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 12,
        heap_bytes: 40 << 10,
        ..Default::default()
    }))
    .expect("device");
    let graph = Graph::new();
    let (weight, _) = linear(&graph, 64, 64);
    let data = graph.input(Shape::matrix(2, 64));
    let out = graph.matmul(data, weight);
    graph.retain(out);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    drop(weights);
    runtime.write(&program, data, &random(128, 5));
    runtime.run(&program);
    assert_eq!(runtime.read(&program, out).len(), 128);
    drop(program);
    let rebuilt = runtime.weights(&graph, Precision::Single);
    assert_eq!(rebuilt.tensors(), 2);
}

#[test]
fn two_stores_of_one_model_hold_their_own_words() {
    let runtime = open();
    let graph = Graph::new();
    let (weight, _) = linear(&graph, 32, 32);
    let data = graph.input(Shape::matrix(2, 32));
    let out = graph.matmul(data, weight);
    graph.retain(out);
    let first = runtime.weights(&graph, Precision::Single);
    let second = runtime.weights(&graph, Precision::Single);
    assert_ne!(
        first.offset(),
        second.offset(),
        "two stores of one model were laid over each other",
    );
    let narrow = runtime.compile(&graph, &first);
    let wide = runtime.compile(&graph, &second);
    runtime.write(&narrow, weight, &[1.0; 32 * 32]);
    runtime.write(&wide, weight, &[2.0; 32 * 32]);
    runtime.write(&narrow, data, &[1.0; 64]);
    runtime.write(&wide, data, &[1.0; 64]);
    runtime.run(&narrow);
    runtime.run(&wide);
    assert_close(&runtime.read(&narrow, out), &[32.0; 64], 1e-4);
    assert_close(&runtime.read(&wide, out), &[64.0; 64], 1e-4);
}

#[test]
fn one_device_program_serves_every_store_of_its_model() {
    let runtime = open();
    let graph = Graph::new();
    let (weight, _) = linear(&graph, 32, 32);
    let data = graph.input(Shape::matrix(2, 32));
    graph.retain(graph.matmul(data, weight));
    let first = runtime.weights(&graph, Precision::Half);
    let second = runtime.weights(&graph, Precision::Half);
    assert_ne!(first.offset(), second.offset());
    runtime.compile(&graph, &first);
    runtime.compile(&graph, &second);
    assert_eq!(
        runtime.declared_kernels(),
        1,
        "a device program that carries the stores it addresses is rebuilt for every store",
    );
}

#[test]
fn a_half_store_keeps_an_odd_tensor() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(3), Init::Zero);
    let scaled = graph.mul(weight, graph.fill(Shape::vector(3), 2.0));
    graph.retain(scaled);
    let half = runtime.weights(&graph, Precision::Half);
    let program = runtime.compile(&graph, &half);
    let values = [0.25, -0.5, 1.75];
    runtime.write(&program, weight, &values);
    runtime.run(&program);
    assert_close(&runtime.read(&program, weight), &values, 1e-3);
    assert_close(&runtime.read(&program, scaled), &[0.5, -1.0, 3.5], 1e-3);
}

#[test]
fn a_store_of_another_runtime_is_refused() {
    let graph = Graph::new();
    let (weight, _) = linear(&graph, 4, 4);
    let data = graph.input(Shape::matrix(2, 4));
    graph.retain(graph.matmul(data, weight));
    let first = open();
    let second = open();
    let weights = first.weights(&graph, Precision::Single);
    assert!(
        refuses(|| {
            let _ = second.compile(&graph, &weights);
        }),
        "a weight store of another runtime's heap was compiled into a tape",
    );
    let program = first.compile(&graph, &weights);
    assert!(
        refuses(|| second.run(&program)),
        "a program of another runtime's heap was run",
    );
    assert!(
        refuses(|| {
            let _ = second.read(&program, weight);
        }),
        "a program of another runtime's heap was read",
    );
    assert!(
        refuses(|| second.write(&program, weight, &[1.0; 16])),
        "a program of another runtime's heap was written",
    );
}
