use neura_abi::Element;
use neura_compiler::{ReadWrite, kernel};
use neura_gpu::{Backends, Binding, GpuRequest, Submission};
use neura_graph::{Graph, Init, Shape};
use neura_runtime::{Runtime, RuntimeRequest};

#[kernel(workgroup_size = 64)]
fn seed(lid: u32, arena: ReadWrite<f32>) {
    if lid < 4u32 {
        arena[lid] = (lid + 1u32) as f32;
    }
}

#[test]
fn an_engine_compute_pipeline_writes_directly_into_the_model_arena() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 12,
        ..Default::default()
    }))
    .expect("native compute device");
    let graph = Graph::new();
    let observation = graph.resident(Shape::vector(4), Element::Single);
    let action = graph.mul(observation, graph.fill(Shape::vector(4), 3.0));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let engine = runtime.context().declare(seed());
    let group = engine.bind_group(&[Binding {
        index: 0,
        buffer: program.heap().binding(program.span(observation).offset, 16),
    }]);
    let mut submission = Submission::new(runtime.context().device(), "engine frame");
    submission.dispatch(&engine, &group, &[], [1, 1, 1]);
    submission.submit(runtime.context().queue());
    runtime.run(&program);
    assert_eq!(runtime.read(&program, action), [3.0, 6.0, 9.0, 12.0]);
}

fn narrow_precision_inference(backends: Backends, element: Element) {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        gpu: GpuRequest {
            backends,
            ..Default::default()
        },
        ..Default::default()
    }))
    .expect("a native compute device");
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 4), Init::Zero, element);
    let input = graph.input(Shape::matrix(2, 4), element);
    let product = graph.matmul(input, weight);
    graph.retain(product);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        weight,
        &[
            1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0,
        ],
    );
    runtime.write(&program, input, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, product),
        [1.0, 4.0, 9.0, 16.0, 5.0, 12.0, 21.0, 32.0],
        "a {} store feeds the device the numbers it declares",
        element.name(),
    );
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[test]
fn vulkan_runs_narrow_precision_rust_kernels() {
    narrow_precision_inference(Backends::VULKAN, Element::Half);
    narrow_precision_inference(Backends::VULKAN, Element::Bfloat16);
}

#[cfg(target_os = "windows")]
#[test]
fn dx12_runs_narrow_precision_rust_kernels() {
    narrow_precision_inference(Backends::DX12, Element::Half);
    narrow_precision_inference(Backends::DX12, Element::Bfloat16);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_runs_narrow_precision_rust_kernels() {
    narrow_precision_inference(Backends::METAL, Element::Half);
    narrow_precision_inference(Backends::METAL, Element::Bfloat16);
}

fn training_tape(backends: Backends) {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        gpu: GpuRequest {
            backends,
            ..Default::default()
        },
        ..Default::default()
    }))
    .expect("a native compute device");
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(4, 512), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(512, 8), Init::Zero, Element::Single);
    let product = graph.matmul(left, right);
    let loss = graph.sum(product);
    let gradient = graph.backward(loss).of(left);
    graph.retain(gradient);
    graph.retain(product);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, left, &[1.0; 4 * 512]);
    let mut right_values = Vec::new();
    for _ in 0..512 {
        right_values.extend((1..=8).map(|column| column as f32));
    }
    runtime.write(&program, right, &right_values);
    runtime.run(&program);
    let expected = (1..=8)
        .map(|column| column as f32 * 512.0)
        .collect::<Vec<_>>();
    for row in runtime.read(&program, product).chunks(8) {
        assert_eq!(row, expected);
    }
    assert_eq!(runtime.read(&program, gradient), [36.0; 4 * 512]);
    assert_eq!(
        runtime.read(&program, loss),
        [expected.iter().sum::<f32>() * 4.0]
    );
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[test]
fn vulkan_executes_a_complete_training_tape() {
    training_tape(Backends::VULKAN);
}

#[cfg(target_os = "windows")]
#[test]
fn dx12_executes_a_complete_training_tape() {
    training_tape(Backends::DX12);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_executes_a_complete_training_tape() {
    training_tape(Backends::METAL);
}
