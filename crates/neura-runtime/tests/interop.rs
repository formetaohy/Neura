use neura_gpu::{Backends, Binding, BindingSpec, ComputeProgram, GpuRequest, Submission};
use neura_graph::{Graph, Init, Shape};
use neura_runtime::{Precision, Runtime, RuntimeRequest};

#[test]
fn an_engine_compute_pipeline_writes_directly_into_the_model_arena() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 12,
        ..Default::default()
    }))
    .expect("native compute device");
    let graph = Graph::new();
    let observation = graph.resident(Shape::vector(4));
    let action = graph.mul(observation, graph.fill(Shape::vector(4), 3.0));
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    let engine = runtime.context().declare(ComputeProgram::new(
        "engine observation",
        "@group(0) @binding(0) var<storage, read_write> arena: array<f32>;
@compute @workgroup_size(64)
fn seed(@builtin(local_invocation_index) lane: u32) {
    if (lane < 4u) {
        arena[lane] = f32(lane + 1u);
    }
}",
        "seed",
        &[BindingSpec::writable_storage(0)],
    ));
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
    let left = graph.parameter(Shape::matrix(4, 512), Init::Zero);
    let right = graph.parameter(Shape::matrix(512, 8), Init::Zero);
    let product = graph.matmul(left, right);
    let loss = graph.sum(product);
    let gradient = graph.backward(loss).of(left);
    graph.retain(gradient);
    graph.retain(product);
    let weights = runtime.weights(&graph, Precision::Single);
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
