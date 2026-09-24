use neura_gpu::{
    Backend, Backends, Binding, BindingSpec, BufferUsages, ComputeProgram, GpuBuffer, GpuContext,
    GpuRequest, Submission,
};

fn compute(backend: Backend, backends: Backends) {
    let context = pollster::block_on(GpuContext::open(&GpuRequest {
        backends,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("{backend:?} could not run native compute: {error}"));
    assert_eq!(context.adapter_info().backend, backend);
    let device = context.device().clone();
    let queue = context.queue().clone();
    let alignment = context.binding_alignment();
    let source = GpuBuffer::new(
        &device,
        "native input",
        alignment + 256,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let middle = GpuBuffer::new(
        &device,
        "native intermediate",
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let output = GpuBuffer::new(
        &device,
        "native output",
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let readback = GpuBuffer::new(
        &device,
        "native readback",
        256,
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );
    let pipeline = context.declare(ComputeProgram::new(
        "native buffer dispatch",
        "@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_index) lane: u32) {
    output[lane] = input[lane] * 2u;
}",
        "main",
        &[
            BindingSpec::dynamic_storage(0),
            BindingSpec::writable_storage(1),
        ],
    ));
    let first = pipeline.bind_group(&[
        Binding {
            index: 0,
            buffer: source.binding(0, 256),
        },
        Binding {
            index: 1,
            buffer: middle.binding(0, 256),
        },
    ]);
    let second = pipeline.bind_group(&[
        Binding {
            index: 0,
            buffer: middle.binding(0, 256),
        },
        Binding {
            index: 1,
            buffer: output.binding(0, 256),
        },
    ]);
    let values = (1..=64u32).collect::<Vec<_>>();
    source.write_at(&queue, alignment, bytemuck::cast_slice(&values));
    let mut submission = Submission::new(&device, "native compute chain");
    submission.dispatch(&pipeline, &first, &[alignment as u32], [1, 1, 1]);
    submission.dispatch(&pipeline, &second, &[0], [1, 1, 1]);
    submission.submit(&queue);
    let mut transfer = Submission::new(&device, "native transfer");
    transfer.copy(&output, 0, &readback, 0, 256);
    let index = transfer.submit(&queue);
    assert!(pipeline.is_compiled());
    drop(first);
    drop(second);
    drop(pipeline);
    drop(source);
    drop(middle);
    drop(output);
    drop(context);
    let result = readback.read(&queue, index, 256);
    let values = bytemuck::cast_slice::<u8, u32>(&result);
    assert_eq!(
        values,
        (1..=64u32).map(|number| number * 4).collect::<Vec<_>>()
    );
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[test]
fn vulkan_executes_native_compute() {
    compute(Backend::Vulkan, Backends::VULKAN);
}

#[cfg(target_os = "windows")]
#[test]
fn dx12_executes_native_compute() {
    compute(Backend::Dx12, Backends::DX12);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_executes_native_compute() {
    compute(Backend::Metal, Backends::METAL);
}
