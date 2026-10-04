use neura_compiler::{Read, ReadWrite, kernel};
use neura_gpu::{Backends, Binding, BufferUsages, GpuBuffer, GpuContext, GpuRequest, Submission};

#[kernel(workgroup_size = 64)]
fn double(lid: u32, input: Read<u32>, output: ReadWrite<u32>) {
    output[lid] = input[lid] * 2u32;
}

fn compute(backends: Backends) {
    let backend = backends.backend();
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
    let pipeline = context.declare(double());
    let first = pipeline.bind_group(&[
        Binding {
            index: 0,
            buffer: source.binding(alignment, 256),
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
    submission.dispatch(&pipeline, &first, [1, 1, 1]);
    submission.dispatch(&pipeline, &second, [1, 1, 1]);
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

#[test]
fn every_platform_backend_executes_native_compute() {
    for backends in Backends::PLATFORM {
        compute(backends);
    }
}

#[kernel(workgroup_size = 8)]
fn signed_remainders(lid: u32, left: Read<u32>, right: Read<u32>, out: ReadWrite<u32>) {
    out[lid] = ((left[lid] as i32) % (right[lid] as i32)) as u32;
}

#[kernel(workgroup_size = 8)]
fn unsigned_remainders(lid: u32, left: Read<u32>, right: Read<u32>, out: ReadWrite<u32>) {
    out[lid] = left[lid] % right[lid];
}

fn remainder(left: u32, right: u32, signed: bool) -> u32 {
    if signed {
        ((left as i32) % (right as i32)) as u32
    } else {
        left % right
    }
}

fn remainders(backends: Backends, signed: bool) {
    let context = pollster::block_on(GpuContext::open(&GpuRequest {
        backends,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("{backends:?} could not run native compute: {error}"));
    let device = context.device().clone();
    let queue = context.queue().clone();
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    let left = GpuBuffer::new(&device, "remainder left", 256, storage);
    let right = GpuBuffer::new(&device, "remainder right", 256, storage);
    let output = GpuBuffer::new(
        &device,
        "remainder output",
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let readback = GpuBuffer::new(
        &device,
        "remainder readback",
        256,
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );
    let values = [
        (-7i32) as u32,
        7,
        (-7i32) as u32,
        7,
        (-9i32) as u32,
        9,
        (-100i32) as u32,
        0,
    ];
    let divisors = [
        3u32,
        (-3i32) as u32,
        (-3i32) as u32,
        3,
        4,
        (-4i32) as u32,
        33,
        5,
    ];
    left.write_at(&queue, 0, bytemuck::cast_slice(&values));
    right.write_at(&queue, 0, bytemuck::cast_slice(&divisors));
    let pipeline = if signed {
        context.declare(signed_remainders())
    } else {
        context.declare(unsigned_remainders())
    };
    let group = pipeline.bind_group(&[
        Binding {
            index: 0,
            buffer: left.binding(0, 256),
        },
        Binding {
            index: 1,
            buffer: right.binding(0, 256),
        },
        Binding {
            index: 2,
            buffer: output.binding(0, 256),
        },
    ]);
    let mut submission = Submission::new(&device, "remainder dispatch");
    submission.dispatch(&pipeline, &group, [1, 1, 1]);
    submission.submit(&queue);
    let mut transfer = Submission::new(&device, "remainder transfer");
    transfer.copy(&output, 0, &readback, 0, 256);
    let index = transfer.submit(&queue);
    let bytes = readback.read(&queue, index, 256);
    let measured = bytemuck::cast_slice::<u8, u32>(&bytes);
    let expected = values
        .iter()
        .zip(&divisors)
        .map(|(a, b)| remainder(*a, *b, signed))
        .collect::<Vec<_>>();
    assert_eq!(&measured[..values.len()], expected.as_slice());
}

#[test]
fn every_platform_backend_keeps_the_sign_of_a_signed_remainder() {
    for backends in Backends::PLATFORM {
        remainders(backends, true);
    }
}

#[test]
fn every_platform_backend_takes_an_unsigned_remainder() {
    for backends in Backends::PLATFORM {
        remainders(backends, false);
    }
}
