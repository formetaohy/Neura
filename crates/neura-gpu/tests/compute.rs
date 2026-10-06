use neura_compiler::{Read, ReadWrite, kernel};
use neura_gpu::{
    Backends, Binding, BufferUsages, DeviceType, GpuBuffer, GpuContext, GpuRequest, Submission,
};

#[kernel(workgroup_size = 64)]
fn double(lid: u32, input: Read<u32>, output: ReadWrite<u32>) {
    output[lid] = input[lid] * 2u32;
}

fn compute(backends: Backends) {
    let backend = backends.backend();
    let context = GpuContext::open(&GpuRequest {
        backends,
        ..Default::default()
    })
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
    let context = GpuContext::open(&GpuRequest {
        backends,
        ..Default::default()
    })
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

#[kernel(workgroup_size = 8)]
fn sums_before_zero(lid: u32, input: Read<u32>, out: ReadWrite<u32>) {
    let mut total = 0u32;
    let mut index = 0u32;
    loop {
        if index >= 8u32 {
            break;
        }
        let at = lid * 8u32 + index;
        match input[at] {
            0u32 => {
                break;
            }
            _ => {
                total += input[at];
            }
        }
        index += 1u32;
    }
    out[lid] = total;
}

fn breaks_out_of_a_match(backends: Backends) {
    let context = GpuContext::open(&GpuRequest {
        backends,
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("{backends:?} could not run native compute: {error}"));
    let device = context.device().clone();
    let queue = context.queue().clone();
    let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    let input = GpuBuffer::new(&device, "break input", 256, storage);
    let output = GpuBuffer::new(
        &device,
        "break output",
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let readback = GpuBuffer::new(
        &device,
        "break readback",
        256,
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );
    let values = (0..64u32)
        .map(|index| if index % 8 == index / 8 { 0 } else { index + 1 })
        .collect::<Vec<_>>();
    let expected = (0..8u32)
        .map(|row| {
            (0..8u32)
                .take_while(|column| *column != row)
                .map(|column| row * 8 + column + 1)
                .sum::<u32>()
        })
        .collect::<Vec<_>>();
    input.write_at(&queue, 0, bytemuck::cast_slice(&values));
    let pipeline = context.declare(sums_before_zero());
    let group = pipeline.bind_group(&[
        Binding {
            index: 0,
            buffer: input.binding(0, 256),
        },
        Binding {
            index: 1,
            buffer: output.binding(0, 256),
        },
    ]);
    let mut submission = Submission::new(&device, "break dispatch");
    submission.dispatch(&pipeline, &group, [1, 1, 1]);
    submission.submit(&queue);
    let mut transfer = Submission::new(&device, "break transfer");
    transfer.copy(&output, 0, &readback, 0, 256);
    let index = transfer.submit(&queue);
    let bytes = readback.read(&queue, index, 256);
    let measured = bytemuck::cast_slice::<u8, u32>(&bytes);
    assert_eq!(&measured[..8], expected.as_slice());
}

#[test]
fn every_platform_backend_leaves_the_loop_a_match_breaks() {
    for backends in Backends::PLATFORM {
        breaks_out_of_a_match(backends);
    }
}

#[kernel(workgroup_size = 1)]
fn stamp_past_a_gigabyte(lid: u32, index: Read<u32>, storage: ReadWrite<u32>) {
    storage[index[lid]] += 7u32;
}

fn past_a_gigabyte(backends: Backends) {
    let backend = backends.backend();
    let context = GpuContext::open(&GpuRequest {
        backends,
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("{backend:?} could not run native compute: {error}"));
    if context.adapter_info().device_type == DeviceType::Cpu {
        return;
    }
    let bytes = (1u64 << 30) + 256;
    let limits = context.limits();
    assert!(
        limits.max_storage_buffer_binding_size >= bytes,
        "a {backend:?} device binds {} bytes in one storage binding, and the limit this device reports must reach the whole resource it addresses",
        limits.max_storage_buffer_binding_size,
    );
    let device = context.device().clone();
    let queue = context.queue().clone();
    let far = 1u32 << 28;
    let storage = GpuBuffer::new(
        &device,
        "native gigabyte storage",
        bytes,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
    );
    let index = GpuBuffer::new(
        &device,
        "native gigabyte index",
        4,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let readback = GpuBuffer::new(
        &device,
        "native gigabyte readback",
        8,
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );
    index.write(&queue, &far.to_ne_bytes());
    let pipeline = context.declare(stamp_past_a_gigabyte());
    let group = pipeline.bind_group(&[
        Binding {
            index: 0,
            buffer: index.binding(0, 4),
        },
        Binding {
            index: 1,
            buffer: storage.binding(0, bytes),
        },
    ]);
    let mut setup = Submission::new(&device, "native gigabyte setup");
    setup.clear(&storage, 0, 4);
    setup.clear(&storage, 1 << 30, 4);
    setup.submit(&queue);
    let mut submission = Submission::new(&device, "native gigabyte dispatch");
    submission.dispatch(&pipeline, &group, [1, 1, 1]);
    submission.dispatch(&pipeline, &group, [1, 1, 1]);
    submission.submit(&queue);
    let mut transfer = Submission::new(&device, "native gigabyte transfer");
    transfer.copy(&storage, 0, &readback, 0, 4);
    transfer.copy(&storage, 1 << 30, &readback, 4, 4);
    let index = transfer.submit(&queue);
    drop(storage);
    drop(context);
    let measured = readback.read(&queue, index, 8);
    assert_eq!(
        u32::from_ne_bytes(measured[0..4].try_into().expect("four bytes")),
        0,
        "a task that stamps an element past a gigabyte leaves the first element alone",
    );
    assert_eq!(
        u32::from_ne_bytes(measured[4..8].try_into().expect("four bytes")),
        14,
        "two tasks read and stamped the element one gigabyte into the binding",
    );
}

#[test]
fn every_platform_backend_reaches_past_a_gigabyte() {
    for backends in Backends::PLATFORM {
        past_a_gigabyte(backends);
    }
}
