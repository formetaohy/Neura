use crate::native::FRAMES_IN_FLIGHT;
use crate::{
    Backends, BufferUsages, Device, DeviceType, GpuBuffer, GpuRequest, PowerPreference, Queue,
    Submission,
};
use std::sync::Arc;

fn release(backends: Backends) {
    let device = Device::open(&GpuRequest {
        backends,
        ..Default::default()
    })
    .expect("a native compute device");
    let state = Arc::downgrade(&device.state);
    let queue = Queue::of(&device);
    let buffer = GpuBuffer::new(&device, "pending upload", 4, BufferUsages::COPY_DST);
    buffer.write(&queue, &[1, 0, 0, 0]);
    drop(buffer);
    drop(queue);
    drop(device);
    assert!(
        state.upgrade().is_none(),
        "a pending upload retained its device"
    );
}

#[test]
fn pending_uploads_do_not_keep_a_device_alive() {
    for backends in Backends::PLATFORM {
        release(backends);
    }
}

fn recycling(backends: Backends) {
    let device = Device::open(&GpuRequest {
        backends,
        ..Default::default()
    })
    .expect("a native compute device");
    let queue = Queue::of(&device);
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
    let first = GpuBuffer::new(&device, "a recycled target", 16, usage);
    let second = GpuBuffer::new(&device, "another recycled target", 16, usage);
    for round in 0..(FRAMES_IN_FLIGHT * 3) {
        first.write(&queue, &[round as u8, 1, 2, 3]);
        second.write(&queue, &[4, 5, 6, round as u8]);
        let mut submission = Submission::new(&device, "frame recycling");
        submission.clear(&second, 0, 16);
        submission.submit(&queue);
        assert!(
            device.native().frames() <= FRAMES_IN_FLIGHT,
            "a queue records no more than {FRAMES_IN_FLIGHT} frames at once",
        );
    }
    queue.drain();
    assert!(
        device.native().frames() <= FRAMES_IN_FLIGHT,
        "a queue of {} submissions records no frame of its own for each of them",
        FRAMES_IN_FLIGHT * 3,
    );
}

#[test]
fn a_queue_recycles_the_frames_of_its_submissions() {
    for backends in Backends::PLATFORM {
        recycling(backends);
    }
}

#[test]
fn a_power_class_ranks_hardware_by_its_preference() {
    let performance = PowerPreference::HighPerformance;
    let power = PowerPreference::LowPower;
    assert!(DeviceType::Discrete.rank(performance) > DeviceType::Integrated.rank(performance));
    assert!(DeviceType::Integrated.rank(performance) > DeviceType::Virtual.rank(performance));
    assert!(DeviceType::Virtual.rank(performance) > DeviceType::Cpu.rank(performance));
    assert!(DeviceType::Cpu.rank(performance) > DeviceType::Other.rank(performance));
    assert!(DeviceType::Integrated.rank(power) > DeviceType::Discrete.rank(power));
    assert!(DeviceType::Discrete.rank(power) > DeviceType::Virtual.rank(power));
}

#[cfg(dx12_backend)]
#[test]
fn a_cached_dxil_the_driver_refuses_is_a_miss() {
    use crate::GpuContext;
    use crate::native::dx12;
    use neura_compiler::{Read, ReadWrite, kernel};

    #[kernel(workgroup_size = 64)]
    fn scale(lid: u32, input: Read<u32>, output: ReadWrite<u32>) {
        output[lid] = input[lid] * 3u32;
    }

    let directory =
        std::env::temp_dir().join(format!("neura-cache-refused-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&directory);
    let context = GpuContext::open(&GpuRequest {
        backends: Backends::DX12,
        artifacts: Some(directory.clone()),
        ..Default::default()
    })
    .expect("a D3D12 compute device");
    let cache = context.artifact_cache().clone();
    let program = scale();
    let (_, _, key) = dx12::hlsl(&program);
    let poison = b"a dxil module this driver refuses";
    cache.store(&key, poison);
    let pipeline = context.declare(program);
    pipeline.compile();
    assert!(pipeline.is_compiled());
    assert_ne!(
        cache.load(&key).as_deref(),
        Some(&poison[..]),
        "a refused artifact is replaced by a fresh compile",
    );
    assert_eq!(cache.evictions(), 1, "a refused artifact leaves the cache");
    drop(pipeline);
    let _ = std::fs::remove_dir_all(&directory);
}
