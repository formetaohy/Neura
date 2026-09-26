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
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    release(Backends::VULKAN);
    #[cfg(target_os = "windows")]
    release(Backends::DX12);
    #[cfg(target_os = "macos")]
    release(Backends::METAL);
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
    #[cfg(any(target_os = "windows", target_os = "linux"))]
    recycling(Backends::VULKAN);
    #[cfg(target_os = "windows")]
    recycling(Backends::DX12);
    #[cfg(target_os = "macos")]
    recycling(Backends::METAL);
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
