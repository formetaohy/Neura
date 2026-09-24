use crate::{Backends, BufferUsages, Device, GpuBuffer, GpuRequest, Queue};
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
