mod training;

use neura_gpu::{Device, GpuContext, GpuRequest};
use neura_runtime::{MemoryRequest, Runtime};
use std::sync::LazyLock;

pub use training::assert_converges;

static DEVICE: LazyLock<Device> = LazyLock::new(|| {
    GpuContext::open(&GpuRequest::default())
        .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
        .device()
        .clone()
});

pub fn open() -> Runtime {
    open_with(MemoryRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    })
}

pub fn open_with(memory: MemoryRequest) -> Runtime {
    Runtime::from_device(DEVICE.clone(), memory)
}
