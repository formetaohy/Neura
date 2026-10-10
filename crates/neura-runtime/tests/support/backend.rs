use crate::support::shared;
use neura_gpu::{Backend, GpuRequest};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

pub fn open_with(backend: Backend) -> Runtime {
    shared::runtime(RuntimeRequest {
        gpu: GpuRequest {
            backend: Some(backend),
            ..Default::default()
        },
        memory: MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
    })
}
