pub fn open_with(backend: neura_gpu::Backend) -> neura_runtime::Runtime {
    neura_runtime::Runtime::open(neura_runtime::RuntimeRequest {
        gpu: neura_gpu::GpuRequest {
            backend: Some(backend),
            ..Default::default()
        },
        memory: neura_runtime::MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
    })
    .unwrap_or_else(|error| panic!("no device runs the tests over {backend:?}: {error}"))
}
