pub fn open_with(backends: neura_runtime::Backends) -> neura_runtime::Runtime {
    neura_runtime::Runtime::open(neura_runtime::RuntimeRequest {
        gpu: neura_runtime::GpuRequest {
            backends,
            ..Default::default()
        },
        memory: neura_runtime::MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
    })
    .unwrap_or_else(|error| panic!("no device runs the tests over {backends:?}: {error}"))
}
