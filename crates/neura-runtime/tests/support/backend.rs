pub fn open_with(backends: neura_runtime::Backends) -> neura_runtime::Runtime {
    pollster::block_on(neura_runtime::Runtime::open(
        neura_runtime::RuntimeRequest {
            gpu: neura_runtime::GpuRequest {
                backends,
                ..Default::default()
            },
            readback_bytes: 4 << 20,
            ..Default::default()
        },
    ))
    .unwrap_or_else(|error| panic!("no device runs the tests over {backends:?}: {error}"))
}
