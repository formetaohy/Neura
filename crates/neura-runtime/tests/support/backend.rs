pub fn backends() -> Vec<neura_runtime::Backends> {
    let mut offered = vec![neura_runtime::Backends::VULKAN];
    if cfg!(target_os = "windows") {
        offered.insert(0, neura_runtime::Backends::DX12);
    }
    if cfg!(target_os = "macos") {
        offered.insert(0, neura_runtime::Backends::METAL);
    }
    offered
}

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
