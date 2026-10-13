use neura_gpu::{Device, GpuContext, GpuRequest};
use neura_runtime::{Runtime, RuntimeRequest};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

static DEVICES: LazyLock<Mutex<HashMap<GpuRequest, Device>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn runtime(request: RuntimeRequest) -> Runtime {
    let device = DEVICES
        .lock()
        .expect("the test device registry is never poisoned")
        .entry(request.gpu.clone())
        .or_insert_with(|| {
            GpuContext::open(&request.gpu)
                .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
                .device()
                .clone()
        })
        .clone();
    Runtime::from_device(device, request.memory)
}
