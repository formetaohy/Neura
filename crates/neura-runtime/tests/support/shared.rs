use neura_gpu::{Device, GpuContext};
use neura_runtime::{Runtime, RuntimeRequest};
use std::sync::{LazyLock, Mutex};

static HELD: LazyLock<Mutex<Vec<Device>>> = LazyLock::new(|| Mutex::new(Vec::new()));

pub fn runtime(request: RuntimeRequest) -> Runtime {
    let device = GpuContext::open(&request.gpu)
        .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
        .device()
        .clone();
    HELD.lock()
        .expect("a list of held devices is never poisoned")
        .push(device.clone());
    Runtime::from_device(device, request.memory)
}
