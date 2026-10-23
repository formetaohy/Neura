pub mod attention;
pub mod convolution;

use neura_abi::Element;
use neura_gpu::{Device, GpuContext, GpuRequest};
use neura_graph::Init;
use neura_precision::{pack, unpack};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

static DEVICES: LazyLock<Mutex<HashMap<GpuRequest, Device>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

pub fn open() -> Runtime {
    opened(MemoryRequest {
        readback_bytes: 4 << 20,
        ..Default::default()
    })
}

pub fn opened(memory: MemoryRequest) -> Runtime {
    let request = RuntimeRequest {
        memory,
        ..Default::default()
    };
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

pub fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}

pub fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} numbers came back where {} were expected",
        actual.len(),
        expected.len(),
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        if actual == expected {
            continue;
        }
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index} came back as {actual} where {expected} was expected",
        );
    }
}

pub fn rounded(element: Element, values: &[f32]) -> Vec<f32> {
    unpack(element, values.len(), &pack(element, 1.0, values))
}

fn ggml_kind(element: Element) -> u32 {
    match element {
        Element::Half => 1,
        Element::Q8_0 => 8,
        Element::Q3K => 11,
        Element::Q4K => 12,
        Element::Q5K => 13,
        Element::Q6K => 14,
        other => panic!("{} storage carries no ggml type", other.name()),
    }
}

pub fn gguf_bytes(alignment: u64, tensors: &[(Element, &[u64], &[u8])]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    let key = "general.alignment";
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key.as_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&(alignment as u32).to_le_bytes());
    let mut at = 0u64;
    for (element, shape, payload) in tensors {
        let name = element.name();
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in *shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&ggml_kind(*element).to_le_bytes());
        bytes.extend_from_slice(&at.to_le_bytes());
        at = (at + payload.len() as u64).next_multiple_of(alignment);
    }
    while !(bytes.len() as u64).is_multiple_of(alignment) {
        bytes.push(0);
    }
    for (_, _, payload) in tensors {
        bytes.extend_from_slice(payload);
        while !(bytes.len() as u64).is_multiple_of(alignment) {
            bytes.push(0);
        }
    }
    bytes
}
