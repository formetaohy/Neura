pub mod shared;

use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

pub fn open() -> Runtime {
    shared::runtime(RuntimeRequest {
        memory: MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
        ..Default::default()
    })
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
