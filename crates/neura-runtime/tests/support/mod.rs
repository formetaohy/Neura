pub fn open() -> neura_runtime::Runtime {
    pollster::block_on(neura_runtime::Runtime::open(
        neura_runtime::RuntimeRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
    ))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
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
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index} came back as {actual} where {expected} was expected",
        );
    }
}
