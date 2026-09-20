use neura_program::Init;

pub fn open() -> neura_runtime::Runtime {
    pollster::block_on(neura_runtime::Runtime::open(
        neura_runtime::RuntimeRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
    ))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

pub fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}

pub fn matmul_reference(
    left: &[f32],
    right: &[f32],
    rows: u32,
    depth: u32,
    columns: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; (rows * columns) as usize];
    for row in 0..rows {
        for column in 0..columns {
            let mut total = 0.0f32;
            for step in 0..depth {
                total +=
                    left[(row * depth + step) as usize] * right[(step * columns + column) as usize];
            }
            out[(row * columns + column) as usize] = total;
        }
    }
    out
}

pub fn softmax_reference(input: &[f32], columns: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for (index, row) in input.chunks(columns as usize).enumerate() {
        let largest = row.iter().copied().fold(f32::MIN, f32::max);
        let exponentiated = row.iter().map(|x| (x - largest).exp()).collect::<Vec<_>>();
        let total: f32 = exponentiated.iter().sum();
        for (offset, value) in exponentiated.iter().enumerate() {
            out[index * columns as usize + offset] = value / total;
        }
    }
    out
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
