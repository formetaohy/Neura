use neura_program::Init;

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

pub fn log_softmax_reference(input: &[f32], columns: u32) -> Vec<f32> {
    let mut out = input.to_vec();
    for (index, row) in input.chunks(columns as usize).enumerate() {
        let largest = row.iter().copied().fold(f32::MIN, f32::max);
        let total: f32 = row.iter().map(|x| (x - largest).exp()).sum();
        let normalizer = total.ln();
        for (offset, value) in row.iter().enumerate() {
            out[index * columns as usize + offset] = value - largest - normalizer;
        }
    }
    out
}

pub fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}
