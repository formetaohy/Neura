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
