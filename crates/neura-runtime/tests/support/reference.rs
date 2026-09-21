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

pub fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    neura_program::Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}
