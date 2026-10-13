pub fn matmul_reference(
    left: &[f32],
    right: &[f32],
    rows: u32,
    depth: u32,
    columns: u32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; (rows * columns) as usize];
    for row in 0..rows as usize {
        let target = &mut out[row * columns as usize..(row + 1) * columns as usize];
        for step in 0..depth as usize {
            let value = left[row * depth as usize + step];
            let source = &right[step * columns as usize..(step + 1) * columns as usize];
            for (output, operand) in target.iter_mut().zip(source) {
                *output += value * operand;
            }
        }
    }
    out
}
