pub fn argmax_reference(values: &[f32], rows: u32, columns: u32) -> Vec<f32> {
    (0..rows)
        .map(|row| {
            let mut best = f32::MIN;
            let mut index = 0.0f32;
            for column in 0..columns {
                let value = values[(row * columns + column) as usize];
                if value > best {
                    best = value;
                    index = column as f32;
                }
            }
            index
        })
        .collect()
}

pub fn one_hot_reference(indices: &[f32], classes: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; indices.len() * classes as usize];
    for (row, index) in indices.iter().enumerate() {
        out[row * classes as usize + *index as usize] = 1.0;
    }
    out
}

pub fn gather_reference(table: &[f32], indices: &[f32], width: u32) -> Vec<f32> {
    indices
        .iter()
        .flat_map(|index| {
            let row = *index as usize * width as usize;
            table[row..row + width as usize].iter().copied()
        })
        .collect()
}

pub fn scatter_reference(table: &[f32], indices: &[f32], updates: &[f32], width: u32) -> Vec<f32> {
    let mut out = table.to_vec();
    for (row, index) in indices.iter().enumerate() {
        let chosen = *index as usize * width as usize;
        for (column, value) in out[chosen..chosen + width as usize].iter_mut().enumerate() {
            *value += updates[row * width as usize + column];
        }
    }
    out
}

pub fn write_reference(table: &[f32], indices: &[f32], updates: &[f32], width: u32) -> Vec<f32> {
    let mut out = table.to_vec();
    for (row, index) in indices.iter().enumerate() {
        let chosen = *index as usize * width as usize;
        for (column, value) in out[chosen..chosen + width as usize].iter_mut().enumerate() {
            *value = updates[row * width as usize + column];
        }
    }
    out
}

pub fn gumbel_reference(seed: u32, index: u32) -> f32 {
    let mut hash = seed ^ index.wrapping_mul(0x9e37_79b9);
    hash ^= hash >> 16;
    hash = hash.wrapping_mul(0x7feb_352d);
    hash ^= hash >> 15;
    hash = hash.wrapping_mul(0x846c_a68b);
    hash ^= hash >> 16;
    let unit = (hash >> 8) as f32 * (1.0 / 16_777_216.0);
    -(-unit.ln()).ln()
}

pub fn counts(values: &[f32], classes: u32) -> Vec<u32> {
    let mut counts = vec![0u32; classes as usize];
    for value in values {
        counts[*value as usize] += 1;
    }
    counts
}
