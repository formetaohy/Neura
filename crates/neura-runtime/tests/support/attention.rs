#[derive(Clone, Copy)]
pub struct Shapes {
    pub heads: u32,
    pub batch: u32,
    pub queries: u32,
    pub keys: u32,
    pub width: u32,
    pub causal: bool,
    pub scale: f32,
}

fn planes(shapes: Shapes) -> usize {
    (shapes.heads * shapes.batch) as usize
}

fn masked(shapes: Shapes, row: usize, column: usize) -> bool {
    if shapes.causal && column > row {
        return false;
    }
    true
}

fn score(
    shapes: Shapes,
    queries: &[f32],
    keys: &[f32],
    plane: usize,
    row: usize,
    column: usize,
) -> f64 {
    let width = shapes.width as usize;
    let at = (plane * shapes.queries as usize + row) * width;
    let other = (plane * shapes.keys as usize + column) * width;
    (0..width)
        .map(|depth| f64::from(queries[at + depth]) * f64::from(keys[other + depth]))
        .sum::<f64>()
        * f64::from(shapes.scale)
}

pub fn attention_forward(
    shapes: Shapes,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let width = shapes.width as usize;
    let mut out = vec![0.0f32; planes(shapes) * shapes.queries as usize * width];
    let mut statistics = vec![0.0f32; planes(shapes) * shapes.queries as usize];
    for plane in 0..planes(shapes) {
        for row in 0..shapes.queries as usize {
            let mut logits = Vec::with_capacity(shapes.keys as usize);
            for column in 0..shapes.keys as usize {
                if masked(shapes, row, column) {
                    logits.push(score(shapes, queries, keys, plane, row, column));
                } else {
                    logits.push(f64::NEG_INFINITY);
                }
            }
            let largest = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let total = logits
                .iter()
                .map(|logit| (logit - largest).exp())
                .sum::<f64>();
            statistics[plane * shapes.queries as usize + row] = (largest + total.ln()) as f32;
            for depth in 0..width {
                let mut sum = 0.0f64;
                for (column, logit) in logits.iter().enumerate() {
                    let weight = (logit - largest).exp() / total;
                    let at = (plane * shapes.keys as usize + column) * width + depth;
                    sum += weight * f64::from(values[at]);
                }
                out[(plane * shapes.queries as usize + row) * width + depth] = sum as f32;
            }
        }
    }
    (out, statistics)
}

pub fn attention_backward(
    shapes: Shapes,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    out: &[f32],
    statistics: &[f32],
    gradient: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let width = shapes.width as usize;
    let mut query_grad = vec![0.0f32; queries.len()];
    let mut key_grad = vec![0.0f32; keys.len()];
    let mut value_grad = vec![0.0f32; values.len()];
    for plane in 0..planes(shapes) {
        for row in 0..shapes.queries as usize {
            let at = (plane * shapes.queries as usize + row) * width;
            let normalizer = f64::from(statistics[plane * shapes.queries as usize + row]);
            let row_dot = (0..width)
                .map(|depth| f64::from(gradient[at + depth]) * f64::from(out[at + depth]))
                .sum::<f64>();
            for column in 0..shapes.keys as usize {
                let other = (plane * shapes.keys as usize + column) * width;
                let weight = if masked(shapes, row, column) {
                    (score(shapes, queries, keys, plane, row, column) - normalizer).exp()
                } else {
                    0.0
                };
                let weighted = (0..width)
                    .map(|depth| f64::from(gradient[at + depth]) * f64::from(values[other + depth]))
                    .sum::<f64>();
                let scored = weight * (weighted - row_dot) * f64::from(shapes.scale);
                for depth in 0..width {
                    value_grad[other + depth] += (weight * f64::from(gradient[at + depth])) as f32;
                    key_grad[other + depth] += (scored * f64::from(queries[at + depth])) as f32;
                    query_grad[at + depth] += (scored * f64::from(keys[other + depth])) as f32;
                }
            }
        }
    }
    (query_grad, key_grad, value_grad)
}
