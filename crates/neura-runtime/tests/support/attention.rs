#[derive(Clone, Copy)]
pub struct Shapes {
    pub heads: u32,
    pub key_heads: u32,
    pub batch: u32,
    pub queries: u32,
    pub keys: u32,
    pub width: u32,
    pub causal: bool,
    pub origin: u32,
    pub reach: u32,
    pub scale: f32,
}

fn planes(shapes: Shapes) -> usize {
    (shapes.heads * shapes.batch) as usize
}

fn groups(shapes: Shapes) -> usize {
    (shapes.heads / shapes.key_heads) as usize
}

fn key_plane(shapes: Shapes, plane: usize) -> usize {
    let batch = shapes.batch as usize;
    let head = plane / batch;
    (head / groups(shapes)) * batch + plane % batch
}

fn key_position(shapes: Shapes, column: usize) -> usize {
    let keys = shapes.keys as usize;
    let total = shapes.origin as usize + shapes.queries as usize;
    if total <= keys {
        return column;
    }
    column + keys * ((total - 1 - column) / keys)
}

fn masked(shapes: Shapes, row: usize, column: usize) -> bool {
    if !shapes.causal {
        return true;
    }
    let position = key_position(shapes, column);
    let query = shapes.origin as usize + row;
    if position > query {
        return false;
    }
    if shapes.reach > 0 && query - position >= shapes.reach as usize {
        return false;
    }
    true
}

#[derive(Clone, Copy)]
pub struct Bias<'a> {
    pub dims: [u32; 4],
    pub data: &'a [f32],
}

#[derive(Clone, Copy)]
pub struct Walked<'a> {
    pub output: &'a [f32],
    pub statistics: &'a [f32],
    pub gradient: &'a [f32],
    pub bias: Option<Bias<'a>>,
}

fn bias_of(bias: Option<Bias<'_>>, head: usize, batch: usize, row: usize, column: usize) -> f64 {
    let Some(bias) = bias else {
        return 0.0;
    };
    let dims = bias.dims.map(|dim| dim as usize);
    let walked = [head, batch, row, column];
    let mut stride = 1usize;
    let mut address = 0usize;
    for axis in (0..4).rev() {
        address += if dims[axis] == 1 { 0 } else { walked[axis] } * stride;
        stride *= dims[axis].max(1);
    }
    f64::from(bias.data[address])
}

fn score(
    shapes: Shapes,
    queries: &[f32],
    keys: &[f32],
    plane: usize,
    source: usize,
    row: usize,
    column: usize,
) -> f64 {
    let width = shapes.width as usize;
    let at = (plane * shapes.queries as usize + row) * width;
    let other = (source * shapes.keys as usize + column) * width;
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
    attention_forward_beside(shapes, queries, keys, values, None)
}

pub fn attention_forward_beside(
    shapes: Shapes,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    bias: Option<Bias<'_>>,
) -> (Vec<f32>, Vec<f32>) {
    let width = shapes.width as usize;
    let mut out = vec![0.0f32; planes(shapes) * shapes.queries as usize * width];
    let mut statistics = vec![0.0f32; planes(shapes) * shapes.queries as usize];
    for plane in 0..planes(shapes) {
        let source = key_plane(shapes, plane);
        for row in 0..shapes.queries as usize {
            let mut logits = Vec::with_capacity(shapes.keys as usize);
            for column in 0..shapes.keys as usize {
                if masked(shapes, row, column) {
                    let batch = shapes.batch as usize;
                    logits.push(
                        score(shapes, queries, keys, plane, source, row, column)
                            + bias_of(bias, plane / batch, plane % batch, row, column),
                    );
                } else {
                    logits.push(f64::NEG_INFINITY);
                }
            }
            let largest = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            for logit in &mut logits {
                *logit = (*logit - largest).exp();
            }
            let total = logits.iter().sum::<f64>();
            statistics[plane * shapes.queries as usize + row] = (largest + total.ln()) as f32;
            for weight in &mut logits {
                *weight /= total;
            }
            for depth in 0..width {
                let mut sum = 0.0f64;
                for (column, weight) in logits.iter().enumerate() {
                    let at = (source * shapes.keys as usize + column) * width + depth;
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
    attention_backward_beside(
        shapes,
        queries,
        keys,
        values,
        Walked {
            output: out,
            statistics,
            gradient,
            bias: None,
        },
    )
}

pub fn attention_backward_beside(
    shapes: Shapes,
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    walked: Walked<'_>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let Walked {
        output: out,
        statistics,
        gradient,
        bias,
    } = walked;
    let width = shapes.width as usize;
    let mut query_grad = vec![0.0f32; queries.len()];
    let mut key_grad = vec![0.0f32; keys.len()];
    let mut value_grad = vec![0.0f32; values.len()];
    for plane in 0..planes(shapes) {
        let source = key_plane(shapes, plane);
        for row in 0..shapes.queries as usize {
            let at = (plane * shapes.queries as usize + row) * width;
            let normalizer = f64::from(statistics[plane * shapes.queries as usize + row]);
            let row_dot = (0..width)
                .map(|depth| f64::from(gradient[at + depth]) * f64::from(out[at + depth]))
                .sum::<f64>();
            for column in 0..shapes.keys as usize {
                let other = (source * shapes.keys as usize + column) * width;
                let weight = if masked(shapes, row, column) {
                    let batch = shapes.batch as usize;
                    (score(shapes, queries, keys, plane, source, row, column)
                        + bias_of(bias, plane / batch, plane % batch, row, column)
                        - normalizer)
                        .exp()
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
