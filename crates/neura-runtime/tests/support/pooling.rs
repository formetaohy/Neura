pub fn pool2d_reference(
    input: &[f32],
    input_shape: [u32; 4],
    window: neura_graph::Window,
    max: bool,
) -> Vec<f32> {
    let [batch, channels, rows, columns] = input_shape;
    let reach_rows = window.reach_rows();
    let reach_columns = window.reach_columns();
    let output_rows = (rows + 2 * window.pad_rows() - reach_rows) / window.stride_rows() + 1;
    let output_columns =
        (columns + 2 * window.pad_columns() - reach_columns) / window.stride_columns() + 1;
    let mut out = vec![0.0f32; (batch * channels * output_rows * output_columns) as usize];
    for plane in 0..batch {
        for channel in 0..channels {
            for row in 0..output_rows {
                for column in 0..output_columns {
                    let mut total = 0.0f32;
                    let mut best = f32::NEG_INFINITY;
                    for reach_row in 0..reach_rows {
                        let used_row = row * window.stride_rows() + reach_row;
                        if used_row < window.pad_rows() {
                            continue;
                        }
                        let used_row = used_row - window.pad_rows();
                        if used_row >= rows {
                            continue;
                        }
                        for reach_column in 0..reach_columns {
                            let used_column = column * window.stride_columns() + reach_column;
                            if used_column < window.pad_columns() {
                                continue;
                            }
                            let used_column = used_column - window.pad_columns();
                            if used_column >= columns {
                                continue;
                            }
                            let source = (((plane * channels + channel) * rows + used_row)
                                * columns
                                + used_column) as usize;
                            total += input[source];
                            best = best.max(input[source]);
                        }
                    }
                    let at = (((plane * channels + channel) * output_rows + row) * output_columns
                        + column) as usize;
                    out[at] = if max {
                        best
                    } else {
                        total / (reach_rows * reach_columns) as f32
                    };
                }
            }
        }
    }
    out
}

pub fn max_pool2d_gradient(
    input: &[f32],
    upstream: &[f32],
    input_shape: [u32; 4],
    window: neura_graph::Window,
) -> Vec<f32> {
    let [batch, channels, rows, columns] = input_shape;
    let reach_rows = window.reach_rows();
    let reach_columns = window.reach_columns();
    let output_rows = (rows + 2 * window.pad_rows() - reach_rows) / window.stride_rows() + 1;
    let output_columns =
        (columns + 2 * window.pad_columns() - reach_columns) / window.stride_columns() + 1;
    let mut out = vec![0.0f32; input.len()];
    for plane in 0..batch {
        for channel in 0..channels {
            for row in 0..output_rows {
                for column in 0..output_columns {
                    let mut best = f32::NEG_INFINITY;
                    let mut chosen = (0u32, 0u32);
                    for reach_row in 0..reach_rows {
                        let used_row = row * window.stride_rows() + reach_row;
                        if used_row < window.pad_rows() {
                            continue;
                        }
                        let used_row = used_row - window.pad_rows();
                        if used_row >= rows {
                            continue;
                        }
                        for reach_column in 0..reach_columns {
                            let used_column = column * window.stride_columns() + reach_column;
                            if used_column < window.pad_columns() {
                                continue;
                            }
                            let used_column = used_column - window.pad_columns();
                            if used_column >= columns {
                                continue;
                            }
                            let value =
                                input[(((plane * channels + channel) * rows + used_row) * columns
                                    + used_column) as usize];
                            if value > best {
                                best = value;
                                chosen = (used_row, used_column);
                            }
                        }
                    }
                    let at = (((plane * channels + channel) * output_rows + row) * output_columns
                        + column) as usize;
                    out[(((plane * channels + channel) * rows + chosen.0) * columns + chosen.1)
                        as usize] += upstream[at];
                }
            }
        }
    }
    out
}
