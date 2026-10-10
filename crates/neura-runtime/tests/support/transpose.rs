pub fn conv2d_transpose_reference(
    input: &[f32],
    taps: &[f32],
    input_shape: [u32; 4],
    groups: u32,
    window: neura_graph::Window,
) -> Vec<f32> {
    let [batch, input_channels, rows, columns] = input_shape;
    let reach_rows = window.reach_rows();
    let reach_columns = window.reach_columns();
    let local_outputs = taps.len() as u32 / (input_channels * reach_rows * reach_columns);
    let output_channels = groups * local_outputs;
    let inputs_per_group = input_channels / groups;
    let output_rows = (rows - 1) * window.stride_rows() + reach_rows - 2 * window.pad_rows();
    let output_columns =
        (columns - 1) * window.stride_columns() + reach_columns - 2 * window.pad_columns();
    let mut out = vec![0.0f32; (batch * output_channels * output_rows * output_columns) as usize];
    for plane in 0..batch {
        for channel in 0..output_channels {
            let group = channel / local_outputs;
            let local = channel % local_outputs;
            for row in 0..output_rows {
                for column in 0..output_columns {
                    let mut total = 0.0f32;
                    for input_channel in 0..inputs_per_group {
                        let source_channel = group * inputs_per_group + input_channel;
                        for reach_row in 0..reach_rows {
                            let Some(used_row) = scattered(
                                row,
                                reach_row,
                                window.pad_rows(),
                                window.stride_rows(),
                                rows,
                            ) else {
                                continue;
                            };
                            for reach_column in 0..reach_columns {
                                let Some(used_column) = scattered(
                                    column,
                                    reach_column,
                                    window.pad_columns(),
                                    window.stride_columns(),
                                    columns,
                                ) else {
                                    continue;
                                };
                                let source =
                                    (((plane * input_channels + source_channel) * rows + used_row)
                                        * columns
                                        + used_column) as usize;
                                let weight = (((source_channel * local_outputs + local)
                                    * reach_rows
                                    + reach_row)
                                    * reach_columns
                                    + reach_column)
                                    as usize;
                                total += input[source] * taps[weight];
                            }
                        }
                    }
                    let at = (((plane * output_channels + channel) * output_rows + row)
                        * output_columns
                        + column) as usize;
                    out[at] = total;
                }
            }
        }
    }
    out
}

fn scattered(position: u32, tap: u32, padding: u32, stride: u32, bound: u32) -> Option<u32> {
    let reached = position + padding;
    if reached < tap {
        return None;
    }
    let shifted = reached - tap;
    let used = shifted / stride;
    (shifted.is_multiple_of(stride) && used < bound).then_some(used)
}

pub fn conv2d_transpose_weight_grad_reference(
    input: &[f32],
    input_shape: [u32; 4],
    gradient: &[f32],
    gradient_shape: [u32; 4],
    groups: u32,
    window: neura_graph::Window,
) -> Vec<f32> {
    let [batch, input_channels, rows, columns] = input_shape;
    let [planes, output_channels, gradient_rows, gradient_columns] = gradient_shape;
    assert_eq!(
        batch, planes,
        "a gradient of the transposed convolution walks the planes of its input",
    );
    let reach_rows = window.reach_rows();
    let reach_columns = window.reach_columns();
    let local_outputs = output_channels / groups;
    let outputs_per_group = input_channels / groups;
    let mut out =
        vec![0.0f32; (input_channels * local_outputs * reach_rows * reach_columns) as usize];
    for plane in 0..batch {
        for channel in 0..input_channels {
            let group = channel / outputs_per_group;
            for row in 0..rows {
                for column in 0..columns {
                    let source = (((plane * input_channels + channel) * rows + row) * columns
                        + column) as usize;
                    for local in 0..local_outputs {
                        let gradient_channel = group * local_outputs + local;
                        for reach_row in 0..reach_rows {
                            let used_row = row * window.stride_rows() + reach_row;
                            if used_row < window.pad_rows() {
                                continue;
                            }
                            let used_row = used_row - window.pad_rows();
                            if used_row >= gradient_rows {
                                continue;
                            }
                            for reach_column in 0..reach_columns {
                                let used_column = column * window.stride_columns() + reach_column;
                                if used_column < window.pad_columns() {
                                    continue;
                                }
                                let used_column = used_column - window.pad_columns();
                                if used_column >= gradient_columns {
                                    continue;
                                }
                                let weight = (((channel * local_outputs + local) * reach_rows
                                    + reach_row)
                                    * reach_columns
                                    + reach_column)
                                    as usize;
                                let at =
                                    (((plane * output_channels + gradient_channel) * gradient_rows
                                        + used_row)
                                        * gradient_columns
                                        + used_column) as usize;
                                out[weight] += input[source] * gradient[at];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
