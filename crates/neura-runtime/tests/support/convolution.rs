pub fn conv2d_reference(
    input: &[f32],
    taps: &[f32],
    input_shape: [u32; 4],
    output_channels: u32,
    window: neura_graph::Window,
) -> Vec<f32> {
    let [batch, channels, rows, columns] = input_shape;
    let reach_rows = window.reach_rows();
    let reach_columns = window.reach_columns();
    let per_group = taps.len() as u32 / (output_channels * reach_rows * reach_columns);
    let groups = channels / per_group;
    let per_group_outputs = output_channels / groups;
    let output_rows = (rows + 2 * window.pad_rows() - window.row_span()) / window.stride_rows() + 1;
    let output_columns =
        (columns + 2 * window.pad_columns() - window.column_span()) / window.stride_columns() + 1;
    let mut out = vec![0.0f32; (batch * output_channels * output_rows * output_columns) as usize];
    for plane in 0..batch {
        for channel in 0..output_channels {
            let group = channel / per_group_outputs;
            for row in 0..output_rows {
                for column in 0..output_columns {
                    let mut total = 0.0f32;
                    for local in 0..per_group {
                        let source_channel = group * per_group + local;
                        for reach_row in 0..reach_rows {
                            let used_row =
                                row * window.stride_rows() + reach_row * window.dilation_rows();
                            if used_row < window.pad_rows() {
                                continue;
                            }
                            let used_row = used_row - window.pad_rows();
                            if used_row >= rows {
                                continue;
                            }
                            for reach_column in 0..reach_columns {
                                let used_column = column * window.stride_columns()
                                    + reach_column * window.dilation_columns();
                                if used_column < window.pad_columns() {
                                    continue;
                                }
                                let used_column = used_column - window.pad_columns();
                                if used_column >= columns {
                                    continue;
                                }
                                let source =
                                    (((plane * channels + source_channel) * rows + used_row)
                                        * columns
                                        + used_column) as usize;
                                let weight = (((channel * per_group + local) * reach_rows
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

pub fn conv2d_input_reference(
    taps: &[f32],
    slope: &[f32],
    input_shape: [u32; 4],
    output_channels: u32,
    groups: u32,
    window: neura_graph::Window,
) -> Vec<f32> {
    let [batch, channels, rows, columns] = input_shape;
    let local = channels / groups;
    let per_group_outputs = output_channels / groups;
    let reach_rows = window.reach_rows();
    let reach_columns = window.reach_columns();
    let output_rows = (rows + 2 * window.pad_rows() - window.row_span()) / window.stride_rows() + 1;
    let output_columns =
        (columns + 2 * window.pad_columns() - window.column_span()) / window.stride_columns() + 1;
    let mut out = vec![0.0f32; (batch * channels * rows * columns) as usize];
    for plane in 0..batch {
        for channel in 0..channels {
            let group = channel / local;
            let local_channel = channel % local;
            for row in 0..rows {
                for column in 0..columns {
                    let mut total = 0.0f32;
                    for output in 0..per_group_outputs {
                        let source_channel = group * per_group_outputs + output;
                        for reach_row in 0..reach_rows {
                            let used_row = row as i32 + window.pad_rows() as i32
                                - (reach_row * window.dilation_rows()) as i32;
                            if used_row < 0
                                || !(used_row as u32).is_multiple_of(window.stride_rows())
                            {
                                continue;
                            }
                            let gradient_row = used_row as u32 / window.stride_rows();
                            if gradient_row >= output_rows {
                                continue;
                            }
                            for reach_column in 0..reach_columns {
                                let used_column = column as i32 + window.pad_columns() as i32
                                    - (reach_column * window.dilation_columns()) as i32;
                                if used_column < 0
                                    || !(used_column as u32).is_multiple_of(window.stride_columns())
                                {
                                    continue;
                                }
                                let gradient_column = used_column as u32 / window.stride_columns();
                                if gradient_column >= output_columns {
                                    continue;
                                }
                                let source = (((plane * output_channels + source_channel)
                                    * output_rows
                                    + gradient_row)
                                    * output_columns
                                    + gradient_column)
                                    as usize;
                                let weight = (((source_channel * local + local_channel)
                                    * reach_rows
                                    + reach_row)
                                    * reach_columns
                                    + reach_column)
                                    as usize;
                                total += slope[source] * taps[weight];
                            }
                        }
                    }
                    out[(((plane * channels + channel) * rows + row) * columns + column)
                        as usize] = total;
                }
            }
        }
    }
    out
}

pub fn conv2d_weight_reference(
    input: &[f32],
    slope: &[f32],
    input_shape: [u32; 4],
    output_channels: u32,
    groups: u32,
    window: neura_graph::Window,
) -> Vec<f32> {
    let [batch, channels, rows, columns] = input_shape;
    let local = channels / groups;
    let per_group_outputs = output_channels / groups;
    let reach_rows = window.reach_rows();
    let reach_columns = window.reach_columns();
    let output_rows = (rows + 2 * window.pad_rows() - window.row_span()) / window.stride_rows() + 1;
    let output_columns =
        (columns + 2 * window.pad_columns() - window.column_span()) / window.stride_columns() + 1;
    let mut out = vec![0.0f32; (output_channels * local * reach_rows * reach_columns) as usize];
    for channel in 0..output_channels {
        let group = channel / per_group_outputs;
        for local_channel in 0..local {
            let source_channel = group * local + local_channel;
            for reach_row in 0..reach_rows {
                for reach_column in 0..reach_columns {
                    let mut total = 0.0f32;
                    for plane in 0..batch {
                        for row in 0..output_rows {
                            let used_row =
                                row * window.stride_rows() + reach_row * window.dilation_rows();
                            if used_row < window.pad_rows() {
                                continue;
                            }
                            let used_row = used_row - window.pad_rows();
                            if used_row >= rows {
                                continue;
                            }
                            for column in 0..output_columns {
                                let used_column = column * window.stride_columns()
                                    + reach_column * window.dilation_columns();
                                if used_column < window.pad_columns() {
                                    continue;
                                }
                                let used_column = used_column - window.pad_columns();
                                if used_column >= columns {
                                    continue;
                                }
                                let source =
                                    (((plane * channels + source_channel) * rows + used_row)
                                        * columns
                                        + used_column) as usize;
                                let at = (((plane * output_channels + channel) * output_rows + row)
                                    * output_columns
                                    + column) as usize;
                                total += input[source] * slope[at];
                            }
                        }
                    }
                    let at = (((channel * local + local_channel) * reach_rows + reach_row)
                        * reach_columns
                        + reach_column) as usize;
                    out[at] = total;
                }
            }
        }
    }
    out
}
