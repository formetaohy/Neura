pub fn conv2d_reference(
    input: &[f32],
    taps: &[f32],
    input_shape: [u32; 4],
    output_channels: u32,
    window: neura_program::Window,
) -> Vec<f32> {
    let [batch, channels, rows, columns] = input_shape;
    let reach_rows = window.reach_rows();
    let reach_columns = window.reach_columns();
    let output_rows = (rows + 2 * window.pad_rows() - reach_rows) / window.stride_rows() + 1;
    let output_columns =
        (columns + 2 * window.pad_columns() - reach_columns) / window.stride_columns() + 1;
    let mut out = vec![0.0f32; (batch * output_channels * output_rows * output_columns) as usize];
    for plane in 0..batch {
        for channel in 0..output_channels {
            for row in 0..output_rows {
                for column in 0..output_columns {
                    let mut total = 0.0f32;
                    for source_channel in 0..channels {
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
                                let source =
                                    (((plane * channels + source_channel) * rows + used_row)
                                        * columns
                                        + used_column) as usize;
                                let weight = (((channel * channels + source_channel) * reach_rows
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
