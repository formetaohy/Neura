#[neura_compiler::module]
mod source {
    fn run_conv2d(task: Task, lid: u32) {
        let input = values[task.a];
        let taps = values[task.b];
        let output = values[task.out];
        let channels = taps.dims.y;
        let out_channels = taps.dims.x / (input.dims.y / channels);
        let input_rows = i32(input.dims.z);
        let input_columns = i32(input.dims.w);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            let first_channel = (at.y / out_channels) * channels;
            let mut total = 0.0;
            for channel in stride(0u32, channels, 1u32) {
                for reach_row in stride(0u32, task.reach_rows, 1u32) {
                    let row =
                        i32(at.z) * i32(task.stride_rows) + i32(reach_row) - i32(task.pad_rows);
                    if row < 0i32 || row >= input_rows {
                        continue;
                    }
                    for reach_column in stride(0u32, task.reach_columns, 1u32) {
                        let column = i32(at.w) * i32(task.stride_columns) + i32(reach_column)
                            - i32(task.pad_columns);
                        if column < 0i32 || column >= input_columns {
                            continue;
                        }
                        let source = read_address(
                            uvec4(at.x, first_channel + channel, u32(row), u32(column)),
                            input.strides,
                        );
                        let weight = read_address(
                            uvec4(at.y, channel, reach_row, reach_column),
                            taps.strides,
                        );
                        total = total + fetch(input, source) * fetch(taps, weight);
                    }
                }
            }
            publish(output, index, chained(task, at, total));
        }
    }

    fn run_conv2d_input_grad(task: Task, lid: u32) {
        let taps = values[task.a];
        let gradient = values[task.b];
        let output = values[task.out];
        let channels = taps.dims.y;
        let out_channels = taps.dims.x / (output.dims.y / channels);
        let gradient_rows = i32(gradient.dims.z);
        let gradient_columns = i32(gradient.dims.w);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            let first_channel = (at.y / channels) * out_channels;
            let local_channel = at.y % channels;
            let mut total = 0.0;
            for channel in stride(first_channel, first_channel + out_channels, 1u32) {
                for reach_row in stride(0u32, task.reach_rows, 1u32) {
                    let shifted_row = i32(at.z) + i32(task.pad_rows) - i32(reach_row);
                    if shifted_row < 0i32 || shifted_row % i32(task.stride_rows) != 0i32 {
                        continue;
                    }
                    let row = shifted_row / i32(task.stride_rows);
                    if row >= gradient_rows {
                        continue;
                    }
                    for reach_column in stride(0u32, task.reach_columns, 1u32) {
                        let shifted_column = i32(at.w) + i32(task.pad_columns) - i32(reach_column);
                        if shifted_column < 0i32
                            || shifted_column % i32(task.stride_columns) != 0i32
                        {
                            continue;
                        }
                        let column = shifted_column / i32(task.stride_columns);
                        if column >= gradient_columns {
                            continue;
                        }
                        let source = read_address(
                            uvec4(at.x, channel, u32(row), u32(column)),
                            gradient.strides,
                        );
                        let weight = read_address(
                            uvec4(channel, local_channel, reach_row, reach_column),
                            taps.strides,
                        );
                        total = total + fetch(gradient, source) * fetch(taps, weight);
                    }
                }
            }
            publish(output, index, chained(task, at, total));
        }
    }

    fn run_conv2d_weight_chunk(task: Task, lid: u32) {
        let input = values[task.a];
        let gradient = values[task.b];
        let taps = values[task.c];
        let output = values[task.out];
        let chunks = output.dims.z;
        let channels = taps.dims.y;
        let out_channels = taps.dims.x / (input.dims.y / channels);
        let input_rows = i32(input.dims.z);
        let input_columns = i32(input.dims.w);
        let gradient_rows = gradient.dims.z;
        let gradient_columns = gradient.dims.w;
        let reach_rows = taps.dims.z;
        let reach_columns = taps.dims.w;
        let positions = input.dims.x * gradient_rows * gradient_columns;
        let plane = gradient_rows * gradient_columns;
        let limit = (positions + chunks - 1u32) / chunks;
        let first = task.slot * limit;
        let last = min(first + limit, positions);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let reach_row = index / reach_columns % reach_rows;
            let reach_column = index % reach_columns;
            let channel = index / (reach_columns * reach_rows) % channels;
            let out_channel = index / (reach_columns * reach_rows * channels);
            let first_channel = (out_channel / out_channels) * channels;
            let mut total = 0.0;
            for position in stride(first, last, 1u32) {
                let batch = position / plane;
                let within = position % plane;
                let row = within / gradient_columns;
                let column = within % gradient_columns;
                let used_row =
                    i32(row) * i32(task.stride_rows) + i32(reach_row) - i32(task.pad_rows);
                if used_row < 0i32 || used_row >= input_rows {
                    continue;
                }
                let used_column = i32(column) * i32(task.stride_columns) + i32(reach_column)
                    - i32(task.pad_columns);
                if used_column < 0i32 || used_column >= input_columns {
                    continue;
                }
                let source = read_address(uvec4(batch, out_channel, row, column), gradient.strides);
                let weight = read_address(
                    uvec4(
                        batch,
                        first_channel + channel,
                        u32(used_row),
                        u32(used_column),
                    ),
                    input.strides,
                );
                total = total + fetch(gradient, source) * fetch(input, weight);
            }
            publish(
                output,
                read_address(uvec4(0u32, 0u32, task.slot, index), output.strides),
                total,
            );
        }
    }

    fn run_conv2d_weight_fold(task: Task, lid: u32) {
        let partials = values[task.a];
        let output = values[task.out];
        let chunks = partials.dims.z;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let mut total = 0.0;
            for chunk in stride(0u32, chunks, 1u32) {
                total = total
                    + fetch(
                        partials,
                        read_address(uvec4(0u32, 0u32, chunk, index), partials.strides),
                    );
            }
            publish(
                output,
                index,
                chained(task, coordinates(index, output.dims), total),
            );
        }
    }

    fn run_conv2d_weight_grad(task: Task, lid: u32) {
        match task.geometry {
            strategy::WEIGHT_CHUNK => run_conv2d_weight_chunk(task, lid),
            strategy::WEIGHT_FOLD => run_conv2d_weight_fold(task, lid),
            _ => refuse(kind::CONV2D_WEIGHT_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }
}
