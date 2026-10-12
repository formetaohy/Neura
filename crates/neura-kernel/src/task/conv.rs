use neura_compiler::Compiler;

const COLUMNS: u32 = 4;
const SPAN: u32 = 4;
const WINDOW: u32 = COLUMNS + SPAN - 1;

pub(crate) fn install(compiler: &mut Compiler) {
    compiler.constant("CONV_COLUMNS", COLUMNS);
    compiler.constant("CONV_SPAN", SPAN);
    compiler.constant("CONV_WINDOW", WINDOW);
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn run_conv2d(task: Task, lid: u32) {
        let input = values[task.a];
        let taps = values[task.b];
        let output = values[task.out];
        let channels = taps.dims.y;
        let out_channels = taps.dims.x / (input.dims.y / channels);
        let input_rows = i32(input.dims.z);
        let input_columns = i32(input.dims.w);
        if task.stride_columns == 1u32
            && task.dilation_columns == 1u32
            && task.reach_columns >= 2u32
            && task.reach_columns <= CONV_SPAN
        {
            for group in stride(
                task.first + lid * CONV_COLUMNS,
                task.first + task.count,
                WORKGROUP_SIZE * CONV_COLUMNS,
            ) {
                let at = coordinates(group, output.dims);
                if at.w + CONV_COLUMNS <= output.dims.w
                    && group + CONV_COLUMNS <= task.first + task.count
                {
                    conv2d_columns(
                        task,
                        at,
                        group,
                        input,
                        taps,
                        output,
                        channels,
                        out_channels,
                        input_rows,
                        input_columns,
                    );
                } else {
                    for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
                        let index = group + offset;
                        if index < task.first + task.count {
                            conv2d_element(
                                task,
                                index,
                                input,
                                taps,
                                output,
                                channels,
                                out_channels,
                                input_rows,
                                input_columns,
                            );
                        }
                    }
                }
            }
            return;
        }
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            conv2d_element(
                task,
                index,
                input,
                taps,
                output,
                channels,
                out_channels,
                input_rows,
                input_columns,
            );
        }
    }

    fn conv2d_columns(
        task: Task,
        at: uvec4,
        group: u32,
        input: Value,
        taps: Value,
        output: Value,
        channels: u32,
        out_channels: u32,
        input_rows: i32,
        input_columns: i32,
    ) {
        let first_channel = (at.y / out_channels) * channels;
        let plane = at.x * input.strides.x + first_channel * input.strides.y;
        let mut totals = scalar_array(0.0, CONV_COLUMNS);
        for channel in stride(0u32, channels, 1u32) {
            let planes = plane + channel * input.strides.y;
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = i32(at.z) * i32(task.stride_rows)
                    + i32(reach_row) * i32(task.dilation_rows)
                    - i32(task.pad_rows);
                if row < 0i32 || row >= input_rows {
                    continue;
                }
                let left = i32(at.w) - i32(task.pad_columns);
                let mut window = scalar_array(0.0, CONV_WINDOW);
                for slot in unroll(0u32, CONV_WINDOW, 1u32) {
                    let column = left + i32(slot);
                    let inside = column >= 0i32 && column < input_columns;
                    let address = select(
                        0u32,
                        planes + u32(row) * input.strides.z + u32(column) * input.strides.w,
                        inside,
                    );
                    window[slot] = select(0.0, fetch(input, address), inside);
                }
                let mut weight = read_address(uvec4(at.y, channel, reach_row, 0u32), taps.strides);
                for reach_column in unroll(0u32, CONV_SPAN, 1u32) {
                    if reach_column < task.reach_columns {
                        let weighed = fetch(taps, weight);
                        for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
                            totals[offset] =
                                totals[offset] + window[offset + reach_column] * weighed;
                        }
                        weight = weight + taps.strides.w;
                    }
                }
            }
        }
        for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
            publish(
                output,
                group + offset,
                chained(task, uvec4(at.x, at.y, at.z, at.w + offset), totals[offset]),
            );
        }
    }

    fn conv2d_element(
        task: Task,
        index: u32,
        input: Value,
        taps: Value,
        output: Value,
        channels: u32,
        out_channels: u32,
        input_rows: i32,
        input_columns: i32,
    ) {
        let at = coordinates(index, output.dims);
        let first_channel = (at.y / out_channels) * channels;
        let mut total = 0.0;
        for channel in stride(0u32, channels, 1u32) {
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = i32(at.z) * i32(task.stride_rows)
                    + i32(reach_row) * i32(task.dilation_rows)
                    - i32(task.pad_rows);
                if row < 0i32 || row >= input_rows {
                    continue;
                }
                let mut weight = read_address(uvec4(at.y, channel, reach_row, 0u32), taps.strides);
                let mut source = read_address(
                    uvec4(at.x, first_channel + channel, u32(row), 0u32),
                    input.strides,
                );
                for reach_column in stride(0u32, task.reach_columns, 1u32) {
                    let column = i32(at.w) * i32(task.stride_columns)
                        + i32(reach_column) * i32(task.dilation_columns)
                        - i32(task.pad_columns);
                    if column >= 0i32 && column < input_columns {
                        total = total
                            + fetch(input, source + u32(column) * input.strides.w)
                                * fetch(taps, weight);
                    }
                    weight = weight + taps.strides.w;
                }
            }
        }
        publish(output, index, chained(task, at, total));
    }

    fn run_conv2d_input_grad(task: Task, lid: u32) {
        match task.geometry {
            strategy::INPUT_FOLD => run_conv2d_input_fold(task, lid),
            strategy::INPUT_CHUNK => run_conv2d_input_chunk(task, lid),
            _ => refuse(kind::CONV2D_INPUT_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }

    fn window_at(shifted: i32, stride: u32, bound: i32) -> i32 {
        if shifted < 0i32 {
            return -1i32;
        }
        if stride == 1u32 {
            if shifted >= bound {
                return -1i32;
            }
            return shifted;
        }
        if shifted % i32(stride) != 0i32 {
            return -1i32;
        }
        let mapped = shifted / i32(stride);
        if mapped >= bound {
            return -1i32;
        }
        return mapped;
    }

    fn run_conv2d_transpose(task: Task, lid: u32) {
        match task.geometry {
            strategy::INPUT_FOLD => run_conv2d_input_fold(task, lid),
            strategy::INPUT_CHUNK => run_conv2d_input_chunk(task, lid),
            _ => refuse(kind::CONV2D_TRANSPOSE, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_conv2d_input_chunk(task: Task, lid: u32) {
        let filter = values[task.a];
        let gradient = values[task.b];
        let source = values[task.out];
        let channels = filter.dims.y;
        let out_channels = filter.dims.x / (source.dims.y / channels);
        let gradient_rows = i32(gradient.dims.z);
        let gradient_columns = i32(gradient.dims.w);
        let per_chunk = ceil_div(out_channels, task.splits);
        let first_block = task.slot * per_chunk;
        let last_block = min(first_block + per_chunk, out_channels);
        if task.stride_rows == 1u32
            && task.stride_columns == 1u32
            && task.dilation_columns == 1u32
            && task.reach_columns >= 2u32
            && task.reach_columns <= CONV_SPAN
        {
            for group in stride(
                task.first + lid * CONV_COLUMNS,
                task.first + task.count,
                WORKGROUP_SIZE * CONV_COLUMNS,
            ) {
                let at = coordinates(group, source.dims);
                if at.w + CONV_COLUMNS <= source.dims.w
                    && group + CONV_COLUMNS <= task.first + task.count
                {
                    conv2d_input_columns(
                        task,
                        at,
                        group,
                        filter,
                        gradient,
                        source,
                        channels,
                        out_channels,
                        first_block,
                        last_block,
                        gradient_rows,
                        gradient_columns,
                    );
                } else {
                    for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
                        let index = group + offset;
                        if index < task.first + task.count {
                            conv2d_input_element(
                                task,
                                index,
                                filter,
                                gradient,
                                source,
                                channels,
                                out_channels,
                                first_block,
                                last_block,
                                gradient_rows,
                                gradient_columns,
                            );
                        }
                    }
                }
            }
            return;
        }
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            conv2d_input_element(
                task,
                index,
                filter,
                gradient,
                source,
                channels,
                out_channels,
                first_block,
                last_block,
                gradient_rows,
                gradient_columns,
            );
        }
    }

    fn conv2d_input_columns(
        task: Task,
        at: uvec4,
        group: u32,
        filter: Value,
        gradient: Value,
        source: Value,
        channels: u32,
        out_channels: u32,
        first_block: u32,
        last_block: u32,
        gradient_rows: i32,
        gradient_columns: i32,
    ) {
        let first_channel = (at.y / channels) * out_channels;
        let local_channel = at.y % channels;
        let plane = at.x * gradient.strides.x;
        let left = i32(at.w) + i32(task.pad_columns) - i32(CONV_SPAN) + 1i32;
        let mut totals = scalar_array(0.0, CONV_COLUMNS);
        for channel in stride(
            first_channel + first_block,
            first_channel + last_block,
            1u32,
        ) {
            let planes = plane + channel * gradient.strides.y;
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = window_at(
                    i32(at.z) + i32(task.pad_rows) - i32(reach_row) * i32(task.dilation_rows),
                    task.stride_rows,
                    gradient_rows,
                );
                if row < 0i32 {
                    continue;
                }
                let mut window = scalar_array(0.0, CONV_WINDOW);
                for slot in unroll(0u32, CONV_WINDOW, 1u32) {
                    let column = left + i32(slot);
                    let inside = column >= 0i32 && column < gradient_columns;
                    let address = select(
                        0u32,
                        planes + u32(row) * gradient.strides.z + u32(column) * gradient.strides.w,
                        inside,
                    );
                    window[slot] = select(0.0, fetch(gradient, address), inside);
                }
                let mut weight = read_address(
                    uvec4(channel, local_channel, reach_row, 0u32),
                    filter.strides,
                );
                for reach_column in unroll(0u32, CONV_SPAN, 1u32) {
                    if reach_column < task.reach_columns {
                        let weighed = fetch(filter, weight);
                        for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
                            totals[offset] = totals[offset]
                                + window[offset + CONV_SPAN - 1u32 - reach_column] * weighed;
                        }
                        weight = weight + filter.strides.w;
                    }
                }
            }
        }
        for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
            conv2d_input_publish(
                task,
                group + offset,
                uvec4(at.x, at.y, at.z, at.w + offset),
                totals[offset],
                source,
            );
        }
    }

    fn conv2d_input_publish(task: Task, index: u32, at: uvec4, total: f32, source: Value) {
        if task.extra == NO_VALUE {
            publish(source, index, chained(task, at, total));
        } else {
            let partials = values[task.extra];
            publish(
                partials,
                read_address(uvec4(0u32, 0u32, task.slot, index), partials.strides),
                total,
            );
        }
    }

    fn conv2d_input_element(
        task: Task,
        index: u32,
        filter: Value,
        gradient: Value,
        source: Value,
        channels: u32,
        out_channels: u32,
        first_block: u32,
        last_block: u32,
        gradient_rows: i32,
        gradient_columns: i32,
    ) {
        let at = coordinates(index, source.dims);
        let first_channel = (at.y / channels) * out_channels;
        let local_channel = at.y % channels;
        let mut total = 0.0;
        for channel in stride(
            first_channel + first_block,
            first_channel + last_block,
            1u32,
        ) {
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = window_at(
                    i32(at.z) + i32(task.pad_rows) - i32(reach_row) * i32(task.dilation_rows),
                    task.stride_rows,
                    gradient_rows,
                );
                if row < 0i32 {
                    continue;
                }
                let mut weight = read_address(
                    uvec4(channel, local_channel, reach_row, 0u32),
                    filter.strides,
                );
                let mut source_at =
                    read_address(uvec4(at.x, channel, u32(row), 0u32), gradient.strides);
                for reach_column in stride(0u32, task.reach_columns, 1u32) {
                    let column = window_at(
                        i32(at.w) + i32(task.pad_columns)
                            - i32(reach_column) * i32(task.dilation_columns),
                        task.stride_columns,
                        gradient_columns,
                    );
                    if column >= 0i32 {
                        total = total
                            + fetch(gradient, source_at + u32(column) * gradient.strides.w)
                                * fetch(filter, weight);
                    }
                    weight = weight + filter.strides.w;
                }
            }
        }
        conv2d_input_publish(task, index, at, total, source);
    }

    fn run_conv2d_input_fold(task: Task, lid: u32) {
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
            let mut batch = first / plane;
            let mut within = first % plane;
            let mut row = within / gradient_columns;
            let mut column = within % gradient_columns;
            for position in stride(first, last, 1u32) {
                let used_row = i32(row) * i32(task.stride_rows)
                    + i32(reach_row) * i32(task.dilation_rows)
                    - i32(task.pad_rows);
                let used_column = i32(column) * i32(task.stride_columns)
                    + i32(reach_column) * i32(task.dilation_columns)
                    - i32(task.pad_columns);
                if used_row >= 0i32
                    && used_row < input_rows
                    && used_column >= 0i32
                    && used_column < input_columns
                {
                    let source =
                        read_address(uvec4(batch, out_channel, row, column), gradient.strides);
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
                column = column + 1u32;
                if column >= gradient_columns {
                    column = 0u32;
                    row = row + 1u32;
                    if row >= gradient_rows {
                        row = 0u32;
                        batch = batch + 1u32;
                    }
                }
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
