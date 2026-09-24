fn run_conv2d(task: Task, lid: u32, slot: u32) {
    let input = values[task.a];
    let taps = values[task.b];
    let output = values[task.out];
    let channels = taps.dims.y;
    let reach_rows = taps.dims.z;
    let reach_columns = taps.dims.w;
    let input_rows = i32(input.dims.z);
    let input_columns = i32(input.dims.w);
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, output.dims);
        var total = 0.0;
        for (var channel = 0u; channel < channels; channel = channel + 1u) {
            for (var reach_row = 0u; reach_row < reach_rows; reach_row = reach_row + 1u) {
                let row = i32(at.z) * i32(task.stride_rows) + i32(reach_row) - i32(task.pad_rows);
                if (row < 0 || row >= input_rows) { continue; }
                for (var reach_column = 0u; reach_column < reach_columns; reach_column = reach_column + 1u) {
                    let column = i32(at.w) * i32(task.stride_columns) + i32(reach_column) - i32(task.pad_columns);
                    if (column < 0 || column >= input_columns) { continue; }
                    let source = read_address(vec4<u32>(at.x, channel, u32(row), u32(column)), input.strides);
                    let weight = read_address(vec4<u32>(at.y, channel, reach_row, reach_column), taps.strides);
                    total = total + fetch(input, source) * fetch(taps, weight);
                }
            }
        }
        publish(output, index, chained(task, at, total, slot));
    }
}

fn run_conv2d_input_grad(task: Task, lid: u32, slot: u32) {
    let taps = values[task.a];
    let gradient = values[task.b];
    let output = values[task.out];
    let output_channels = taps.dims.x;
    let reach_rows = taps.dims.z;
    let reach_columns = taps.dims.w;
    let gradient_rows = i32(gradient.dims.z);
    let gradient_columns = i32(gradient.dims.w);
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, output.dims);
        var total = 0.0;
        for (var channel = 0u; channel < output_channels; channel = channel + 1u) {
            for (var reach_row = 0u; reach_row < reach_rows; reach_row = reach_row + 1u) {
                let shifted_row = i32(at.z) + i32(task.pad_rows) - i32(reach_row);
                if (shifted_row < 0 || shifted_row % i32(task.stride_rows) != 0) { continue; }
                let row = shifted_row / i32(task.stride_rows);
                if (row >= gradient_rows) { continue; }
                for (var reach_column = 0u; reach_column < reach_columns; reach_column = reach_column + 1u) {
                    let shifted_column = i32(at.w) + i32(task.pad_columns) - i32(reach_column);
                    if (shifted_column < 0 || shifted_column % i32(task.stride_columns) != 0) { continue; }
                    let column = shifted_column / i32(task.stride_columns);
                    if (column >= gradient_columns) { continue; }
                    let source = read_address(vec4<u32>(at.x, channel, u32(row), u32(column)), gradient.strides);
                    let weight = read_address(vec4<u32>(channel, at.y, reach_row, reach_column), taps.strides);
                    total = total + fetch(gradient, source) * fetch(taps, weight);
                }
            }
        }
        publish(output, index, chained(task, at, total, slot));
    }
}

fn run_conv2d_weight_chunk(task: Task, lid: u32, slot: u32) {
    let input = values[task.a];
    let gradient = values[task.b];
    let taps = values[task.c];
    let output = values[task.out];
    let chunks = output.dims.z;
    let input_channels = input.dims.y;
    let input_rows = i32(input.dims.z);
    let input_columns = i32(input.dims.w);
    let gradient_rows = gradient.dims.z;
    let gradient_columns = gradient.dims.w;
    let reach_rows = taps.dims.z;
    let reach_columns = taps.dims.w;
    let positions = input.dims.x * gradient_rows * gradient_columns;
    let plane = gradient_rows * gradient_columns;
    let limit = (positions + chunks - 1u) / chunks;
    let first = task.slot * limit;
    let last = min(first + limit, positions);
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let reach_row = index / reach_columns % reach_rows;
        let reach_column = index % reach_columns;
        let channel = index / (reach_columns * reach_rows) % input_channels;
        let out_channel = index / (reach_columns * reach_rows * input_channels);
        var total = 0.0;
        for (var position = first; position < last; position = position + 1u) {
            let batch = position / plane;
            let within = position % plane;
            let row = within / gradient_columns;
            let column = within % gradient_columns;
            let used_row = i32(row) * i32(task.stride_rows) + i32(reach_row) - i32(task.pad_rows);
            if (used_row < 0 || used_row >= input_rows) { continue; }
            let used_column = i32(column) * i32(task.stride_columns) + i32(reach_column) - i32(task.pad_columns);
            if (used_column < 0 || used_column >= input_columns) { continue; }
            let source = read_address(vec4<u32>(batch, out_channel, row, column), gradient.strides);
            let weight = read_address(vec4<u32>(batch, channel, u32(used_row), u32(used_column)), input.strides);
            total = total + fetch(gradient, source) * fetch(input, weight);
        }
        publish(output, read_address(vec4<u32>(0u, 0u, task.slot, index), output.strides), total);
    }
}

fn run_conv2d_weight_fold(task: Task, lid: u32, slot: u32) {
    let partials = values[task.a];
    let output = values[task.out];
    let chunks = partials.dims.z;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        var total = 0.0;
        for (var chunk = 0u; chunk < chunks; chunk = chunk + 1u) {
            total = total + fetch(partials, read_address(vec4<u32>(0u, 0u, chunk, index), partials.strides));
        }
        publish(output, index, chained(task, coordinates(index, output.dims), total, slot));
    }
}

fn run_conv2d_weight_grad(task: Task, lid: u32, slot: u32) {
    switch (task.geometry) {
        case WEIGHT_CHUNK: { run_conv2d_weight_chunk(task, lid, slot); }
        case WEIGHT_FOLD: { run_conv2d_weight_fold(task, lid, slot); }
        default: { refuse(slot, Conv2dWeightGrad, task.geometry); }
    }
}
