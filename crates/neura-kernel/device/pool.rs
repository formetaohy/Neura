#[neura_compiler::module]
mod source {
    fn window_max(source: Value, task: Task, window: uvec4) -> u32 {
        let rows = i32(source.dims.z);
        let columns = i32(source.dims.w);
        let mut best = -3.4028235e38;
        let mut chosen = 0u32;
        for reach_row in stride(0u32, task.reach_rows, 1u32) {
            let row = i32(window.z) * i32(task.stride_rows) + i32(reach_row) - i32(task.pad_rows);
            if row < 0i32 || row >= rows {
                continue;
            }
            for reach_column in stride(0u32, task.reach_columns, 1u32) {
                let column = i32(window.w) * i32(task.stride_columns) + i32(reach_column)
                    - i32(task.pad_columns);
                if column < 0i32 || column >= columns {
                    continue;
                }
                let inside = read_address(
                    uvec4(window.x, window.y, u32(row), u32(column)),
                    source.strides,
                );
                let value = fetch(source, inside);
                if value > best {
                    best = value;
                    chosen = inside;
                }
            }
        }
        return chosen;
    }

    fn run_pool_max2d(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            let chosen = window_max(source, task, at);
            publish(output, index, chained(task, at, fetch(source, chosen)));
        }
    }

    fn run_pool_mean2d(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let rows = i32(source.dims.z);
        let columns = i32(source.dims.w);
        let area = f32(task.reach_rows * task.reach_columns);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            let mut total = 0.0;
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = i32(at.z) * i32(task.stride_rows) + i32(reach_row) - i32(task.pad_rows);
                if row < 0i32 || row >= rows {
                    continue;
                }
                for reach_column in stride(0u32, task.reach_columns, 1u32) {
                    let column = i32(at.w) * i32(task.stride_columns) + i32(reach_column)
                        - i32(task.pad_columns);
                    if column < 0i32 || column >= columns {
                        continue;
                    }
                    total = total
                        + fetch(
                            source,
                            read_address(uvec4(at.x, at.y, u32(row), u32(column)), source.strides),
                        );
                }
            }
            publish(output, index, chained(task, at, total / area));
        }
    }

    fn run_pool_max2d_input_grad(task: Task, lid: u32) {
        let source = values[task.a];
        let gradient = values[task.b];
        let output = values[task.out];
        let gradient_rows = i32(gradient.dims.z);
        let gradient_columns = i32(gradient.dims.w);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            let own = read_address(at, source.strides);
            let mut total = 0.0;
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
                    if shifted_column < 0i32 || shifted_column % i32(task.stride_columns) != 0i32 {
                        continue;
                    }
                    let column = shifted_column / i32(task.stride_columns);
                    if column >= gradient_columns {
                        continue;
                    }
                    let window = uvec4(at.x, at.y, u32(row), u32(column));
                    if window_max(source, task, window) == own {
                        total = total + fetch(gradient, read_address(window, gradient.strides));
                    }
                }
            }
            publish(output, index, chained(task, at, total));
        }
    }

    fn run_pool_mean2d_input_grad(task: Task, lid: u32) {
        let gradient = values[task.b];
        let output = values[task.out];
        let gradient_rows = i32(gradient.dims.z);
        let gradient_columns = i32(gradient.dims.w);
        let area = f32(task.reach_rows * task.reach_columns);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            let mut total = 0.0;
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
                    if shifted_column < 0i32 || shifted_column % i32(task.stride_columns) != 0i32 {
                        continue;
                    }
                    let column = shifted_column / i32(task.stride_columns);
                    if column >= gradient_columns {
                        continue;
                    }
                    total = total
                        + fetch(
                            gradient,
                            read_address(
                                uvec4(at.x, at.y, u32(row), u32(column)),
                                gradient.strides,
                            ),
                        );
                }
            }
            publish(output, index, chained(task, at, total / area));
        }
    }

    fn run_pool2d(task: Task, lid: u32) {
        match task.kind {
            kind::POOL_MAX2D => run_pool_max2d(task, lid),
            kind::POOL_MEAN2D => run_pool_mean2d(task, lid),
            _ => refuse(task.kind, 0u32),
        }
    }

    fn run_pool2d_input_grad(task: Task, lid: u32) {
        match task.kind {
            kind::POOL_MAX2D_INPUT_GRAD => run_pool_max2d_input_grad(task, lid),
            kind::POOL_MEAN2D_INPUT_GRAD => run_pool_mean2d_input_grad(task, lid),
            _ => refuse(task.kind, 0u32),
        }
    }
}
