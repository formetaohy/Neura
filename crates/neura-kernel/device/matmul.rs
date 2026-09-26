#[neura_compiler::module]
mod source {
    fn template_matmul_load(
        lid: u32,
        base_row: u32,
        base_column: u32,
        base_depth: u32,
        buffer: u32,
        left_plane: u32,
        right_plane: u32,
        left: Value,
        right: Value,
        rows: u32,
        depth: u32,
        columns: u32,
    ) {
        let left_slot = buffer * MATMUL_ROWS * MATMUL_DEPTH;
        for unit in stride(lid, MATMUL_ROWS * MATMUL_DEPTH, WORKGROUP_SIZE) {
            let row = base_row + unit / MATMUL_DEPTH;
            let column = base_depth + unit % MATMUL_DEPTH;
            let inside = row < rows && column < depth;
            let address = select(
                0u32,
                left_plane + row * left.strides.z + column * left.strides.w,
                inside,
            );
            matmul_left[left_slot + unit] = select(0.0, fetch(left, address), inside);
        }
        let right_slot = buffer * MATMUL_DEPTH * MATMUL_COLUMNS;
        for unit in stride(lid, MATMUL_DEPTH * MATMUL_COLUMNS, WORKGROUP_SIZE) {
            let row = base_depth + unit / MATMUL_COLUMNS;
            let column = base_column + unit % MATMUL_COLUMNS;
            let inside = row < depth && column < columns;
            let address = select(
                0u32,
                right_plane + row * right.strides.z + column * right.strides.w,
                inside,
            );
            matmul_right[right_slot + unit] = select(0.0, fetch(right, address), inside);
        }
    }

    fn template_run_matmul(task: Task, lid: u32) {
        let left = values[task.a];
        let right = values[task.b];
        let output = values[task.out];
        let rows = left.dims.z;
        let depth = left.dims.w;
        let columns = right.dims.w;
        let plane_columns = max(left.dims.y, right.dims.y);
        let planes = max(left.dims.x, right.dims.x) * plane_columns;
        let row_blocks = (rows + MATMUL_ROWS - 1u32) / MATMUL_ROWS;
        let column_blocks = (columns + MATMUL_COLUMNS - 1u32) / MATMUL_COLUMNS;
        let depth_blocks = (depth + MATMUL_DEPTH - 1u32) / MATMUL_DEPTH;
        let tiles_per_plane = row_blocks * column_blocks;
        let first_block = (task.slot * depth_blocks) / task.splits;
        let last_block = ((task.slot + 1u32) * depth_blocks) / task.splits;
        let thread_row = (lid / MATMUL_THREAD_COLUMNS) * MATMUL_REGISTER_ROWS;
        let thread_column = (lid % MATMUL_THREAD_COLUMNS) * MATMUL_REGISTER_COLUMNS;
        for tile in stride(task.first, task.first + task.count, 1u32) {
            let plane = tile / tiles_per_plane;
            let plane_row = plane / plane_columns;
            let plane_column = plane % plane_columns;
            let within = tile % tiles_per_plane;
            let base_row = (within / column_blocks) * MATMUL_ROWS;
            let base_column = (within % column_blocks) * MATMUL_COLUMNS;
            let left_plane = plane_row * left.strides.x + plane_column * left.strides.y;
            let right_plane = plane_row * right.strides.x + plane_column * right.strides.y;
            let out_plane = (task.slot * planes + plane) * rows * columns;
            let mut acc = scalar_array(0.0, MATMUL_REGISTERS);
            let mut buffer = 0u32;
            template_matmul_load(
                lid,
                base_row,
                base_column,
                first_block * MATMUL_DEPTH,
                0u32,
                left_plane,
                right_plane,
                left,
                right,
                rows,
                depth,
                columns,
            );
            workgroup_barrier();
            for block in stride(first_block, last_block, 1u32) {
                if block + 1u32 < last_block {
                    template_matmul_load(
                        lid,
                        base_row,
                        base_column,
                        (block + 1u32) * MATMUL_DEPTH,
                        1u32 - buffer,
                        left_plane,
                        right_plane,
                        left,
                        right,
                        rows,
                        depth,
                        columns,
                    );
                }
                let left_slot = buffer * MATMUL_ROWS * MATMUL_DEPTH;
                let right_slot = buffer * MATMUL_DEPTH * MATMUL_COLUMNS;
                for step in stride(0u32, MATMUL_DEPTH, 1u32) {
                    let mut left_registers = scalar_array(0.0, MATMUL_REGISTER_ROWS);
                    let mut right_registers = scalar_array(0.0, MATMUL_REGISTER_COLUMNS);
                    for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                        left_registers[row] =
                            matmul_left[left_slot + (thread_row + row) * MATMUL_DEPTH + step];
                    }
                    for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                        right_registers[column] = matmul_right
                            [right_slot + step * MATMUL_COLUMNS + thread_column + column];
                    }
                    for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                        for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                            let register = row * MATMUL_REGISTER_COLUMNS + column;
                            acc[register] =
                                acc[register] + left_registers[row] * right_registers[column];
                        }
                    }
                }
                workgroup_barrier();
                buffer = 1u32 - buffer;
            }
            for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                    let out_row = base_row + thread_row + row;
                    let out_column = base_column + thread_column + column;
                    if out_row < rows && out_column < columns {
                        let index = out_row * columns + out_column;
                        publish(
                            output,
                            out_plane + index,
                            chained(
                                task,
                                uvec4(plane_row, plane_column, out_row, out_column),
                                acc[row * MATMUL_REGISTER_COLUMNS + column],
                            ),
                        );
                    }
                }
            }
        }
    }

    fn run_matmul_fold(task: Task, lid: u32) {
        let partials = values[task.a];
        let output = values[task.out];
        let elements = output.dims.x * output.dims.y * output.dims.z * output.dims.w;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let mut total = 0.0;
            for split in stride(0u32, task.splits, 1u32) {
                total = total + fetch(partials, split * elements + index);
            }
            publish(
                output,
                index,
                chained(task, coordinates(index, output.dims), total),
            );
        }
    }

    fn run_matmul(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::MATMUL, refusal::GEOMETRY, task.geometry),
        }
    }
}
