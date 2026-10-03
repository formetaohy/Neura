#[neura_compiler::module]
mod source {
    fn template_matmul_cooperative_load(
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
            let row = unit / MATMUL_DEPTH;
            let column = unit % MATMUL_DEPTH;
            let at_row = base_row + row;
            let at_column = base_depth + column;
            let inside = at_row < rows && at_column < depth;
            let address = select(
                0u32,
                left_plane + at_row * left.strides.z + at_column * left.strides.w,
                inside,
            );
            scratch_half[left_slot + unit] = f16(select(0.0, fetch(left, address), inside));
        }
        let right_slot = buffer * MATMUL_COLUMNS * MATMUL_DEPTH;
        for unit in stride(lid, MATMUL_COLUMNS * MATMUL_DEPTH, WORKGROUP_SIZE) {
            let column = base_column + unit / MATMUL_DEPTH;
            let row = base_depth + unit % MATMUL_DEPTH;
            let inside = row < depth && column < columns;
            let address = select(
                0u32,
                right_plane + row * right.strides.z + column * right.strides.w,
                inside,
            );
            scratch_half[SCRATCH_COOPERATIVE_RIGHT + right_slot + unit] =
                f16(select(0.0, fetch(right, address), inside));
        }
    }

    fn template_run_matmul_cooperative(task: Task, lid: u32) {
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
        let subgroup = lid / COOPMAT_SUBGROUP;
        let subgroup_row = subgroup / MATMUL_SUBGROUP_COLUMNS;
        let subgroup_column = subgroup % MATMUL_SUBGROUP_COLUMNS;
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
            let mut accumulator = coopmat_accumulator(0.0);
            template_matmul_cooperative_load(
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
            let mut buffer = 0u32;
            for block in stride(first_block, last_block, 1u32) {
                if block + 1u32 < last_block {
                    template_matmul_cooperative_load(
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
                let left_slot = buffer * MATMUL_ROWS * MATMUL_DEPTH
                    + subgroup_row * COOPMAT_ROWS * MATMUL_DEPTH;
                let right_slot = SCRATCH_COOPERATIVE_RIGHT
                    + buffer * MATMUL_COLUMNS * MATMUL_DEPTH
                    + subgroup_column * COOPMAT_COLUMNS * MATMUL_DEPTH;
                let left_fragment = coopmat_load_row(&scratch_half[left_slot], MATMUL_DEPTH);
                let right_fragment = coopmat_load_column(&scratch_half[right_slot], MATMUL_DEPTH);
                accumulator = coopmat_muladd(left_fragment, right_fragment, accumulator);
                workgroup_barrier();
                buffer = 1u32 - buffer;
            }
            let copy_row =
                subgroup_row * COOPMAT_ROWS * MATMUL_COLUMNS + subgroup_column * COOPMAT_COLUMNS;
            coopmat_store(&scratch[copy_row], MATMUL_COLUMNS, accumulator);
            workgroup_barrier();
            for unit in stride(lid, MATMUL_ROWS * MATMUL_COLUMNS, WORKGROUP_SIZE) {
                let row = unit / MATMUL_COLUMNS;
                let column = unit % MATMUL_COLUMNS;
                let out_row = base_row + row;
                let out_column = base_column + column;
                if out_row < rows && out_column < columns {
                    publish(
                        output,
                        out_plane + out_row * columns + out_column,
                        chained(
                            task,
                            uvec4(plane_row, plane_column, out_row, out_column),
                            scratch[unit],
                        ),
                    );
                }
            }
            workgroup_barrier();
        }
    }
}
