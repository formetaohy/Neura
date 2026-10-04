use neura_compiler::{Compiler, ast};
use neura_profile::{Geometry, MatmulStrategy};

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

pub(crate) fn install_tiles(compiler: &mut Compiler, geometry: &Geometry) {
    compiler.constant("SCRATCH_MATMUL_RIGHT", geometry.matmul_right());
    if let Some(cooperative) = geometry.cooperative() {
        let fragment = cooperative.fragment();
        compiler.constant("COOPMAT_ROWS", fragment.rows());
        compiler.constant("COOPMAT_COLUMNS", fragment.columns());
        compiler.constant("COOPMAT_DEPTH", fragment.depth());
    }
    cooperative::define(compiler);
    specialize(compiler, geometry);
}

fn specialize(compiler: &mut Compiler, geometry: &Geometry) {
    for (index, tile) in geometry.walked() {
        let suffix = index.to_string();
        let mut constants = vec![
            ("MATMUL_ROWS", tile.rows()),
            ("MATMUL_COLUMNS", tile.columns()),
            ("MATMUL_DEPTH", tile.depth()),
        ];
        match tile.strategy() {
            MatmulStrategy::Staged => {
                constants.extend([
                    ("MATMUL_LEFT_STRIDE", tile.left_stride()),
                    ("MATMUL_THREAD_COLUMNS", tile.thread_columns()),
                    ("MATMUL_REGISTER_ROWS", tile.register_rows()),
                    ("MATMUL_REGISTER_COLUMNS", tile.register_columns()),
                    ("MATMUL_REGISTERS", tile.registers()),
                ]);
                compiler.specialize(
                    "template_matmul_load",
                    &format!("matmul_load_{suffix}"),
                    &constants,
                );
                compiler.specialize(
                    "template_run_matmul",
                    &format!("run_matmul_{suffix}"),
                    &constants,
                );
            }
            MatmulStrategy::Streamed => {
                constants.extend([
                    ("MATMUL_LEFT_STRIDE", tile.depth() + 1),
                    ("MATMUL_THREAD_COLUMNS", tile.thread_columns()),
                    ("MATMUL_REGISTER_ROWS", tile.register_rows()),
                    ("MATMUL_REGISTER_COLUMNS", tile.register_columns()),
                    ("MATMUL_REGISTERS", tile.registers()),
                ]);
                compiler.specialize(
                    "template_run_matmul_stream",
                    &format!("run_matmul_{suffix}"),
                    &constants,
                );
            }
            MatmulStrategy::Cooperative => {
                constants.extend([
                    ("MATMUL_LEFT_STRIDE", tile.left_stride()),
                    ("MATMUL_STAGE_DEPTH", tile.depth()),
                    ("MATMUL_FRAGMENT_ROWS", tile.fragment_grid().0),
                    ("MATMUL_FRAGMENT_COLUMNS", tile.fragment_grid().1),
                    ("MATMUL_ACCUMULATORS", tile.accumulators()),
                    ("COOPMAT_ROWS", tile.fragment_rows()),
                    ("COOPMAT_COLUMNS", tile.fragment_columns()),
                    ("COOPMAT_DEPTH", tile.fragment_depth()),
                    ("COOPMAT_SUBGROUP", tile.subgroup()),
                    ("MATMUL_SUBGROUP_ROWS", tile.subgroup_rows()),
                    ("MATMUL_SUBGROUP_COLUMNS", tile.subgroup_columns()),
                    (
                        "SCRATCH_COOPERATIVE_RIGHT",
                        u32::try_from(2 * tile.rows() as u64 * tile.depth() as u64)
                            .expect("a cooperative panel fits one device word address"),
                    ),
                ]);
                compiler.specialize(
                    "template_matmul_cooperative_load",
                    &format!("matmul_cooperative_load_{suffix}"),
                    &constants,
                );
                compiler.specialize(
                    "template_run_matmul_cooperative",
                    &format!("run_matmul_{suffix}"),
                    &constants,
                );
            }
        }
        compiler.insert_case(
            "run_matmul",
            ast::Arm {
                pattern: ast::Pattern::Integer(*index),
                body: vec![ast::Statement::Expression(ast::Expression::call(
                    format!("run_matmul_{index}"),
                    vec![ast::Expression::name("task"), ast::Expression::name("lid")],
                ))],
            },
        );
    }
}

#[neura_compiler::module]
mod device {
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
        let left_slot = buffer * MATMUL_ROWS * MATMUL_LEFT_STRIDE;
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
            scratch[left_slot + row * MATMUL_LEFT_STRIDE + column] =
                select(0.0, fetch(left, address), inside);
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
            scratch[SCRATCH_MATMUL_RIGHT + right_slot + unit] =
                select(0.0, fetch(right, address), inside);
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
        let thread_column = lid % MATMUL_THREAD_COLUMNS;
        let mut acc = scalar_array(0.0, MATMUL_REGISTERS);
        let mut left_registers = scalar_array(0.0, MATMUL_REGISTER_ROWS);
        let mut right_registers = scalar_array(0.0, MATMUL_REGISTER_COLUMNS);
        let segmented = task.split == split::SEGMENT;
        let mut segment_start = 0u32;
        if segmented {
            if task.count == 0u32 {
                return;
            }
            segment_start = whole_index(
                fetch(values[task.segment], task.plane),
                rows + 1u32,
                kind::MATMUL,
                refusal::INDEX,
            );
        }
        for tile in stride(task.first, task.first + task.count, 1u32) {
            let plane = tile / tiles_per_plane;
            let mut plane_row = plane / plane_columns;
            let mut plane_column = plane % plane_columns;
            let within = tile % tiles_per_plane;
            let mut base_row = (within / column_blocks) * MATMUL_ROWS;
            let mut base_column = (within % column_blocks) * MATMUL_COLUMNS;
            let mut left_plane = plane_row * left.strides.x + plane_column * left.strides.y;
            let mut right_plane = plane_row * right.strides.x + plane_column * right.strides.y;
            let mut out_plane = (task.slot * planes + plane) * rows * columns;
            let mut row_bound = rows;
            let mut chain_row = 0u32;
            if segmented {
                base_row = 0u32;
                base_column = (tile % column_blocks) * MATMUL_COLUMNS;
                plane_row = 0u32;
                plane_column = 0u32;
                chain_row = segment_start + (tile / column_blocks) * MATMUL_ROWS;
                left_plane = chain_row * left.strides.z;
                right_plane = task.plane * right.strides.x;
                out_plane = (task.slot * rows + chain_row) * columns;
                row_bound = task.keys;
            }
            for register in unroll(0u32, MATMUL_REGISTERS, 1u32) {
                acc[register] = 0.0;
            }
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
                row_bound,
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
                        row_bound,
                        depth,
                        columns,
                    );
                }
                let left_slot = buffer * MATMUL_ROWS * MATMUL_LEFT_STRIDE;
                let right_slot = buffer * MATMUL_DEPTH * MATMUL_COLUMNS;
                for step in stride(0u32, MATMUL_DEPTH, 1u32) {
                    for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                        left_registers[row] =
                            scratch[left_slot + (thread_row + row) * MATMUL_LEFT_STRIDE + step];
                    }
                    for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                        right_registers[column] = scratch[SCRATCH_MATMUL_RIGHT
                            + right_slot
                            + step * MATMUL_COLUMNS
                            + thread_column
                            + column * MATMUL_THREAD_COLUMNS];
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
                    let out_column = base_column + thread_column + column * MATMUL_THREAD_COLUMNS;
                    if out_row < row_bound && out_column < columns {
                        let index = out_row * columns + out_column;
                        publish(
                            output,
                            out_plane + index,
                            chained(
                                task,
                                uvec4(plane_row, plane_column, chain_row + out_row, out_column),
                                acc[row * MATMUL_REGISTER_COLUMNS + column],
                            ),
                        );
                    }
                }
            }
        }
    }

    fn template_run_matmul_stream(task: Task, lid: u32) {
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
        let segmented = task.split == split::SEGMENT;
        let mut segment_start = 0u32;
        if segmented {
            if task.count == 0u32 {
                return;
            }
            segment_start = whole_index(
                fetch(values[task.segment], task.plane),
                rows + 1u32,
                kind::MATMUL,
                refusal::INDEX,
            );
        }
        for tile in stride(task.first, task.first + task.count, 1u32) {
            let plane = tile / tiles_per_plane;
            let mut plane_row = plane / plane_columns;
            let mut plane_column = plane % plane_columns;
            let within = tile % tiles_per_plane;
            let mut base_row = (within / column_blocks) * MATMUL_ROWS;
            let mut base_column = (within % column_blocks) * MATMUL_COLUMNS;
            let mut left_plane = plane_row * left.strides.x + plane_column * left.strides.y;
            let mut right_plane = plane_row * right.strides.x + plane_column * right.strides.y;
            let mut out_plane = (task.slot * planes + plane) * rows * columns;
            let mut row_bound = rows;
            let mut chain_row = 0u32;
            if segmented {
                base_row = 0u32;
                base_column = (tile % column_blocks) * MATMUL_COLUMNS;
                plane_row = 0u32;
                plane_column = 0u32;
                chain_row = segment_start + (tile / column_blocks) * MATMUL_ROWS;
                left_plane = chain_row * left.strides.z;
                right_plane = task.plane * right.strides.x;
                out_plane = (task.slot * rows + chain_row) * columns;
                row_bound = task.keys;
            }
            let mut acc = scalar_array(0.0, MATMUL_REGISTERS);
            for block in stride(first_block, last_block, 1u32) {
                for step in stride(
                    block * MATMUL_DEPTH,
                    min((block + 1u32) * MATMUL_DEPTH, depth),
                    1u32,
                ) {
                    let mut left_registers = scalar_array(0.0, MATMUL_REGISTER_ROWS);
                    for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                        let at = base_row + thread_row + row;
                        let inside = at < row_bound;
                        let address = select(
                            0u32,
                            left_plane + at * left.strides.z + step * left.strides.w,
                            inside,
                        );
                        left_registers[row] = select(0.0, fetch(left, address), inside);
                    }
                    let mut right_registers = scalar_array(0.0, MATMUL_REGISTER_COLUMNS);
                    for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                        let at = base_column + thread_column + column;
                        let inside = at < columns;
                        let address = select(
                            0u32,
                            right_plane + step * right.strides.z + at * right.strides.w,
                            inside,
                        );
                        right_registers[column] = select(0.0, fetch(right, address), inside);
                    }
                    for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                        for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                            let register = row * MATMUL_REGISTER_COLUMNS + column;
                            acc[register] =
                                acc[register] + left_registers[row] * right_registers[column];
                        }
                    }
                }
            }
            for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                    let out_row = base_row + thread_row + row;
                    let out_column = base_column + thread_column + column;
                    if out_row < row_bound && out_column < columns {
                        let index = out_row * columns + out_column;
                        publish(
                            output,
                            out_plane + index,
                            chained(
                                task,
                                uvec4(plane_row, plane_column, chain_row + out_row, out_column),
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

#[neura_compiler::module]
mod cooperative {
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
        let left_slot = buffer * MATMUL_ROWS * MATMUL_STAGE_DEPTH;
        for unit in stride(lid, MATMUL_ROWS * MATMUL_STAGE_DEPTH, WORKGROUP_SIZE) {
            let row = unit / MATMUL_STAGE_DEPTH;
            let column = unit % MATMUL_STAGE_DEPTH;
            let at_row = base_row + row;
            let at_column = base_depth + column;
            let inside = at_row < rows && at_column < depth;
            let address = select(
                0u32,
                left_plane + at_row * left.strides.z + at_column * left.strides.w,
                inside,
            );
            scratch_half[left_slot + row * MATMUL_LEFT_STRIDE + column] =
                f16(select(0.0, fetch(left, address), inside));
        }
        let right_slot = SCRATCH_COOPERATIVE_RIGHT + buffer * MATMUL_STAGE_DEPTH * MATMUL_COLUMNS;
        for unit in stride(lid, MATMUL_STAGE_DEPTH * MATMUL_COLUMNS, WORKGROUP_SIZE) {
            let row = base_depth + unit / MATMUL_COLUMNS;
            let column = base_column + unit % MATMUL_COLUMNS;
            let inside = row < depth && column < columns;
            let address = select(
                0u32,
                right_plane + row * right.strides.z + column * right.strides.w,
                inside,
            );
            scratch_half[right_slot + unit] = f16(select(0.0, fetch(right, address), inside));
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
        let stage_blocks = (depth + MATMUL_STAGE_DEPTH - 1u32) / MATMUL_STAGE_DEPTH;
        let tiles_per_plane = row_blocks * column_blocks;
        let first_block = (task.slot * stage_blocks) / task.splits;
        let last_block = ((task.slot + 1u32) * stage_blocks) / task.splits;
        let subgroup = lid / COOPMAT_SUBGROUP;
        let subgroup_row = subgroup / MATMUL_SUBGROUP_COLUMNS;
        let subgroup_column = subgroup % MATMUL_SUBGROUP_COLUMNS;
        let subgroup_slot = subgroup * COOPMAT_ROWS * COOPMAT_COLUMNS;
        let copy_words =
            MATMUL_SUBGROUP_ROWS * MATMUL_SUBGROUP_COLUMNS * COOPMAT_ROWS * COOPMAT_COLUMNS;
        let segmented = task.split == split::SEGMENT;
        let mut segment_start = 0u32;
        if segmented {
            if task.count == 0u32 {
                return;
            }
            segment_start = whole_index(
                fetch(values[task.segment], task.plane),
                rows + 1u32,
                kind::MATMUL,
                refusal::INDEX,
            );
        }
        for tile in stride(task.first, task.first + task.count, 1u32) {
            let plane = tile / tiles_per_plane;
            let mut plane_row = plane / plane_columns;
            let mut plane_column = plane % plane_columns;
            let within = tile % tiles_per_plane;
            let mut base_row = (within / column_blocks) * MATMUL_ROWS;
            let mut base_column = (within % column_blocks) * MATMUL_COLUMNS;
            let mut left_plane = plane_row * left.strides.x + plane_column * left.strides.y;
            let mut right_plane = plane_row * right.strides.x + plane_column * right.strides.y;
            let mut out_plane = (task.slot * planes + plane) * rows * columns;
            let mut row_bound = rows;
            let mut chain_row = 0u32;
            if segmented {
                base_row = 0u32;
                base_column = (tile % column_blocks) * MATMUL_COLUMNS;
                plane_row = 0u32;
                plane_column = 0u32;
                chain_row = segment_start + (tile / column_blocks) * MATMUL_ROWS;
                left_plane = chain_row * left.strides.z;
                right_plane = task.plane * right.strides.x;
                out_plane = (task.slot * rows + chain_row) * columns;
                row_bound = task.keys;
            }
            let mut accumulators = scalar_array(coopmat_accumulator(0.0), MATMUL_ACCUMULATORS);
            let mut buffer = 0u32;
            template_matmul_cooperative_load(
                lid,
                base_row,
                base_column,
                first_block * MATMUL_STAGE_DEPTH,
                0u32,
                left_plane,
                right_plane,
                left,
                right,
                row_bound,
                depth,
                columns,
            );
            workgroup_barrier();
            for block in stride(first_block, last_block, 1u32) {
                if block + 1u32 < last_block {
                    template_matmul_cooperative_load(
                        lid,
                        base_row,
                        base_column,
                        (block + 1u32) * MATMUL_STAGE_DEPTH,
                        1u32 - buffer,
                        left_plane,
                        right_plane,
                        left,
                        right,
                        row_bound,
                        depth,
                        columns,
                    );
                }
                let left_slot = buffer * MATMUL_ROWS * MATMUL_LEFT_STRIDE;
                let right_slot =
                    SCRATCH_COOPERATIVE_RIGHT + buffer * MATMUL_STAGE_DEPTH * MATMUL_COLUMNS;
                for step in unroll(0u32, MATMUL_STAGE_DEPTH / COOPMAT_DEPTH, 1u32) {
                    for fragment_row in unroll(0u32, MATMUL_FRAGMENT_ROWS, 1u32) {
                        for fragment_column in unroll(0u32, MATMUL_FRAGMENT_COLUMNS, 1u32) {
                            let left_fragment = coopmat_load_row(
                                &scratch_half[left_slot
                                    + (subgroup_row * MATMUL_FRAGMENT_ROWS + fragment_row)
                                        * COOPMAT_ROWS
                                        * MATMUL_LEFT_STRIDE
                                    + step * COOPMAT_DEPTH],
                                MATMUL_LEFT_STRIDE,
                            );
                            let right_fragment = coopmat_load_b_row(
                                &scratch_half[right_slot
                                    + step * COOPMAT_DEPTH * MATMUL_COLUMNS
                                    + (subgroup_column * MATMUL_FRAGMENT_COLUMNS
                                        + fragment_column)
                                        * COOPMAT_COLUMNS],
                                MATMUL_COLUMNS,
                            );
                            let register = fragment_row * MATMUL_FRAGMENT_COLUMNS + fragment_column;
                            accumulators[register] = coopmat_muladd(
                                left_fragment,
                                right_fragment,
                                accumulators[register],
                            );
                        }
                    }
                }
                workgroup_barrier();
                buffer = 1u32 - buffer;
            }
            for fragment_row in unroll(0u32, MATMUL_FRAGMENT_ROWS, 1u32) {
                for fragment_column in unroll(0u32, MATMUL_FRAGMENT_COLUMNS, 1u32) {
                    let register = fragment_row * MATMUL_FRAGMENT_COLUMNS + fragment_column;
                    coopmat_store(
                        &scratch[subgroup_slot],
                        COOPMAT_COLUMNS,
                        accumulators[register],
                    );
                    workgroup_barrier();
                    for unit in stride(lid, copy_words, WORKGROUP_SIZE) {
                        let slot = unit / (COOPMAT_ROWS * COOPMAT_COLUMNS);
                        let within_slot = unit % (COOPMAT_ROWS * COOPMAT_COLUMNS);
                        let out_row = base_row
                            + (slot / MATMUL_SUBGROUP_COLUMNS)
                                * MATMUL_FRAGMENT_ROWS
                                * COOPMAT_ROWS
                            + fragment_row * COOPMAT_ROWS
                            + within_slot / COOPMAT_COLUMNS;
                        let out_column = base_column
                            + (slot % MATMUL_SUBGROUP_COLUMNS)
                                * MATMUL_FRAGMENT_COLUMNS
                                * COOPMAT_COLUMNS
                            + fragment_column * COOPMAT_COLUMNS
                            + within_slot % COOPMAT_COLUMNS;
                        if out_row < row_bound && out_column < columns {
                            publish(
                                output,
                                out_plane + out_row * columns + out_column,
                                chained(
                                    task,
                                    uvec4(plane_row, plane_column, chain_row + out_row, out_column),
                                    scratch[unit],
                                ),
                            );
                        }
                    }
                    workgroup_barrier();
                }
            }
        }
    }
}
