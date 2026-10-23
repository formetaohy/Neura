use neura_compiler::{Compiler, ast};
use neura_profile::{Geometry, MatmulStrategy};

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
    compiler.constant("QUANT_RUN_BLOCKS", neura_profile::BLOCK_RUN_WORDS);
    compiler.select("run_weight_q8_0", "run_weight");
}

pub(crate) fn install_tiles(compiler: &mut Compiler, geometry: &Geometry) {
    compiler.constant("SCRATCH_MATMUL_RIGHT", geometry.matmul_right());
    compiler.constant("MATMUL_RUN", neura_profile::STREAMED_DEPTH);
    compiler.constant("FOLDED_LANES", neura_profile::FOLDED_LANES);
    compiler.constant("FOLDED_RUN", neura_profile::FOLDED_RUN);
    compiler.constant("SCRATCH_MATMUL_RUN", geometry.matmul_run());
    compiler.constant("FOLDED_SHIFT", neura_profile::FOLDED_LANES.trailing_zeros());
    compiler.constant(
        "FOLDED_ROW_STRIDE",
        u32::try_from(neura_profile::folded_stage_words(1))
            .expect("a staged left fits one device word"),
    );
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
            MatmulStrategy::Folded => {
                constants.push(("MATMUL_FOLDS", tile.folds()));
                constants.push((
                    "MATMUL_FOLD_UNROLL",
                    if tile.stages_a_run() {
                        tile.folds()
                    } else {
                        tile.folds().min(4)
                    },
                ));
                constants.push((
                    "MATMUL_FOLDED_PARTIALS",
                    match tile.rows() > 1 {
                        true => 0u32,
                        false => u32::try_from(neura_profile::folded_stage_words(tile.rows()))
                            .expect("a staged left fits one device word"),
                    },
                ));
                let (dot, stage, gate) = match (tile.rows(), tile.stages_a_run()) {
                    (1, true) => ("dot_run", "stage", "run_gate"),
                    (1, false) => ("dot_staged", "stage_untouched", "run_gate_none"),
                    (2, true) => ("dot_pair", "stage", "run_gate"),
                    (2, false) => ("dot_pair", "stage_untouched", "run_gate_none"),
                    (4, false) => ("dot_quad", "stage_untouched", "run_gate_none"),
                    (rows, _) => panic!(
                        "a folded tile folds {rows} rows, and this device folds one, two or four"
                    ),
                };
                let dot_name = match dot {
                    "dot_pair" => format!("dot_pair_{suffix}"),
                    "dot_quad" => format!("dot_quad_{suffix}"),
                    _ => format!("dot_{suffix}"),
                };
                compiler.select(dot, &dot_name);
                compiler.select(stage, &format!("stage_{suffix}"));
                compiler.select(gate, &format!("gate_{suffix}"));
                let template = match tile.rows() {
                    1 => "template_run_matmul_folded",
                    2 => "template_run_matmul_folded_fused",
                    _ => "template_run_matmul_folded_quad",
                };
                compiler.specialize(template, &format!("run_matmul_{suffix}"), &constants);
            }
            MatmulStrategy::Staged if tile.run().is_some() => {
                let codes = tile.run() == Some(neura_profile::Run::Codes);
                constants.extend([
                    ("MATMUL_LEFT_STRIDE", tile.left_stride()),
                    (
                        "MATMUL_RUN_STRIDE",
                        match codes {
                            true => tile.columns() * neura_profile::RUN_SLOT,
                            false => u32::try_from(tile.run_words())
                                .expect("a run of a staged tile fits one device word address"),
                        },
                    ),
                    ("MATMUL_RUN_BLOCKS", tile.depth() / neura_abi::QUANT_BLOCK),
                    ("MATMUL_THREAD_COLUMNS", tile.thread_columns()),
                    ("MATMUL_REGISTER_ROWS", tile.register_rows()),
                    ("MATMUL_REGISTER_COLUMNS", tile.register_columns()),
                    ("MATMUL_REGISTERS", tile.registers()),
                ]);
                let (load, run, name) = match codes {
                    true => (
                        "template_matmul_load_codes",
                        "template_run_matmul_codes",
                        "matmul_load_codes",
                    ),
                    false => (
                        "template_matmul_load_run",
                        "template_run_matmul_run",
                        "matmul_load_run",
                    ),
                };
                compiler.specialize(load, &format!("{name}_{suffix}"), &constants);
                compiler.specialize(run, &format!("run_matmul_{suffix}"), &constants);
            }
            MatmulStrategy::Staged => {
                constants.extend([
                    ("MATMUL_LEFT_STRIDE", tile.left_stride()),
                    ("MATMUL_RIGHT_STRIDE", tile.depth() + 1),
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
    fn run_gate() {
        workgroup_barrier();
    }

    fn run_gate_none() {}

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
        let left_across = left.strides.z < left.strides.w;
        for unit in stride(lid, MATMUL_ROWS * MATMUL_DEPTH, WORKGROUP_SIZE) {
            let row = select(unit / MATMUL_DEPTH, unit % MATMUL_ROWS, left_across);
            let column = select(unit % MATMUL_DEPTH, unit / MATMUL_ROWS, left_across);
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
        let right_slot = buffer * MATMUL_COLUMNS * MATMUL_RIGHT_STRIDE;
        let right_across = right.strides.z < right.strides.w;
        for unit in stride(lid, MATMUL_DEPTH * MATMUL_COLUMNS, WORKGROUP_SIZE) {
            let step = select(unit / MATMUL_COLUMNS, unit % MATMUL_DEPTH, right_across);
            let seat = select(unit % MATMUL_COLUMNS, unit / MATMUL_DEPTH, right_across);
            let row = base_depth + step;
            let column = base_column + seat;
            let inside = row < depth && column < columns;
            let address = select(
                0u32,
                right_plane + row * right.strides.z + column * right.strides.w,
                inside,
            );
            scratch[SCRATCH_MATMUL_RIGHT + right_slot + seat * MATMUL_RIGHT_STRIDE + step] =
                select(0.0, fetch(right, address), inside);
        }
    }

    fn template_matmul_load_run(
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
        let left_across = left.strides.z < left.strides.w;
        for unit in stride(lid, MATMUL_ROWS * MATMUL_DEPTH, WORKGROUP_SIZE) {
            let row = select(unit / MATMUL_DEPTH, unit % MATMUL_ROWS, left_across);
            let column = select(unit % MATMUL_DEPTH, unit / MATMUL_ROWS, left_across);
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
        let right_slot = buffer * MATMUL_COLUMNS * MATMUL_RUN_STRIDE;
        for block in unroll(0u32, MATMUL_RUN_BLOCKS, 1u32) {
            let at_depth = base_depth + block * Q8_0_BLOCK;
            for unit in stride(lid, MATMUL_COLUMNS * QUANT_RUN_BLOCKS, WORKGROUP_SIZE) {
                let seat = unit / QUANT_RUN_BLOCKS;
                let within = unit % QUANT_RUN_BLOCKS;
                let column = base_column + seat;
                let inside = at_depth < depth && column < columns;
                let address = select(
                    0u32,
                    right_plane + at_depth * right.strides.z + column * right.strides.w,
                    inside,
                );
                let byte = (address / Q8_0_BLOCK) * Q8_0_BYTES;
                let start = byte >> 2u32;
                let odd = (byte & 2u32) == 2u32;
                let low = ggml_raw(right, start + within);
                let high = ggml_raw(right, start + within + 1u32);
                let shifted = select(low, (low >> 16u32) | (high << 16u32), odd);
                scratch[SCRATCH_MATMUL_RIGHT
                    + right_slot
                    + seat * MATMUL_RUN_STRIDE
                    + block * QUANT_RUN_BLOCKS
                    + within] = select(0.0, bitcast_f32(shifted), inside);
            }
        }
    }

    fn run_weight_q8_0(base: u32, at: u32) -> f32 {
        let block = at / Q8_0_BLOCK;
        let within = at % Q8_0_BLOCK;
        let held = scratch[base + QUANT_RUN_BLOCKS * block];
        let scale = unpack2x16float(bitcast_u32(held)).x;
        let offset = (2u32 + within) / 4u32;
        let word = bitcast_u32(scratch[base + QUANT_RUN_BLOCKS * block + offset]);
        let byte = (word >> (((2u32 + within) % 4u32) * 8u32)) & 0xffu32;
        return ggml_signed(byte) * scale;
    }

    fn template_matmul_load_codes(
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
        let left_across = left.strides.z < left.strides.w;
        for unit in stride(lid, MATMUL_ROWS * MATMUL_DEPTH, WORKGROUP_SIZE) {
            let row = select(unit / MATMUL_DEPTH, unit % MATMUL_ROWS, left_across);
            let column = select(unit % MATMUL_DEPTH, unit / MATMUL_ROWS, left_across);
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
        let right_slot = buffer * MATMUL_RUN_STRIDE * MATMUL_RUN_BLOCKS;
        for block in unroll(0u32, MATMUL_RUN_BLOCKS, 1u32) {
            let at_depth = base_depth + block * QUANT_BLOCK;
            for unit in stride(lid, MATMUL_COLUMNS * PANEL_PARTS, WORKGROUP_SIZE) {
                let column = base_column + unit / PANEL_PARTS;
                let part = unit % PANEL_PARTS;
                let inside = at_depth < depth && column < columns;
                let address = select(
                    0u32,
                    right_plane + at_depth * right.strides.z + column * right.strides.w,
                    inside,
                );
                codes(
                    right,
                    address,
                    inside,
                    SCRATCH_MATMUL_RIGHT
                        + right_slot
                        + block * MATMUL_RUN_STRIDE
                        + (unit / PANEL_PARTS) * RUN_SLOT,
                    part,
                );
            }
        }
    }

    fn template_run_matmul_folded(task: Task, lid: u32) {
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
        let walked = (last_block - first_block) * MATMUL_DEPTH;
        let lane = lid % FOLDED_LANES;
        let warp = lid / FOLDED_LANES;
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
            for slot in stride(lid, walked, WORKGROUP_SIZE) {
                let at = first_block * MATMUL_DEPTH + slot;
                let staged = slot + (slot >> FOLDED_SHIFT);
                for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                    let at_row = base_row + row;
                    let inside = at_row < row_bound && at < depth;
                    let address = select(
                        0u32,
                        left_plane + at_row * left.strides.z + at * left.strides.w,
                        inside,
                    );
                    scratch[row * FOLDED_ROW_STRIDE + staged] =
                        select(0.0, fetch(left, address), inside);
                }
            }
            workgroup_barrier();
            for group in stride(0u32, MATMUL_FOLDS, MATMUL_FOLD_UNROLL) {
                for fold in unroll(0u32, MATMUL_FOLD_UNROLL, 1u32) {
                    let column = base_column + warp * MATMUL_FOLDS + group + fold;
                    let blocks = last_block - first_block;
                    let mut acc = scalar_array(0.0, MATMUL_ROWS);
                    for chunk in stride(0u32, (blocks + FOLDED_LANES - 1u32) / FOLDED_LANES, 1u32) {
                        template_gate();
                        let step = chunk * FOLDED_LANES + lane;
                        let at = (first_block + step) * MATMUL_DEPTH;
                        let staged = template_stage(
                            right,
                            right_plane + at * right.strides.z + column * right.strides.w,
                            min(blocks - chunk * FOLDED_LANES, FOLDED_LANES),
                            warp,
                            lane,
                        );
                        template_gate();
                        if step < blocks && at < depth {
                            let last = min(at + MATMUL_DEPTH, depth) - at;
                            for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                                let inside = base_row + row < row_bound && column < columns;
                                acc[row] = acc[row]
                                    + template_dot(
                                        right,
                                        left,
                                        staged,
                                        step * (MATMUL_DEPTH + 1u32) + row * FOLDED_ROW_STRIDE,
                                        last,
                                        inside,
                                    );
                            }
                        }
                    }
                    let folded = MATMUL_FOLDED_PARTIALS
                        + lane * (MATMUL_COLUMNS + 1u32)
                        + warp * MATMUL_FOLDS
                        + group
                        + fold;
                    for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                        scratch[folded + row * FOLDED_LANES * (MATMUL_COLUMNS + 1u32)] = acc[row];
                    }
                }
            }
            workgroup_barrier();
            if lid < MATMUL_COLUMNS {
                let column = base_column + lid;
                for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                    let mut total = 0.0;
                    for other in unroll(0u32, FOLDED_LANES, 1u32) {
                        total = total
                            + scratch[MATMUL_FOLDED_PARTIALS
                                + row * FOLDED_LANES * (MATMUL_COLUMNS + 1u32)
                                + other * (MATMUL_COLUMNS + 1u32)
                                + lid];
                    }
                    let out_row = base_row + row;
                    if out_row < row_bound && column < columns {
                        let index = out_row * columns + column;
                        publish(
                            output,
                            out_plane + index,
                            chained(
                                task,
                                uvec4(plane_row, plane_column, chain_row + out_row, column),
                                total,
                            ),
                        );
                    }
                }
            }
        }
    }

    fn template_run_matmul_folded_fused(task: Task, lid: u32) {
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
        let walked = (last_block - first_block) * MATMUL_DEPTH;
        let lane = lid % FOLDED_LANES;
        let warp = lid / FOLDED_LANES;
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
            for slot in stride(lid, walked, WORKGROUP_SIZE) {
                let at = first_block * MATMUL_DEPTH + slot;
                let staged = slot + (slot >> FOLDED_SHIFT);
                for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                    let at_row = base_row + row;
                    let inside = at_row < row_bound && at < depth;
                    let address = select(
                        0u32,
                        left_plane + at_row * left.strides.z + at * left.strides.w,
                        inside,
                    );
                    scratch[row * FOLDED_ROW_STRIDE + staged] =
                        select(0.0, fetch(left, address), inside);
                }
            }
            workgroup_barrier();
            let mut acc = scalar_array(0.0, MATMUL_ROWS * MATMUL_FOLDS);
            for fold in unroll(0u32, MATMUL_FOLDS, 1u32) {
                let column = base_column + warp * MATMUL_FOLDS + fold;
                let blocks = last_block - first_block;
                for chunk in stride(0u32, (blocks + FOLDED_LANES - 1u32) / FOLDED_LANES, 1u32) {
                    template_gate();
                    let step = chunk * FOLDED_LANES + lane;
                    let at = (first_block + step) * MATMUL_DEPTH;
                    let staged = template_stage(
                        right,
                        right_plane + at * right.strides.z + column * right.strides.w,
                        min(blocks - chunk * FOLDED_LANES, FOLDED_LANES),
                        warp,
                        lane,
                    );
                    template_gate();
                    if step < blocks && at < depth {
                        let last = min(at + MATMUL_DEPTH, depth) - at;
                        let inside = base_row < row_bound && column < columns;
                        let paired = template_dot_pair(
                            right,
                            left,
                            staged,
                            step * (MATMUL_DEPTH + 1u32),
                            step * (MATMUL_DEPTH + 1u32) + FOLDED_ROW_STRIDE,
                            last,
                            inside,
                        );
                        acc[fold] = acc[fold] + paired.x;
                        acc[MATMUL_FOLDS + fold] = acc[MATMUL_FOLDS + fold] + paired.y;
                    }
                }
            }
            workgroup_barrier();
            for fold in unroll(0u32, MATMUL_FOLDS, 1u32) {
                let column = warp * MATMUL_FOLDS + fold;
                for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                    scratch[row * FOLDED_LANES * (MATMUL_COLUMNS + 1u32)
                        + lane * (MATMUL_COLUMNS + 1u32)
                        + column] = acc[row * MATMUL_FOLDS + fold];
                }
            }
            workgroup_barrier();
            if lid < MATMUL_COLUMNS {
                let column = base_column + lid;
                for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                    let mut total = 0.0;
                    for other in unroll(0u32, FOLDED_LANES, 1u32) {
                        total = total
                            + scratch[row * FOLDED_LANES * (MATMUL_COLUMNS + 1u32)
                                + other * (MATMUL_COLUMNS + 1u32)
                                + lid];
                    }
                    let out_row = base_row + row;
                    if out_row < row_bound && column < columns {
                        let index = out_row * columns + column;
                        publish(
                            output,
                            out_plane + index,
                            chained(
                                task,
                                uvec4(plane_row, plane_column, chain_row + out_row, column),
                                total,
                            ),
                        );
                    }
                }
            }
        }
    }

    fn template_run_matmul_folded_quad(task: Task, lid: u32) {
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
        let walked = (last_block - first_block) * MATMUL_DEPTH;
        let lane = lid % FOLDED_LANES;
        let warp = lid / FOLDED_LANES;
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
            for slot in stride(lid, walked, WORKGROUP_SIZE) {
                let at = first_block * MATMUL_DEPTH + slot;
                let staged = slot + (slot >> FOLDED_SHIFT);
                for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                    let at_row = base_row + row;
                    let inside = at_row < row_bound && at < depth;
                    let address = select(
                        0u32,
                        left_plane + at_row * left.strides.z + at * left.strides.w,
                        inside,
                    );
                    scratch[row * FOLDED_ROW_STRIDE + staged] =
                        select(0.0, fetch(left, address), inside);
                }
            }
            workgroup_barrier();
            let mut acc = scalar_array(0.0, MATMUL_ROWS * MATMUL_FOLDS);
            for fold in unroll(0u32, MATMUL_FOLDS, 1u32) {
                let column = base_column + warp * MATMUL_FOLDS + fold;
                let blocks = last_block - first_block;
                for chunk in stride(0u32, (blocks + FOLDED_LANES - 1u32) / FOLDED_LANES, 1u32) {
                    template_gate();
                    let step = chunk * FOLDED_LANES + lane;
                    let at = (first_block + step) * MATMUL_DEPTH;
                    let staged = template_stage(
                        right,
                        right_plane + at * right.strides.z + column * right.strides.w,
                        min(blocks - chunk * FOLDED_LANES, FOLDED_LANES),
                        warp,
                        lane,
                    );
                    template_gate();
                    if step < blocks && at < depth {
                        let last = min(at + MATMUL_DEPTH, depth) - at;
                        let inside = base_row < row_bound && column < columns;
                        let quad = template_dot_quad(
                            right,
                            left,
                            staged,
                            step * (MATMUL_DEPTH + 1u32),
                            step * (MATMUL_DEPTH + 1u32) + FOLDED_ROW_STRIDE,
                            step * (MATMUL_DEPTH + 1u32) + 2u32 * FOLDED_ROW_STRIDE,
                            step * (MATMUL_DEPTH + 1u32) + 3u32 * FOLDED_ROW_STRIDE,
                            last,
                            inside,
                        );
                        acc[fold] = acc[fold] + quad.x;
                        acc[MATMUL_FOLDS + fold] = acc[MATMUL_FOLDS + fold] + quad.y;
                        acc[2u32 * MATMUL_FOLDS + fold] = acc[2u32 * MATMUL_FOLDS + fold] + quad.z;
                        acc[3u32 * MATMUL_FOLDS + fold] = acc[3u32 * MATMUL_FOLDS + fold] + quad.w;
                    }
                }
            }
            workgroup_barrier();
            for fold in unroll(0u32, MATMUL_FOLDS, 1u32) {
                let column = warp * MATMUL_FOLDS + fold;
                for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                    scratch[row * FOLDED_LANES * (MATMUL_COLUMNS + 1u32)
                        + lane * (MATMUL_COLUMNS + 1u32)
                        + column] = acc[row * MATMUL_FOLDS + fold];
                }
            }
            workgroup_barrier();
            if lid < MATMUL_COLUMNS {
                let column = base_column + lid;
                for row in unroll(0u32, MATMUL_ROWS, 1u32) {
                    let mut total = 0.0;
                    for other in unroll(0u32, FOLDED_LANES, 1u32) {
                        total = total
                            + scratch[row * FOLDED_LANES * (MATMUL_COLUMNS + 1u32)
                                + other * (MATMUL_COLUMNS + 1u32)
                                + lid];
                    }
                    let out_row = base_row + row;
                    if out_row < row_bound && column < columns {
                        let index = out_row * columns + column;
                        publish(
                            output,
                            out_plane + index,
                            chained(
                                task,
                                uvec4(plane_row, plane_column, chain_row + out_row, column),
                                total,
                            ),
                        );
                    }
                }
            }
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
                let right_slot = buffer * MATMUL_COLUMNS * MATMUL_RIGHT_STRIDE;
                for step in unroll(0u32, MATMUL_DEPTH, 1u32) {
                    for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                        left_registers[row] =
                            scratch[left_slot + (thread_row + row) * MATMUL_LEFT_STRIDE + step];
                    }
                    for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                        right_registers[column] = scratch[SCRATCH_MATMUL_RIGHT
                            + right_slot
                            + (thread_column + column * MATMUL_THREAD_COLUMNS)
                                * MATMUL_RIGHT_STRIDE
                            + step];
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

    fn template_run_matmul_run(task: Task, lid: u32) {
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
            template_matmul_load_run(
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
                    template_matmul_load_run(
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
                let right_slot = buffer * MATMUL_COLUMNS * MATMUL_RUN_STRIDE;
                for step in unroll(0u32, MATMUL_DEPTH, 1u32) {
                    for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                        left_registers[row] =
                            scratch[left_slot + (thread_row + row) * MATMUL_LEFT_STRIDE + step];
                    }
                    for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                        right_registers[column] = run_weight(
                            SCRATCH_MATMUL_RIGHT
                                + right_slot
                                + (thread_column + column * MATMUL_THREAD_COLUMNS)
                                    * MATMUL_RUN_STRIDE,
                            step,
                        );
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

    fn template_run_matmul_codes(task: Task, lid: u32) {
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
            template_matmul_load_codes(
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
                    template_matmul_load_codes(
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
                let right_slot = buffer * MATMUL_RUN_STRIDE * MATMUL_RUN_BLOCKS;
                for step in unroll(0u32, MATMUL_DEPTH, 1u32) {
                    for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                        left_registers[row] =
                            scratch[left_slot + (thread_row + row) * MATMUL_LEFT_STRIDE + step];
                    }
                    for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                        right_registers[column] = run_codes(
                            SCRATCH_MATMUL_RIGHT
                                + right_slot
                                + (step / QUANT_BLOCK) * MATMUL_RUN_STRIDE
                                + (thread_column + column * MATMUL_THREAD_COLUMNS) * RUN_SLOT,
                            step % QUANT_BLOCK,
                        );
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
                let step = block * MATMUL_DEPTH;
                let last = min((block + 1u32) * MATMUL_DEPTH, depth) - step;
                for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                    let at_row = base_row + thread_row + row;
                    for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                        let at_column = base_column + thread_column + column;
                        let register = row * MATMUL_REGISTER_COLUMNS + column;
                        acc[register] = acc[register]
                            + dot(
                                right,
                                left,
                                right_plane + step * right.strides.z + at_column * right.strides.w,
                                left_plane + at_row * left.strides.z + step * left.strides.w,
                                last,
                                at_row < row_bound && at_column < columns,
                            );
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
        let left_across = left.strides.z < left.strides.w;
        for unit in stride(lid, MATMUL_ROWS * MATMUL_STAGE_DEPTH, WORKGROUP_SIZE) {
            let row = select(unit / MATMUL_STAGE_DEPTH, unit % MATMUL_ROWS, left_across);
            let column = select(unit % MATMUL_STAGE_DEPTH, unit / MATMUL_ROWS, left_across);
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
        let right_across = right.strides.z < right.strides.w;
        for unit in stride(lid, MATMUL_STAGE_DEPTH * MATMUL_COLUMNS, WORKGROUP_SIZE) {
            let step = select(
                unit / MATMUL_COLUMNS,
                unit % MATMUL_STAGE_DEPTH,
                right_across,
            );
            let seat = select(
                unit % MATMUL_COLUMNS,
                unit / MATMUL_STAGE_DEPTH,
                right_across,
            );
            let row = base_depth + step;
            let column = base_column + seat;
            let inside = row < depth && column < columns;
            let address = select(
                0u32,
                right_plane + row * right.strides.z + column * right.strides.w,
                inside,
            );
            scratch_half[right_slot + step * MATMUL_COLUMNS + seat] =
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
