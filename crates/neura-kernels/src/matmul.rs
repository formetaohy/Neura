use neura_abi::MatmulTile;
use std::fmt::Write as _;

pub fn body(tile: MatmulTile) -> String {
    let mut source = String::new();
    writeln!(
        source,
        "var<workgroup> matmul_left: array<f32, 2u * MATMUL_ROW_TILE * MATMUL_DEPTH_TILE>;"
    )
    .unwrap();
    writeln!(
        source,
        "var<workgroup> matmul_right: array<f32, 2u * MATMUL_DEPTH_TILE * MATMUL_COL_TILE>;"
    )
    .unwrap();
    source.push('\n');
    load(&mut source);
    run(&mut source, tile);
    source
}

fn load(source: &mut String) {
    source.push_str(
        "fn matmul_load(lid: u32, base_row: u32, base_column: u32, base_depth: u32, buffer: u32, left: Value, right: Value, rows: u32, depth: u32, columns: u32) {\n",
    );
    source.push_str(
        "    let left_slot = buffer * MATMUL_ROW_TILE * MATMUL_DEPTH_TILE;\n    for (var unit = lid; unit < MATMUL_ROW_TILE * MATMUL_DEPTH_TILE; unit = unit + WORKGROUP_SIZE) {\n        let row = base_row + unit / MATMUL_DEPTH_TILE;\n        let column = base_depth + unit % MATMUL_DEPTH_TILE;\n        let inside = row < rows && column < depth;\n        let address = select(0u, row * left.strides.z + column * left.strides.w, inside);\n        matmul_left[left_slot + unit] = select(0.0, arena[left.base + address], inside);\n    }\n",
    );
    source.push_str(
        "    let right_slot = buffer * MATMUL_DEPTH_TILE * MATMUL_COL_TILE;\n    for (var unit = lid; unit < MATMUL_DEPTH_TILE * MATMUL_COL_TILE; unit = unit + WORKGROUP_SIZE) {\n        let row = base_depth + unit / MATMUL_COL_TILE;\n        let column = base_column + unit % MATMUL_COL_TILE;\n        let inside = row < depth && column < columns;\n        let address = select(0u, row * right.strides.z + column * right.strides.w, inside);\n        matmul_right[right_slot + unit] = select(0.0, arena[right.base + address], inside);\n    }\n}\n\n",
    );
}

fn run(source: &mut String, tile: MatmulTile) {
    source.push_str("fn run_matmul(task: Task, lid: u32) {\n");
    source.push_str(
        "    let left = values[task.a];\n    let right = values[task.b];\n    let output = values[task.out];\n    let rows = left.dims.z;\n    let depth = left.dims.w;\n    let columns = right.dims.w;\n    let column_blocks = (columns + MATMUL_COL_TILE - 1u) / MATMUL_COL_TILE;\n    let depth_blocks = (depth + MATMUL_DEPTH_TILE - 1u) / MATMUL_DEPTH_TILE;\n    let thread_row = (lid / MATMUL_THREAD_COLUMNS) * MATMUL_REGISTER_ROWS;\n    let thread_column = (lid % MATMUL_THREAD_COLUMNS) * MATMUL_REGISTER_COLUMNS;\n",
    );
    source.push_str(
        "    for (var tile = task.first; tile < task.first + task.count; tile = tile + 1u) {\n",
    );
    source.push_str(
        "        let base_row = (tile / column_blocks) * MATMUL_ROW_TILE;\n        let base_column = (tile % column_blocks) * MATMUL_COL_TILE;\n",
    );
    for register in 0..tile.registers() {
        writeln!(source, "        var acc{register} = 0.0;").unwrap();
    }
    source.push_str(
        "        var buffer = 0u;\n        matmul_load(lid, base_row, base_column, 0u, 0u, left, right, rows, depth, columns);\n        workgroupBarrier();\n        for (var block = 0u; block < depth_blocks; block = block + 1u) {\n            if (block + 1u < depth_blocks) {\n                matmul_load(lid, base_row, base_column, (block + 1u) * MATMUL_DEPTH_TILE, 1u - buffer, left, right, rows, depth, columns);\n            }\n",
    );
    source.push_str(
        "            let left_slot = buffer * MATMUL_ROW_TILE * MATMUL_DEPTH_TILE;\n            let right_slot = buffer * MATMUL_DEPTH_TILE * MATMUL_COL_TILE;\n            for (var step = 0u; step < MATMUL_DEPTH_TILE; step = step + 1u) {\n",
    );
    for row in 0..tile.register_rows() {
        writeln!(
            source,
            "                let left{row} = matmul_left[left_slot + (thread_row + {row}u) * MATMUL_DEPTH_TILE + step];"
        )
        .unwrap();
    }
    for column in 0..tile.register_columns() {
        writeln!(
            source,
            "                let right{column} = matmul_right[right_slot + step * MATMUL_COL_TILE + thread_column + {column}u];"
        )
        .unwrap();
    }
    for row in 0..tile.register_rows() {
        for column in 0..tile.register_columns() {
            let register = row * tile.register_columns() + column;
            writeln!(
                source,
                "                acc{register} = acc{register} + left{row} * right{column};"
            )
            .unwrap();
        }
    }
    source.push_str(
        "            }\n            workgroupBarrier();\n            buffer = 1u - buffer;\n        }\n",
    );
    for row in 0..tile.register_rows() {
        for column in 0..tile.register_columns() {
            let register = row * tile.register_columns() + column;
            writeln!(
                source,
                "        let row{register} = base_row + thread_row + {row}u;\n        let column{register} = base_column + thread_column + {column}u;\n        if (row{register} < rows && column{register} < columns) {{\n            arena[output.base + row{register} * columns + column{register}] = chained(task, row{register} * columns + column{register}, acc{register});\n        }}"
            )
            .unwrap();
        }
    }
    source.push_str("    }\n}\n");
}
