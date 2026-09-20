use neura_abi::{Geometry, MatmulTile, kind};
use std::fmt::Write as _;

pub fn family(geometry: Geometry) -> String {
    let mut source = String::new();
    if !geometry.tiles().is_empty() {
        writeln!(
            source,
            "var<workgroup> matmul_left: array<f32, MATMUL_LEFT_STAGE>;"
        )
        .unwrap();
        writeln!(
            source,
            "var<workgroup> matmul_right: array<f32, MATMUL_RIGHT_STAGE>;"
        )
        .unwrap();
        source.push('\n');
    }
    for (index, tile) in geometry.tiles().iter().enumerate() {
        load(&mut source, index);
        run(&mut source, index, *tile);
    }
    dispatch(&mut source, geometry.tiles().len());
    source
}

fn load(source: &mut String, geometry: usize) {
    let function = format!(
        "fn matmul_load_{geometry}(lid: u32, base_row: u32, base_column: u32, base_depth: u32, buffer: u32, left: Value, right: Value, rows: u32, depth: u32, columns: u32) {{\n"
    );
    source.push_str(&function);
    let left = format!(
        "    let left_slot = buffer * MATMUL_ROWS_{geometry} * MATMUL_DEPTH_{geometry};\n    for (var unit = lid; unit < MATMUL_ROWS_{geometry} * MATMUL_DEPTH_{geometry}; unit = unit + WORKGROUP_SIZE) {{\n        let row = base_row + unit / MATMUL_DEPTH_{geometry};\n        let column = base_depth + unit % MATMUL_DEPTH_{geometry};\n        let inside = row < rows && column < depth;\n        let address = select(0u, row * left.strides.z + column * left.strides.w, inside);\n        matmul_left[left_slot + unit] = select(0.0, arena[left.base + address], inside);\n    }}\n",
    );
    source.push_str(&left);
    let right = format!(
        "    let right_slot = buffer * MATMUL_DEPTH_{geometry} * MATMUL_COLUMNS_{geometry};\n    for (var unit = lid; unit < MATMUL_DEPTH_{geometry} * MATMUL_COLUMNS_{geometry}; unit = unit + WORKGROUP_SIZE) {{\n        let row = base_depth + unit / MATMUL_COLUMNS_{geometry};\n        let column = base_column + unit % MATMUL_COLUMNS_{geometry};\n        let inside = row < depth && column < columns;\n        let address = select(0u, row * right.strides.z + column * right.strides.w, inside);\n        matmul_right[right_slot + unit] = select(0.0, arena[right.base + address], inside);\n    }}\n}}\n\n",
    );
    source.push_str(&right);
}

fn run(source: &mut String, geometry: usize, tile: MatmulTile) {
    writeln!(source, "fn run_matmul_{geometry}(task: Task, lid: u32) {{").unwrap();
    source.push_str("    let left = values[task.a];\n    let right = values[task.b];\n    let output = values[task.out];\n    let rows = left.dims.z;\n    let depth = left.dims.w;\n    let columns = right.dims.w;\n");
    writeln!(
        source,
        "    let column_blocks = (columns + MATMUL_COLUMNS_{geometry} - 1u) / MATMUL_COLUMNS_{geometry};",
    )
    .unwrap();
    writeln!(
        source,
        "    let depth_blocks = (depth + MATMUL_DEPTH_{geometry} - 1u) / MATMUL_DEPTH_{geometry};",
    )
    .unwrap();
    writeln!(
        source,
        "    let thread_row = (lid / MATMUL_THREAD_COLUMNS_{geometry}) * MATMUL_REGISTER_ROWS_{geometry};",
    )
    .unwrap();
    writeln!(
        source,
        "    let thread_column = (lid % MATMUL_THREAD_COLUMNS_{geometry}) * MATMUL_REGISTER_COLUMNS_{geometry};",
    )
    .unwrap();
    source.push_str(
        "    for (var tile = task.first; tile < task.first + task.count; tile = tile + 1u) {\n",
    );
    writeln!(
        source,
        "        let base_row = (tile / column_blocks) * MATMUL_ROWS_{geometry};",
    )
    .unwrap();
    writeln!(
        source,
        "        let base_column = (tile % column_blocks) * MATMUL_COLUMNS_{geometry};",
    )
    .unwrap();
    for register in 0..tile.registers() {
        writeln!(source, "        var acc{register} = 0.0;").unwrap();
    }
    writeln!(
        source,
        "        var buffer = 0u;\n        matmul_load_{geometry}(lid, base_row, base_column, 0u, 0u, left, right, rows, depth, columns);\n        workgroupBarrier();\n        for (var block = 0u; block < depth_blocks; block = block + 1u) {{\n            if (block + 1u < depth_blocks) {{\n                matmul_load_{geometry}(lid, base_row, base_column, (block + 1u) * MATMUL_DEPTH_{geometry}, 1u - buffer, left, right, rows, depth, columns);\n            }}",
    )
    .unwrap();
    writeln!(
        source,
        "            let left_slot = buffer * MATMUL_ROWS_{geometry} * MATMUL_DEPTH_{geometry};",
    )
    .unwrap();
    writeln!(
        source,
        "            let right_slot = buffer * MATMUL_DEPTH_{geometry} * MATMUL_COLUMNS_{geometry};",
    )
    .unwrap();
    writeln!(
        source,
        "            for (var step = 0u; step < MATMUL_DEPTH_{geometry}; step = step + 1u) {{",
    )
    .unwrap();
    for row in 0..tile.register_rows() {
        writeln!(source, "                let left{row} = matmul_left[left_slot + (thread_row + {row}u) * MATMUL_DEPTH_{geometry} + step];").unwrap();
    }
    for column in 0..tile.register_columns() {
        writeln!(source, "                let right{column} = matmul_right[right_slot + step * MATMUL_COLUMNS_{geometry} + thread_column + {column}u];").unwrap();
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
    source.push_str("            }\n            workgroupBarrier();\n            buffer = 1u - buffer;\n        }\n");
    for row in 0..tile.register_rows() {
        for column in 0..tile.register_columns() {
            let register = row * tile.register_columns() + column;
            writeln!(
                source,
                "        let row{register} = base_row + thread_row + {row}u;\n        let column{register} = base_column + thread_column + {column}u;\n        if (row{register} < rows && column{register} < columns) {{\n            arena[output.base + row{register} * columns + column{register}] = chained(task, row{register} * columns + column{register}, acc{register});\n        }}",
            )
            .unwrap();
        }
    }
    source.push_str("    }\n}\n\n");
}

fn dispatch(source: &mut String, tiles: usize) {
    source.push_str("fn run_matmul(task: Task, lid: u32) {\n    switch (task.geometry) {\n");
    for geometry in 0..tiles {
        writeln!(
            source,
            "        case {geometry}u: {{ run_matmul_{geometry}(task, lid); }}",
        )
        .unwrap();
    }
    writeln!(
        source,
        "        default: {{ refuse({}, task.geometry); }}\n    }}\n}}\n",
        kind::constant(kind::MATMUL),
    )
    .unwrap();
}
