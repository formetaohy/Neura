use neura_compiler::{Compiler, ast};
use neura_profile::{Geometry, MatmulStrategy};

pub(crate) fn install(compiler: &mut Compiler, geometry: &Geometry) {
    weight::define(compiler);
    specialize(compiler, geometry);
}

fn specialize(compiler: &mut Compiler, geometry: &Geometry) {
    for (index, tile) in geometry.walked() {
        if tile.strategy() != MatmulStrategy::Staged {
            continue;
        }
        let suffix = index.to_string();
        let constants = [
            ("MATMUL_ROWS", tile.rows()),
            ("MATMUL_COLUMNS", tile.columns()),
            ("MATMUL_DEPTH", tile.depth()),
            ("MATMUL_LEFT_STRIDE", tile.left_stride()),
            ("MATMUL_THREAD_COLUMNS", tile.thread_columns()),
            ("MATMUL_REGISTER_ROWS", tile.register_rows()),
            ("MATMUL_REGISTER_COLUMNS", tile.register_columns()),
            ("MATMUL_REGISTERS", tile.registers()),
        ];
        compiler.specialize(
            "template_run_matmul_weight_grad",
            &format!("matmul_weight_grad_{suffix}"),
            &constants,
        );
        compiler.insert_case(
            "run_matmul_weight_grad",
            ast::Arm {
                pattern: ast::Pattern::Integer(*index),
                body: vec![ast::Statement::Expression(ast::Expression::call(
                    format!("matmul_weight_grad_{index}"),
                    vec![ast::Expression::name("task"), ast::Expression::name("lid")],
                ))],
            },
        );
    }
}

#[neura_compiler::module]
mod weight {
    fn run_matmul_weight_grad(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::MATMUL_WEIGHT_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }

    fn template_run_matmul_weight_grad(task: Task, lid: u32) {
        let left = values[task.a];
        let gradient = values[task.b];
        let output = values[task.out];
        let depth = output.dims.z;
        let columns = output.dims.w;
        let blocks = ceil_div(columns, MATMUL_COLUMNS);
        let base_row = (task.first / blocks) * MATMUL_ROWS;
        let base_column = (task.first % blocks) * MATMUL_COLUMNS;
        let thread_row = (lid / MATMUL_THREAD_COLUMNS) * MATMUL_REGISTER_ROWS;
        let thread_column = lid % MATMUL_THREAD_COLUMNS;
        let mut accumulated = scalar_array(0.0, MATMUL_REGISTERS);
        let mut start = 0u32;
        if task.keys > 0u32 {
            start = whole_index(
                fetch(values[task.segment], task.plane),
                left.dims.z + 1u32,
                kind::MATMUL_WEIGHT_GRAD,
                refusal::INDEX,
            );
        }
        let end = start + task.keys;
        let rounds = ceil_div(task.keys, MATMUL_DEPTH);
        for round in stride(0u32, rounds, 1u32) {
            workgroup_barrier();
            let at_depth = start + round * MATMUL_DEPTH;
            for unit in stride(lid, MATMUL_ROWS * MATMUL_DEPTH, WORKGROUP_SIZE) {
                let row = unit / MATMUL_DEPTH;
                let column = unit % MATMUL_DEPTH;
                let at_row = base_row + row;
                let at = at_depth + column;
                let inside = at < end && at_row < depth;
                let address = select(0u32, at * left.strides.z + at_row * left.strides.w, inside);
                scratch[row * MATMUL_LEFT_STRIDE + column] =
                    select(0.0, fetch(left, address), inside);
            }
            for unit in stride(lid, MATMUL_DEPTH * MATMUL_COLUMNS, WORKGROUP_SIZE) {
                let row = unit / MATMUL_COLUMNS;
                let column = unit % MATMUL_COLUMNS;
                let at = at_depth + row;
                let at_column = base_column + column;
                let inside = at < end && at_column < columns;
                let address = select(
                    0u32,
                    at * gradient.strides.z + at_column * gradient.strides.w,
                    inside,
                );
                scratch[MATMUL_ROWS * MATMUL_LEFT_STRIDE + row * MATMUL_COLUMNS + column] =
                    select(0.0, fetch(gradient, address), inside);
            }
            workgroup_barrier();
            for step in stride(0u32, MATMUL_DEPTH, 1u32) {
                let mut left_registers = scalar_array(0.0, MATMUL_REGISTER_ROWS);
                for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                    left_registers[row] = scratch[(thread_row + row) * MATMUL_LEFT_STRIDE + step];
                }
                let mut right_registers = scalar_array(0.0, MATMUL_REGISTER_COLUMNS);
                for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                    right_registers[column] = scratch[MATMUL_ROWS * MATMUL_LEFT_STRIDE
                        + step * MATMUL_COLUMNS
                        + thread_column
                        + column * MATMUL_THREAD_COLUMNS];
                }
                for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
                    for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                        let register = row * MATMUL_REGISTER_COLUMNS + column;
                        accumulated[register] =
                            accumulated[register] + left_registers[row] * right_registers[column];
                    }
                }
            }
        }
        for row in unroll(0u32, MATMUL_REGISTER_ROWS, 1u32) {
            for column in unroll(0u32, MATMUL_REGISTER_COLUMNS, 1u32) {
                let at_row = base_row + thread_row + row;
                let at_column = base_column + thread_column + column * MATMUL_THREAD_COLUMNS;
                if at_row < depth && at_column < columns {
                    publish(
                        output,
                        (task.plane * depth + at_row) * columns + at_column,
                        accumulated[row * MATMUL_REGISTER_COLUMNS + column],
                    );
                }
            }
        }
    }
}
