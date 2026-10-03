use neura_compiler::{Compiler, ir};
use neura_profile::{Geometry, MatmulStrategy};

pub fn specialize(compiler: &mut Compiler, geometry: &Geometry) {
    for (index, tile) in geometry.tiles().iter().enumerate() {
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
                compiler.constant("COOPMAT_ROWS", tile.rows() / tile.subgroup_rows());
                compiler.constant("COOPMAT_COLUMNS", tile.columns() / tile.subgroup_columns());
                compiler.constant("COOPMAT_DEPTH", tile.depth());
                constants.extend([
                    ("MATMUL_LEFT_STRIDE", tile.depth()),
                    ("COOPMAT_ROWS", tile.rows() / tile.subgroup_rows()),
                    ("COOPMAT_COLUMNS", tile.columns() / tile.subgroup_columns()),
                    ("COOPMAT_DEPTH", tile.depth()),
                    ("COOPMAT_SUBGROUP", geometry.subgroup()),
                    ("MATMUL_SUBGROUP_COLUMNS", tile.subgroup_columns()),
                    ("MATMUL_SUBGROUP_ROWS", tile.subgroup_rows()),
                    ("SCRATCH_COOPERATIVE_RIGHT", 2 * tile.rows() * tile.depth()),
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
            ir::Arm {
                pattern: ir::Pattern::Integer(
                    index
                        .try_into()
                        .expect("a geometry fits in one device word"),
                ),
                body: vec![ir::Statement::Expression(ir::Expression::call(
                    format!("run_matmul_{index}"),
                    vec![ir::Expression::name("task"), ir::Expression::name("lid")],
                ))],
            },
        );
    }
}

pub fn define_cooperative(compiler: &mut Compiler) {
    super::matmul_cooperative::define(compiler);
}
