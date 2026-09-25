use neura_compiler::{Compiler, ir};
use neura_profile::Geometry;

pub fn specialize(compiler: &mut Compiler, geometry: &Geometry) {
    for (index, tile) in geometry.tiles().iter().enumerate() {
        let suffix = index.to_string();
        let constants = [
            ("MATMUL_ROWS", tile.rows()),
            ("MATMUL_COLUMNS", tile.columns()),
            ("MATMUL_DEPTH", tile.depth()),
            ("MATMUL_THREAD_COLUMNS", tile.thread_columns()),
            ("MATMUL_REGISTER_ROWS", tile.register_rows()),
            ("MATMUL_REGISTER_COLUMNS", tile.register_columns()),
            ("MATMUL_REGISTERS", tile.registers()),
        ];
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
