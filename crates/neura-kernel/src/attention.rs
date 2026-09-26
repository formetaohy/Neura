use neura_compiler::{Compiler, ir};
use neura_profile::Geometry;

pub fn define(compiler: &mut Compiler, geometry: &Geometry) {
    super::attention_device::define(compiler);
    if geometry.attention().is_empty() {
        return;
    }
    let stage = geometry.attention_stage_words();
    compiler.workgroup("attention_left", "f32", stage);
    compiler.workgroup("attention_right", "f32", stage);
    for (index, tile) in geometry.attention().iter().enumerate() {
        let suffix = index.to_string();
        let constants = [("ATTN_KEYS", tile.keys()), ("ATTN_WIDTH", tile.width())];
        compiler.specialize(
            "template_stage_attention",
            &format!("stage_attention_{suffix}"),
            &constants,
        );
        for (template, dispatcher) in [
            ("template_attention_forward", "run_attention"),
            ("template_attention_query_grad", "run_attention_query_grad"),
            ("template_attention_key_grad", "run_attention_key_grad"),
            ("template_attention_value_grad", "run_attention_value_grad"),
        ] {
            let specialized = format!("{dispatcher}_{suffix}");
            compiler.specialize(template, &specialized, &constants);
            compiler.insert_case(
                dispatcher,
                ir::Arm {
                    pattern: ir::Pattern::Integer(
                        index
                            .try_into()
                            .expect("a geometry fits in one device word"),
                    ),
                    body: vec![ir::Statement::Expression(ir::Expression::call(
                        specialized,
                        vec![ir::Expression::name("task"), ir::Expression::name("lid")],
                    ))],
                },
            );
        }
    }
}
