use neura_abi::Element;
use neura_compiler::{Compiler, ir};

pub fn define(compiler: &mut Compiler, elements: &[Element]) {
    super::pack_device::define(compiler);
    for element in elements.iter().copied().filter(|element| element.narrow()) {
        compiler.insert_case(
            "run_pack",
            ir::Arm {
                pattern: ir::Pattern::Integer(element.code()),
                body: vec![ir::Statement::Expression(ir::Expression::call(
                    pack_of(element),
                    vec![ir::Expression::name("task"), ir::Expression::name("lid")],
                ))],
            },
        );
    }
}

fn pack_of(element: Element) -> &'static str {
    match element {
        Element::Single => "run_pack_single",
        Element::Half => "run_pack_half",
        Element::Bfloat16 => "run_pack_bfloat16",
    }
}
