use neura_abi::Element;
use neura_compiler::{Compiler, ir};

pub fn define(compiler: &mut Compiler, elements: &[Element]) {
    super::convert_device::define(compiler);
    for element in elements.iter().copied().filter(|element| element.narrow()) {
        compiler.insert_case(
            "run_convert",
            ir::Arm {
                pattern: ir::Pattern::Integer(element.code()),
                body: vec![ir::Statement::Expression(ir::Expression::call(
                    convert_of(element),
                    vec![ir::Expression::name("task"), ir::Expression::name("lid")],
                ))],
            },
        );
    }
}

fn convert_of(element: Element) -> &'static str {
    match element {
        Element::Half => "run_convert_half",
        Element::Bfloat16 => "run_convert_bfloat16",
        Element::Single => panic!("a single precision tensor holds one element per word"),
    }
}
