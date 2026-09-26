use neura_abi::Element;
use neura_compiler::{Compiler, ir};

pub fn define(compiler: &mut Compiler, elements: &[Element]) {
    if matches!(elements, [Element::Single]) {
        compiler.select("fetch_single", "fetch");
        return;
    }
    for element in elements.iter().copied() {
        compiler.insert_case(
            "fetch_by_element",
            ir::Arm {
                pattern: ir::Pattern::Integer(element.code()),
                body: vec![ir::Statement::Return(Some(ir::Expression::call(
                    fetch_of(element),
                    vec![ir::Expression::name("value"), ir::Expression::name("at")],
                )))],
            },
        );
    }
    compiler.select("fetch_by_element", "fetch");
}

fn fetch_of(element: Element) -> &'static str {
    match element {
        Element::Single => "fetch_single",
        Element::Half => "fetch_half",
        Element::Bfloat16 => "fetch_bfloat16",
    }
}
