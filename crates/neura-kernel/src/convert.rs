use neura_abi::Element;
use neura_compiler::{Compiler, ir};

pub fn define(compiler: &mut Compiler, elements: &[Element]) {
    super::convert_device::define(compiler);
    for element in elements
        .iter()
        .copied()
        .filter(|element| element.narrow() && !element.per_block())
    {
        let name = match element.format() {
            Some(format) => {
                let name = format!("run_convert_{}", element.name());
                compiler.specialize(
                    "template_run_convert_fp8",
                    &name,
                    &crate::element::format_constants(format),
                );
                name
            }
            None => convert_of(element).to_owned(),
        };
        compiler.insert_case(
            "run_convert",
            ir::Arm {
                pattern: ir::Pattern::Integer(element.code()),
                body: vec![ir::Statement::Expression(ir::Expression::call(
                    name,
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
        Element::Int8 => "run_convert_int8",
        Element::Single => panic!("a single precision tensor holds one element per word"),
        Element::Int4 => panic!("a block quantized tensor is written by its host"),
        Element::Fp8E4M3 | Element::Fp8E5M2 => panic!(
            "{} storage walks a float grid, and its convert is specialized from the grid the ABI declares",
            element.name(),
        ),
        Element::Fp4E2M1 => panic!("a block quantized tensor is written by its host"),
    }
}
