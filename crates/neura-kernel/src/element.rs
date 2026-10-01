use neura_abi::{Element, FloatFormat};
use neura_compiler::{Compiler, ir};

pub fn define(compiler: &mut Compiler, elements: &[Element]) {
    if matches!(elements, [Element::Single]) {
        compiler.select("fetch_single", "fetch");
        return;
    }
    for element in elements.iter().copied() {
        let name = match element.format() {
            Some(format) => {
                let name = format!("fetch_{}", element.name());
                let template = if element.per_block() {
                    "template_fetch_fp4"
                } else {
                    "template_fetch_fp8"
                };
                compiler.specialize(template, &name, &format_constants(format));
                name
            }
            None => fetch_of(element).to_owned(),
        };
        compiler.insert_case(
            "fetch_by_element",
            ir::Arm {
                pattern: ir::Pattern::Integer(element.code()),
                body: vec![ir::Statement::Return(Some(ir::Expression::call(
                    name,
                    vec![ir::Expression::name("value"), ir::Expression::name("at")],
                )))],
            },
        );
    }
    compiler.select("fetch_by_element", "fetch");
}

pub(crate) fn format_constants(format: FloatFormat) -> [(&'static str, u32); 8] {
    [
        ("FP8_BIAS", format.bias as u32),
        ("FP8_MANTISSA_BITS", format.mantissa_bits),
        ("FP8_SUBNORMAL_SHIFT", format.subnormal_shift as u32),
        ("FP8_SMALLEST", format.smallest.to_bits()),
        ("FP8_NAN", format.nan),
        ("FP8_INFINITY", format.infinity),
        ("FP8_MAX", format.max),
        ("FP8_CEILING", format.ceiling),
    ]
}

fn fetch_of(element: Element) -> &'static str {
    match element {
        Element::Single => "fetch_single",
        Element::Half => "fetch_half",
        Element::Bfloat16 => "fetch_bfloat16",
        Element::Int8 => "fetch_int8",
        Element::Int4 => "fetch_int4",
        Element::Fp8E4M3 | Element::Fp8E5M2 | Element::Fp4E2M1 => panic!(
            "{} storage walks a float grid, and its fetch is specialized from the grid the ABI declares",
            element.name(),
        ),
    }
}
