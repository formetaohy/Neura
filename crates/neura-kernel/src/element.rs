use neura_abi::{Element, FloatFormat};
use neura_compiler::{Compiler, ast};

pub(crate) fn install(compiler: &mut Compiler, elements: &[Element]) {
    device::define(compiler);
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
            ast::Arm {
                pattern: ast::Pattern::Integer(element.code()),
                body: vec![ast::Statement::Return(Some(ast::Expression::call(
                    name,
                    vec![ast::Expression::name("value"), ast::Expression::name("at")],
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

#[neura_compiler::module]
mod device {
    fn fetch_single(value: Value, at: u32) -> f32 {
        return heap[word_of(value, at)];
    }

    fn fetch_half(value: Value, at: u32) -> f32 {
        let pair = unpack2x16float(bitcast_u32(heap[word_of(value, at >> 1u32)]));
        return select(pair.x, pair.y, (at & 1u32) == 1u32);
    }

    fn fetch_bfloat16(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(heap[word_of(value, at >> 1u32)]);
        return select(
            bitcast_f32(word << 16u32),
            bitcast_f32(word & 0xffff0000u32),
            (at & 1u32) == 1u32,
        );
    }

    fn fetch_int8(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(heap[word_of(value, at >> 2u32)]);
        let byte = (word >> ((at & 3u32) * 8u32)) & 0xffu32;
        return (f32(byte) - select(0.0, 256.0, byte >= 128u32))
            * heap[word_of(value, value.table)];
    }

    fn template_fetch_fp4(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(heap[word_of(value, at >> 3u32)]);
        let nibble = (word >> ((at & 7u32) * 4u32)) & 0xfu32;
        let code = ((nibble & 0x8u32) << 4u32) | (nibble & 0x7u32);
        let decoded = fp8_value(
            code,
            FP8_BIAS,
            FP8_MANTISSA_BITS,
            bitcast_f32(FP8_SMALLEST),
            FP8_NAN,
            FP8_INFINITY,
        );
        return decoded * heap[word_of(value, value.table + (at / FP4_BLOCK))];
    }

    fn fetch_int4(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(heap[word_of(value, at >> 3u32)]);
        let nibble = (word >> ((at & 7u32) * 4u32)) & 0xfu32;
        return (f32(nibble) - select(0.0, 16.0, nibble >= 8u32))
            * heap[word_of(value, value.table + (at / INT4_BLOCK))];
    }

    fn fp8_value(
        code: u32,
        bias: u32,
        mantissa_bits: u32,
        smallest: f32,
        nan: u32,
        infinity: u32,
    ) -> f32 {
        let body = code & 0x7fu32;
        let positive = (code & 0x80u32) == 0u32;
        if body > infinity || body == nan {
            return bitcast_f32(0x7fc00000u32);
        }
        if body == infinity {
            return select(
                -bitcast_f32(0x7f800000u32),
                bitcast_f32(0x7f800000u32),
                positive,
            );
        }
        let exponent = body >> mantissa_bits;
        let mantissa = body & ((1u32 << mantissa_bits) - 1u32);
        let magnitude = select(
            bitcast_f32(
                ((exponent + 127u32 - bias) << 23u32) | (mantissa << (23u32 - mantissa_bits)),
            ),
            f32(mantissa) * smallest,
            exponent == 0u32,
        );
        return select(-magnitude, magnitude, positive);
    }

    fn template_fetch_fp8(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(heap[word_of(value, at >> 2u32)]);
        let byte = (word >> ((at & 3u32) * 8u32)) & 0xffu32;
        return fp8_value(
            byte,
            FP8_BIAS,
            FP8_MANTISSA_BITS,
            bitcast_f32(FP8_SMALLEST),
            FP8_NAN,
            FP8_INFINITY,
        );
    }

    fn fetch_by_element(value: Value, at: u32) -> f32 {
        match value.element {
            _ => {
                refuse(refusal::TENSOR, refusal::ELEMENT, value.element);
                return 0.0;
            }
        }
    }
}
