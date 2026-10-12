use neura_abi::{Element, FloatFormat};
use neura_compiler::{Compiler, ast};

pub(crate) fn install(compiler: &mut Compiler, elements: &[Element]) {
    device::define(compiler);
    install_blocks(compiler);
    if matches!(elements, [Element::Single]) {
        compiler.select("fetch_single", "fetch");
        return;
    }
    for element in elements.iter().copied() {
        let name = match element.format() {
            Some(format) => {
                let name = format!("fetch_{}", element.name());
                let template = if element.sidecar() {
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
        Element::Q4_0 => "fetch_q4_0",
        Element::Q4_1 => "fetch_q4_1",
        Element::Q5_0 => "fetch_q5_0",
        Element::Q5_1 => "fetch_q5_1",
        Element::Q8_0 => "fetch_q8_0",
        Element::Q2K => "fetch_q2_k",
        Element::Q3K => "fetch_q3_k",
        Element::Q4K => "fetch_q4_k",
        Element::Q5K => "fetch_q5_k",
        Element::Q6K => "fetch_q6_k",
        Element::Mxfp4 => "fetch_mxfp4",
        Element::Fp8E4M3 | Element::Fp8E5M2 | Element::Fp4E2M1 => panic!(
            "{} storage walks a float grid, and its fetch is specialized from the grid the ABI declares",
            element.name(),
        ),
    }
}

fn install_blocks(compiler: &mut Compiler) {
    for element in Element::ALL {
        let Some(block) = element.block() else {
            continue;
        };
        if matches!(element, Element::Int4 | Element::Fp4E2M1) {
            continue;
        }
        let prefix = element.symbol().trim_start_matches("element::");
        compiler.constant(&format!("{prefix}_BLOCK"), block.elements);
        compiler.constant(&format!("{prefix}_BYTES"), block.bytes);
    }
    compiler.constant("SUPER_BLOCK", neura_abi::SUPER_BLOCK);
    compiler.constant("SUPER_SUB_BLOCK", neura_abi::SUPER_SUB_BLOCK);
    compiler.constant("SUPER_SMALL_BLOCK", neura_abi::SUPER_SMALL_BLOCK);
}

#[neura_compiler::module]
mod device {
    fn fetch_single(value: Value, at: u32) -> f32 {
        return peek(word_of(value, at));
    }

    fn fetch_half(value: Value, at: u32) -> f32 {
        let pair = unpack2x16float(bitcast_u32(peek(word_of(value, at >> 1u32))));
        return select(pair.x, pair.y, (at & 1u32) == 1u32);
    }

    fn fetch_bfloat16(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(peek(word_of(value, at >> 1u32)));
        return select(
            bitcast_f32(word << 16u32),
            bitcast_f32(word & 0xffff0000u32),
            (at & 1u32) == 1u32,
        );
    }

    fn fetch_int8(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(peek(word_of(value, at >> 2u32)));
        let byte = (word >> ((at & 3u32) * 8u32)) & 0xffu32;
        return (f32(byte) - select(0.0, 256.0, byte >= 128u32))
            * peek(word_of(value, value.table));
    }

    fn template_fetch_fp4(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(peek(word_of(value, at >> 3u32)));
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
        return decoded * peek(word_of(value, value.table + (at / FP4_BLOCK)));
    }

    fn fetch_int4(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(peek(word_of(value, at >> 3u32)));
        let nibble = (word >> ((at & 7u32) * 4u32)) & 0xfu32;
        return (f32(nibble) - select(0.0, 16.0, nibble >= 8u32))
            * peek(word_of(value, value.table + (at / INT4_BLOCK)));
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
        let word = bitcast_u32(peek(word_of(value, at >> 2u32)));
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

    fn ggml_byte(value: Value, at: u32) -> u32 {
        let word = bitcast_u32(peek(word_of(value, at >> 2u32)));
        return (word >> ((at & 3u32) * 8u32)) & 0xffu32;
    }

    fn ggml_word(value: Value, at: u32) -> u32 {
        return ggml_byte(value, at)
            | (ggml_byte(value, at + 1u32) << 8u32)
            | (ggml_byte(value, at + 2u32) << 16u32)
            | (ggml_byte(value, at + 3u32) << 24u32);
    }

    fn ggml_half(value: Value, at: u32) -> f32 {
        let pair = unpack2x16float(bitcast_u32(peek(word_of(value, at >> 2u32))));
        return select(pair.x, pair.y, (at & 3u32) == 2u32);
    }

    fn ggml_signed(byte: u32) -> f32 {
        return f32(i32(byte) - select(0i32, 256i32, byte >= 128u32));
    }

    fn ggml_k4_scale(value: Value, base: u32, at: u32) -> u32 {
        let upper = at >= 4u32;
        let seat = select(0u32, at - 4u32, upper);
        let low = select(
            ggml_byte(value, base + at) & 63u32,
            ggml_byte(value, base + at + 4u32) & 0xfu32,
            upper,
        );
        return low | select(0u32, (ggml_byte(value, base + seat) >> 6u32) << 4u32, upper);
    }

    fn ggml_k4_min(value: Value, base: u32, at: u32) -> u32 {
        let upper = at >= 4u32;
        let packed = ggml_byte(value, base + at + 4u32);
        let low = select(packed & 63u32, packed >> 4u32, upper);
        return low | select(0u32, (ggml_byte(value, base + at) >> 6u32) << 4u32, upper);
    }

    fn ggml_k3_code(value: Value, base: u32, at: u32) -> u32 {
        let upper = at >= 8u32;
        let seat = select(at, at - 8u32, upper);
        let packed = ggml_byte(value, base + 98u32 + seat);
        let nibble = select(packed & 0xfu32, packed >> 4u32, upper);
        let high = (ggml_byte(value, base + 106u32 + (at % 4u32)) >> ((at / 4u32) * 2u32)) & 3u32;
        return (high << 4u32) | nibble;
    }

    fn fetch_q4_0(value: Value, at: u32) -> f32 {
        let base = (at / Q4_0_BLOCK) * Q4_0_BYTES;
        let within = at % Q4_0_BLOCK;
        let packed = ggml_byte(value, base + 2u32 + (within % (Q4_0_BLOCK / 2u32)));
        let quant = select(
            packed & 0xfu32,
            packed >> 4u32,
            within >= (Q4_0_BLOCK / 2u32),
        );
        return (f32(quant) - 8.0) * ggml_half(value, base);
    }

    fn fetch_q4_1(value: Value, at: u32) -> f32 {
        let base = (at / Q4_1_BLOCK) * Q4_1_BYTES;
        let within = at % Q4_1_BLOCK;
        let packed = ggml_byte(value, base + 4u32 + (within % (Q4_1_BLOCK / 2u32)));
        let quant = select(
            packed & 0xfu32,
            packed >> 4u32,
            within >= (Q4_1_BLOCK / 2u32),
        );
        return f32(quant) * ggml_half(value, base) + ggml_half(value, base + 2u32);
    }

    fn fetch_q5_0(value: Value, at: u32) -> f32 {
        let base = (at / Q5_0_BLOCK) * Q5_0_BYTES;
        let within = at % Q5_0_BLOCK;
        let packed = ggml_byte(value, base + 6u32 + (within % (Q5_0_BLOCK / 2u32)));
        let nibble = select(
            packed & 0xfu32,
            packed >> 4u32,
            within >= (Q5_0_BLOCK / 2u32),
        );
        let quant = nibble | (((ggml_word(value, base + 2u32) >> within) & 1u32) << 4u32);
        return (f32(quant) - 16.0) * ggml_half(value, base);
    }

    fn fetch_q5_1(value: Value, at: u32) -> f32 {
        let base = (at / Q5_1_BLOCK) * Q5_1_BYTES;
        let within = at % Q5_1_BLOCK;
        let packed = ggml_byte(value, base + 8u32 + (within % (Q5_1_BLOCK / 2u32)));
        let nibble = select(
            packed & 0xfu32,
            packed >> 4u32,
            within >= (Q5_1_BLOCK / 2u32),
        );
        let quant = nibble | (((ggml_word(value, base + 4u32) >> within) & 1u32) << 4u32);
        return f32(quant) * ggml_half(value, base) + ggml_half(value, base + 2u32);
    }

    fn fetch_q8_0(value: Value, at: u32) -> f32 {
        let base = (at / Q8_0_BLOCK) * Q8_0_BYTES;
        return ggml_signed(ggml_byte(value, base + 2u32 + (at % Q8_0_BLOCK)))
            * ggml_half(value, base);
    }

    fn mxfp4_grid(code: u32) -> f32 {
        let magnitude = code & 7u32;
        let exponent = magnitude >> 1u32;
        let mantissa = magnitude & 1u32;
        let carried = bitcast_f32((exponent + 126u32) << 23u32) * (1.0 + 0.5 * f32(mantissa));
        let value = select(carried, f32(mantissa) * 0.5, exponent == 0u32);
        return select(-value, value, (code & 8u32) == 0u32);
    }

    fn fetch_mxfp4(value: Value, at: u32) -> f32 {
        let base = (at / MXFP4_BLOCK) * MXFP4_BYTES;
        let within = at % MXFP4_BLOCK;
        let packed = ggml_byte(value, base + 1u32 + (within % (MXFP4_BLOCK / 2u32)));
        let code = select(
            packed & 0xfu32,
            packed >> 4u32,
            within >= (MXFP4_BLOCK / 2u32),
        );
        let exponent = ggml_byte(value, base);
        let quantum = bitcast_f32(select(0x00400000u32, exponent << 23u32, exponent > 0u32));
        return mxfp4_grid(code) * quantum;
    }

    fn fetch_q2_k(value: Value, at: u32) -> f32 {
        let base = (at / Q2_K_BLOCK) * Q2_K_BYTES;
        let within = at % Q2_K_BLOCK;
        let half = within / 128u32;
        let seat = within % 128u32;
        let sub = seat / SUPER_SMALL_BLOCK;
        let lane = seat % SUPER_SMALL_BLOCK;
        let codes = ggml_byte(value, base + 4u32 + half * 8u32 + sub);
        let packed = ggml_byte(
            value,
            base + 20u32 + half * 32u32 + (sub % 2u32) * 16u32 + lane,
        );
        let quant = (packed >> ((sub / 2u32) * 2u32)) & 3u32;
        return ggml_half(value, base) * f32(codes & 0xfu32) * f32(quant)
            - ggml_half(value, base + 2u32) * f32(codes >> 4u32);
    }

    fn fetch_q3_k(value: Value, at: u32) -> f32 {
        let base = (at / Q3_K_BLOCK) * Q3_K_BYTES;
        let within = at % Q3_K_BLOCK;
        let half = within / 128u32;
        let seat = within % 128u32;
        let sub = seat / SUPER_SMALL_BLOCK;
        let lane = seat % SUPER_SMALL_BLOCK;
        let step = sub % 8u32;
        let packed = ggml_byte(
            value,
            base + 34u32 + half * 32u32 + (step % 2u32) * 16u32 + lane,
        );
        let quant = (packed >> ((step / 2u32) * 2u32)) & 3u32;
        let high = (ggml_byte(value, base + 2u32 + (within % 32u32)) >> (within / 32u32)) & 1u32;
        let factor =
            ggml_half(value, base) * (f32(ggml_k3_code(value, base, half * 8u32 + sub)) - 32.0);
        return factor * (f32(quant) - select(4.0, 0.0, high == 1u32));
    }

    fn fetch_q4_k(value: Value, at: u32) -> f32 {
        let base = (at / Q4_K_BLOCK) * Q4_K_BYTES;
        let within = at % Q4_K_BLOCK;
        let group = within / 64u32;
        let seat = (within % 64u32) / SUPER_SUB_BLOCK;
        let lane = within % SUPER_SUB_BLOCK;
        let sub = group * 2u32 + seat;
        let packed = ggml_byte(value, base + 16u32 + group * 32u32 + lane);
        let quant = select(packed & 0xfu32, packed >> 4u32, seat == 1u32);
        return ggml_half(value, base) * f32(ggml_k4_scale(value, base + 4u32, sub)) * f32(quant)
            - ggml_half(value, base + 2u32) * f32(ggml_k4_min(value, base + 4u32, sub));
    }

    fn fetch_q5_k(value: Value, at: u32) -> f32 {
        let base = (at / Q5_K_BLOCK) * Q5_K_BYTES;
        let within = at % Q5_K_BLOCK;
        let group = within / 64u32;
        let seat = (within % 64u32) / SUPER_SUB_BLOCK;
        let lane = within % SUPER_SUB_BLOCK;
        let sub = group * 2u32 + seat;
        let packed = ggml_byte(value, base + 48u32 + group * 32u32 + lane);
        let nibble = select(packed & 0xfu32, packed >> 4u32, seat == 1u32);
        let fifth = (ggml_byte(value, base + 16u32 + lane) >> (2u32 * group + seat)) & 1u32;
        let quant = nibble | (fifth << 4u32);
        return ggml_half(value, base) * f32(ggml_k4_scale(value, base + 4u32, sub)) * f32(quant)
            - ggml_half(value, base + 2u32) * f32(ggml_k4_min(value, base + 4u32, sub));
    }

    fn fetch_q6_k(value: Value, at: u32) -> f32 {
        let base = (at / Q6_K_BLOCK) * Q6_K_BYTES;
        let within = at % Q6_K_BLOCK;
        let half = within / 128u32;
        let seat = within % 128u32;
        let group = seat / SUPER_SUB_BLOCK;
        let lane = seat % SUPER_SUB_BLOCK;
        let packed = ggml_byte(value, base + half * 64u32 + (group % 2u32) * 32u32 + lane);
        let low = select(packed & 0xfu32, packed >> 4u32, group >= 2u32);
        let high = (ggml_byte(value, base + 128u32 + half * 32u32 + lane) >> (2u32 * group)) & 3u32;
        let quant = low | (high << 4u32);
        let scale = ggml_byte(
            value,
            base + 192u32 + half * 8u32 + group * 2u32 + lane / 16u32,
        );
        return ggml_half(value, base + 208u32) * ggml_signed(scale) * (f32(quant) - 32.0);
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
