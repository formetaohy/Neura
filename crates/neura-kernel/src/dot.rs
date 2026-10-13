use neura_abi::Element;
use neura_compiler::{Compiler, ast};

const _: () = assert!(
    neura_profile::STREAMED_DEPTH == neura_abi::SUPER_SUB_BLOCK,
    "a block quantized operand shares its quanta over a run of the reduction, and a streamed tile walks one run",
);

pub(crate) fn install(compiler: &mut Compiler, elements: &[Element]) {
    device::define(compiler);
    for element in elements.iter().copied() {
        compiler.insert_case(
            "dot_by_element",
            ast::Arm {
                pattern: ast::Pattern::Integer(element.code()),
                body: vec![ast::Statement::Return(Some(ast::Expression::call(
                    dot_of(element),
                    dot_arguments(),
                )))],
            },
        );
    }
    compiler.select("dot_by_element", "dot");
}

fn dot_arguments() -> Vec<ast::Expression> {
    ["right", "left", "right_at", "left_at", "last", "valid"]
        .into_iter()
        .map(ast::Expression::name)
        .collect()
}

fn dot_of(element: Element) -> &'static str {
    match element {
        Element::Q8_0 => "dot_q8_0",
        Element::Q4K => "dot_q4_k",
        Element::Q5K => "dot_q5_k",
        Element::Q6K => "dot_q6_k",
        _ => "dot_element",
    }
}

#[neura_compiler::module]
mod device {
    fn dot_by_element(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        last: u32,
        valid: bool,
    ) -> f32 {
        match right.element {
            _ => {
                refuse(refusal::TENSOR, refusal::ELEMENT, right.element);
                return 0.0;
            }
        }
    }

    fn dot_element(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        last: u32,
        valid: bool,
    ) -> f32 {
        let mut total = 0.0;
        for at in unroll(0u32, MATMUL_RUN, 1u32) {
            let keep = valid && at < last;
            let right_address = select(0u32, right_at + at * right.strides.z, keep);
            let left_address = select(0u32, left_at + at * left.strides.w, keep);
            total = total
                + select(0.0, fetch(left, left_address), keep)
                    * select(0.0, fetch(right, right_address), keep);
        }
        return total;
    }

    fn dot_q8_0(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        last: u32,
        valid: bool,
    ) -> f32 {
        if right.strides.z == 1u32 && right_at % Q8_0_BLOCK == 0u32 {
            let base = (right_at / Q8_0_BLOCK) * Q8_0_BYTES;
            let scale = ggml_half(right, base);
            let mut total = 0.0;
            for word in unroll(0u32, SUPER_SUB_BLOCK / 4u32, 1u32) {
                let packed = ggml_word(right, base + 2u32 + word * 4u32);
                for lane in unroll(0u32, 4u32, 1u32) {
                    let at = word * 4u32 + lane;
                    let keep = valid && at < last;
                    let byte = (packed >> (lane * 8u32)) & 0xffu32;
                    let left_address = select(0u32, left_at + at * left.strides.w, keep);
                    total = total
                        + select(0.0, fetch(left, left_address), keep)
                            * select(0.0, ggml_signed(byte) * scale, keep);
                }
            }
            return total;
        }
        return dot_element(right, left, right_at, left_at, last, valid);
    }

    fn dot_q4_k(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        last: u32,
        valid: bool,
    ) -> f32 {
        if right.strides.z == 1u32 && right_at % SUPER_SUB_BLOCK == 0u32 {
            let base = (right_at / Q4_K_BLOCK) * Q4_K_BYTES;
            let within = right_at % Q4_K_BLOCK;
            let group = within / 64u32;
            let seat = (within % 64u32) / SUPER_SUB_BLOCK;
            let sub = group * 2u32 + seat;
            let shift = select(0u32, 4u32, seat == 1u32);
            let scale = ggml_half(right, base) * f32(ggml_k4_scale(right, base + 4u32, sub));
            let min = ggml_half(right, base + 2u32) * f32(ggml_k4_min(right, base + 4u32, sub));
            let mut total = 0.0;
            for word in unroll(0u32, SUPER_SUB_BLOCK / 4u32, 1u32) {
                let packed = bitcast_u32(peek(word_of(
                    right,
                    (base + 16u32 + group * 32u32 + word * 4u32) / 4u32,
                )));
                for lane in unroll(0u32, 4u32, 1u32) {
                    let at = word * 4u32 + lane;
                    let keep = valid && at < last;
                    let quant = (packed >> (lane * 8u32 + shift)) & 0xfu32;
                    let left_address = select(0u32, left_at + at * left.strides.w, keep);
                    total = total
                        + select(0.0, fetch(left, left_address), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                }
            }
            return total;
        }
        return dot_element(right, left, right_at, left_at, last, valid);
    }

    fn dot_q5_k(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        last: u32,
        valid: bool,
    ) -> f32 {
        if right.strides.z == 1u32 && right_at % SUPER_SUB_BLOCK == 0u32 {
            let base = (right_at / Q5_K_BLOCK) * Q5_K_BYTES;
            let within = right_at % Q5_K_BLOCK;
            let group = within / 64u32;
            let seat = (within % 64u32) / SUPER_SUB_BLOCK;
            let sub = group * 2u32 + seat;
            let nibble_shift = select(0u32, 4u32, seat == 1u32);
            let fifth_shift = 2u32 * group + seat;
            let scale = ggml_half(right, base) * f32(ggml_k4_scale(right, base + 4u32, sub));
            let min = ggml_half(right, base + 2u32) * f32(ggml_k4_min(right, base + 4u32, sub));
            let mut total = 0.0;
            for word in unroll(0u32, SUPER_SUB_BLOCK / 4u32, 1u32) {
                let packed = bitcast_u32(peek(word_of(
                    right,
                    (base + 48u32 + group * 32u32 + word * 4u32) / 4u32,
                )));
                let fifth = bitcast_u32(peek(word_of(right, (base + 16u32 + word * 4u32) / 4u32)));
                for lane in unroll(0u32, 4u32, 1u32) {
                    let at = word * 4u32 + lane;
                    let keep = valid && at < last;
                    let nibble = (packed >> (lane * 8u32 + nibble_shift)) & 0xfu32;
                    let bit = (fifth >> (lane * 8u32 + fifth_shift)) & 1u32;
                    let quant = nibble | (bit << 4u32);
                    let left_address = select(0u32, left_at + at * left.strides.w, keep);
                    total = total
                        + select(0.0, fetch(left, left_address), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                }
            }
            return total;
        }
        return dot_element(right, left, right_at, left_at, last, valid);
    }

    fn dot_q6_k(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        last: u32,
        valid: bool,
    ) -> f32 {
        if right.strides.z == 1u32 && right_at % SUPER_SUB_BLOCK == 0u32 {
            let base = (right_at / Q6_K_BLOCK) * Q6_K_BYTES;
            let within = right_at % Q6_K_BLOCK;
            let side = within / 128u32;
            let group = (within % 128u32) / SUPER_SUB_BLOCK;
            let nibble_shift = select(0u32, 4u32, group >= 2u32);
            let bit_shift = 2u32 * group;
            let byte = base + 192u32 + side * 8u32 + group * 2u32;
            let width = ggml_half(right, base + 208u32);
            let lower = width * ggml_signed(ggml_byte(right, byte));
            let upper = width * ggml_signed(ggml_byte(right, byte + 1u32));
            let mut total = 0.0;
            for word in unroll(0u32, SUPER_SUB_BLOCK / 4u32, 1u32) {
                let low = ggml_word(
                    right,
                    base + side * 64u32 + (group % 2u32) * 32u32 + word * 4u32,
                );
                let high = ggml_word(right, base + 128u32 + side * 32u32 + word * 4u32);
                for lane in unroll(0u32, 4u32, 1u32) {
                    let at = word * 4u32 + lane;
                    let keep = valid && at < last;
                    let nibble = (low >> (lane * 8u32 + nibble_shift)) & 0xfu32;
                    let bit = (high >> (lane * 8u32 + bit_shift)) & 3u32;
                    let quant = nibble | (bit << 4u32);
                    let weight = select(lower, upper, at >= SUPER_SMALL_BLOCK);
                    let left_address = select(0u32, left_at + at * left.strides.w, keep);
                    total = total
                        + select(0.0, fetch(left, left_address), keep)
                            * select(0.0, weight * (f32(quant) - 32.0), keep);
                }
            }
            return total;
        }
        return dot_element(right, left, right_at, left_at, last, valid);
    }
}
