use neura_abi::Element;
use neura_compiler::{Compiler, ast};

use crate::dot::symbol;

pub(crate) fn install(compiler: &mut Compiler, weights: &[Element]) {
    device::define(compiler);
    compiler.specialize("template_dot_quad_element", "dot_quad_element_staged", &[]);
    for element in [Element::Q3K, Element::Q4K, Element::Q5K, Element::Q6K] {
        compiler.specialize(
            &format!("template_dot_quad_{}", symbol(element)),
            &format!("dot_quad_{}_staged", symbol(element)),
            &[],
        );
    }
    compiler.select("generic_dot_quad_by_element", "dot_quad_by_element");
    for element in weights.iter().copied() {
        let target = match element {
            Element::Q3K | Element::Q4K | Element::Q5K | Element::Q6K => {
                format!("dot_quad_{}_staged", symbol(element))
            }
            _ => "dot_quad_element_staged".to_owned(),
        };
        compiler.insert_case(
            "dot_quad_by_element",
            ast::Arm {
                pattern: ast::Pattern::Integer(element.code()),
                body: vec![ast::Statement::Return(Some(ast::Expression::call(
                    target,
                    quad_arguments(),
                )))],
            },
        );
    }
    compiler.select("dot_quad_by_element", "dot_quad");
}

fn quad_arguments() -> Vec<ast::Expression> {
    [
        "right", "left", "right_at", "left_at", "left_at2", "left_at3", "left_at4", "last", "valid",
    ]
    .into_iter()
    .map(ast::Expression::name)
    .collect()
}

#[neura_compiler::module]
mod device {
    fn template_dot_quad_element(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        left_at2: u32,
        left_at3: u32,
        left_at4: u32,
        last: u32,
        valid: bool,
    ) -> Fvec4 {
        let mut total0 = 0.0;
        let mut total1 = 0.0;
        let mut total2 = 0.0;
        let mut total3 = 0.0;
        for at in unroll(0u32, MATMUL_RUN, 1u32) {
            let keep = valid && at < last;
            let right_address = select(0u32, right_at + at * right.strides.z, keep);
            let left_address = select(0u32, left_at + at * left.strides.w, keep);
            let left_address2 = select(0u32, left_at2 + at * left.strides.w, keep);
            let left_address3 = select(0u32, left_at3 + at * left.strides.w, keep);
            let left_address4 = select(0u32, left_at4 + at * left.strides.w, keep);
            total0 = total0
                + select(0.0, template_left(left, left_address), keep)
                    * select(0.0, fetch(right, right_address), keep);
            total1 = total1
                + select(0.0, template_left(left, left_address2), keep)
                    * select(0.0, fetch(right, right_address), keep);
            total2 = total2
                + select(0.0, template_left(left, left_address3), keep)
                    * select(0.0, fetch(right, right_address), keep);
            total3 = total3
                + select(0.0, template_left(left, left_address4), keep)
                    * select(0.0, fetch(right, right_address), keep);
        }
        return fvec4(total0, total1, total2, total3);
    }

    fn template_dot_quad_q3_k(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        left_at2: u32,
        left_at3: u32,
        left_at4: u32,
        last: u32,
        valid: bool,
    ) -> Fvec4 {
        if right.strides.z == 1u32 && right_at % SUPER_SUB_BLOCK == 0u32 {
            let base = (right_at / Q3_K_BLOCK) * Q3_K_BYTES;
            let within = right_at % Q3_K_BLOCK;
            let group = within / SUPER_SUB_BLOCK;
            let half = group / 4u32;
            let j = group % 4u32;
            let width = ggml_half(right, base + 108u32);
            let scale0 = width * ggml_q3_scale(right, base + 96u32, half * 8u32 + j * 2u32);
            let scale1 = width * ggml_q3_scale(right, base + 96u32, half * 8u32 + j * 2u32 + 1u32);
            let mut total0 = 0.0;
            let mut total1 = 0.0;
            let mut total2 = 0.0;
            let mut total3 = 0.0;
            for word in unroll(0u32, SUPER_SUB_BLOCK / 4u32, 1u32) {
                let quants = ggml_run_word(right, base + 32u32 + half * 32u32, word);
                let highs = ggml_run_word(right, base, word);
                for lane in unroll(0u32, 4u32, 1u32) {
                    let at = word * 4u32 + lane;
                    let keep = valid && at < last;
                    let low = (quants >> (lane * 8u32 + 2u32 * j)) & 3u32;
                    let high = (highs >> (lane * 8u32 + half * 4u32 + j)) & 1u32;
                    let scale = select(scale0, scale1, at >= 16u32);
                    let weight = scale * (f32(low) - select(0.0, 4.0, high == 0u32));
                    let left_address = select(0u32, left_at + at * left.strides.w, keep);
                    let left_address2 = select(0u32, left_at2 + at * left.strides.w, keep);
                    let left_address3 = select(0u32, left_at3 + at * left.strides.w, keep);
                    let left_address4 = select(0u32, left_at4 + at * left.strides.w, keep);
                    let kept = select(0.0, weight, keep);
                    total0 = total0 + select(0.0, template_left(left, left_address), keep) * kept;
                    total1 = total1 + select(0.0, template_left(left, left_address2), keep) * kept;
                    total2 = total2 + select(0.0, template_left(left, left_address3), keep) * kept;
                    total3 = total3 + select(0.0, template_left(left, left_address4), keep) * kept;
                }
            }
            return fvec4(total0, total1, total2, total3);
        }
        return template_dot_quad_element(
            right, left, right_at, left_at, left_at2, left_at3, left_at4, last, valid,
        );
    }

    fn template_dot_quad_q4_k(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        left_at2: u32,
        left_at3: u32,
        left_at4: u32,
        last: u32,
        valid: bool,
    ) -> Fvec4 {
        if right.strides.z == 1u32 && right_at % SUPER_SUB_BLOCK == 0u32 {
            let base = (right_at / Q4_K_BLOCK) * Q4_K_BYTES;
            let within = right_at % Q4_K_BLOCK;
            let group = within / 64u32;
            let seat = (within % 64u32) / SUPER_SUB_BLOCK;
            let sub = group * 2u32 + seat;
            let shift = select(0u32, 4u32, seat == 1u32);
            let scale = ggml_half(right, base) * f32(ggml_k4_scale(right, base + 4u32, sub));
            let min = ggml_half(right, base + 2u32) * f32(ggml_k4_min(right, base + 4u32, sub));
            let mut total0 = 0.0;
            let mut total1 = 0.0;
            let mut total2 = 0.0;
            let mut total3 = 0.0;
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
                    let left_address2 = select(0u32, left_at2 + at * left.strides.w, keep);
                    let left_address3 = select(0u32, left_at3 + at * left.strides.w, keep);
                    let left_address4 = select(0u32, left_at4 + at * left.strides.w, keep);
                    total0 = total0
                        + select(0.0, template_left(left, left_address), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                    total1 = total1
                        + select(0.0, template_left(left, left_address2), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                    total2 = total2
                        + select(0.0, template_left(left, left_address3), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                    total3 = total3
                        + select(0.0, template_left(left, left_address4), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                }
            }
            return fvec4(total0, total1, total2, total3);
        }
        return template_dot_quad_element(
            right, left, right_at, left_at, left_at2, left_at3, left_at4, last, valid,
        );
    }

    fn template_dot_quad_q5_k(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        left_at2: u32,
        left_at3: u32,
        left_at4: u32,
        last: u32,
        valid: bool,
    ) -> Fvec4 {
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
            let mut total0 = 0.0;
            let mut total1 = 0.0;
            let mut total2 = 0.0;
            let mut total3 = 0.0;
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
                    let left_address2 = select(0u32, left_at2 + at * left.strides.w, keep);
                    let left_address3 = select(0u32, left_at3 + at * left.strides.w, keep);
                    let left_address4 = select(0u32, left_at4 + at * left.strides.w, keep);
                    total0 = total0
                        + select(0.0, template_left(left, left_address), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                    total1 = total1
                        + select(0.0, template_left(left, left_address2), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                    total2 = total2
                        + select(0.0, template_left(left, left_address3), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                    total3 = total3
                        + select(0.0, template_left(left, left_address4), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                }
            }
            return fvec4(total0, total1, total2, total3);
        }
        return template_dot_quad_element(
            right, left, right_at, left_at, left_at2, left_at3, left_at4, last, valid,
        );
    }

    fn template_dot_quad_q6_k(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        left_at2: u32,
        left_at3: u32,
        left_at4: u32,
        last: u32,
        valid: bool,
    ) -> Fvec4 {
        if right.strides.z == 1u32 && right_at % SUPER_SUB_BLOCK == 0u32 {
            let base = (right_at / Q6_K_BLOCK) * Q6_K_BYTES;
            let within = right_at % Q6_K_BLOCK;
            let side = within / 128u32;
            let group = (within % 128u32) / SUPER_SUB_BLOCK;
            let nibble_shift = select(0u32, 4u32, group >= 2u32);
            let bit_shift = 2u32 * group;
            let scale = base + 192u32 + side * 8u32 + group * 2u32;
            let held_scale = ggml_raw(right, scale >> 2u32);
            let scale_half = (scale & 2u32) == 2u32;
            let held_width = ggml_raw(right, (base + 208u32) >> 2u32);
            let pair = unpack2x16float(held_width);
            let width = select(pair.x, pair.y, (base & 2u32) == 2u32);
            let lower = width
                * ggml_signed(select(
                    held_scale & 0xffu32,
                    (held_scale >> 16u32) & 0xffu32,
                    scale_half,
                ));
            let upper = width
                * ggml_signed(select(
                    (held_scale >> 8u32) & 0xffu32,
                    (held_scale >> 24u32) & 0xffu32,
                    scale_half,
                ));
            let low_start = (base + side * 64u32 + (group % 2u32) * 32u32) >> 2u32;
            let high_start = (base + 128u32 + side * 32u32) >> 2u32;
            let low0 = ggml_raw(right, low_start);
            let low1 = ggml_raw(right, low_start + 1u32);
            let low2 = ggml_raw(right, low_start + 2u32);
            let low3 = ggml_raw(right, low_start + 3u32);
            let low4 = ggml_raw(right, low_start + 4u32);
            let low5 = ggml_raw(right, low_start + 5u32);
            let low6 = ggml_raw(right, low_start + 6u32);
            let low7 = ggml_raw(right, low_start + 7u32);
            let low8 = ggml_raw(right, low_start + 8u32);
            let high0 = ggml_raw(right, high_start);
            let high1 = ggml_raw(right, high_start + 1u32);
            let high2 = ggml_raw(right, high_start + 2u32);
            let high3 = ggml_raw(right, high_start + 3u32);
            let high4 = ggml_raw(right, high_start + 4u32);
            let high5 = ggml_raw(right, high_start + 5u32);
            let high6 = ggml_raw(right, high_start + 6u32);
            let high7 = ggml_raw(right, high_start + 7u32);
            let high8 = ggml_raw(right, high_start + 8u32);
            let odd = (base & 2u32) == 2u32;
            let nibble0 = select(low0, (low0 >> 16u32) | (low1 << 16u32), odd);
            let nibble1 = select(low1, (low1 >> 16u32) | (low2 << 16u32), odd);
            let nibble2 = select(low2, (low2 >> 16u32) | (low3 << 16u32), odd);
            let nibble3 = select(low3, (low3 >> 16u32) | (low4 << 16u32), odd);
            let nibble4 = select(low4, (low4 >> 16u32) | (low5 << 16u32), odd);
            let nibble5 = select(low5, (low5 >> 16u32) | (low6 << 16u32), odd);
            let nibble6 = select(low6, (low6 >> 16u32) | (low7 << 16u32), odd);
            let nibble7 = select(low7, (low7 >> 16u32) | (low8 << 16u32), odd);
            let bits0 = select(high0, (high0 >> 16u32) | (high1 << 16u32), odd);
            let bits1 = select(high1, (high1 >> 16u32) | (high2 << 16u32), odd);
            let bits2 = select(high2, (high2 >> 16u32) | (high3 << 16u32), odd);
            let bits3 = select(high3, (high3 >> 16u32) | (high4 << 16u32), odd);
            let bits4 = select(high4, (high4 >> 16u32) | (high5 << 16u32), odd);
            let bits5 = select(high5, (high5 >> 16u32) | (high6 << 16u32), odd);
            let bits6 = select(high6, (high6 >> 16u32) | (high7 << 16u32), odd);
            let bits7 = select(high7, (high7 >> 16u32) | (high8 << 16u32), odd);
            let mut total0 = 0.0;
            let mut total1 = 0.0;
            let mut total2 = 0.0;
            let mut total3 = 0.0;
            for word in unroll(0u32, SUPER_SUB_BLOCK / 4u32, 1u32) {
                let low = select(
                    select(
                        select(
                            select(
                                select(
                                    select(
                                        select(nibble0, nibble1, word == 1u32),
                                        nibble2,
                                        word == 2u32,
                                    ),
                                    nibble3,
                                    word == 3u32,
                                ),
                                nibble4,
                                word == 4u32,
                            ),
                            nibble5,
                            word == 5u32,
                        ),
                        nibble6,
                        word == 6u32,
                    ),
                    nibble7,
                    word == 7u32,
                );
                let high = select(
                    select(
                        select(
                            select(
                                select(
                                    select(select(bits0, bits1, word == 1u32), bits2, word == 2u32),
                                    bits3,
                                    word == 3u32,
                                ),
                                bits4,
                                word == 4u32,
                            ),
                            bits5,
                            word == 5u32,
                        ),
                        bits6,
                        word == 6u32,
                    ),
                    bits7,
                    word == 7u32,
                );
                for lane in unroll(0u32, 4u32, 1u32) {
                    let at = word * 4u32 + lane;
                    let keep = valid && at < last;
                    let nibble = (low >> (lane * 8u32 + nibble_shift)) & 0xfu32;
                    let bit = (high >> (lane * 8u32 + bit_shift)) & 3u32;
                    let quant = nibble | (bit << 4u32);
                    let weight = select(lower, upper, at >= SUPER_SMALL_BLOCK);
                    let left_address = select(0u32, left_at + at * left.strides.w, keep);
                    let left_address2 = select(0u32, left_at2 + at * left.strides.w, keep);
                    let left_address3 = select(0u32, left_at3 + at * left.strides.w, keep);
                    let left_address4 = select(0u32, left_at4 + at * left.strides.w, keep);
                    total0 = total0
                        + select(0.0, template_left(left, left_address), keep)
                            * select(0.0, weight * (f32(quant) - 32.0), keep);
                    total1 = total1
                        + select(0.0, template_left(left, left_address2), keep)
                            * select(0.0, weight * (f32(quant) - 32.0), keep);
                    total2 = total2
                        + select(0.0, template_left(left, left_address3), keep)
                            * select(0.0, weight * (f32(quant) - 32.0), keep);
                    total3 = total3
                        + select(0.0, template_left(left, left_address4), keep)
                            * select(0.0, weight * (f32(quant) - 32.0), keep);
                }
            }
            return fvec4(total0, total1, total2, total3);
        }
        return template_dot_quad_element(
            right, left, right_at, left_at, left_at2, left_at3, left_at4, last, valid,
        );
    }

    fn generic_dot_quad_by_element(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        left_at2: u32,
        left_at3: u32,
        left_at4: u32,
        last: u32,
        valid: bool,
    ) -> Fvec4 {
        match right.element {
            _ => {
                refuse(refusal::TENSOR, refusal::ELEMENT, right.element);
                return fvec4(0.0, 0.0, 0.0, 0.0);
            }
        }
    }
}
