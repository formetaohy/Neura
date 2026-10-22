use neura_abi::Element;
use neura_compiler::{Compiler, ast};

const _: () = assert!(
    neura_profile::STREAMED_DEPTH == neura_abi::SUPER_SUB_BLOCK,
    "a block quantized operand shares its quanta over a run of the reduction, and a streamed tile walks one run",
);
const _: () = assert!(
    neura_profile::FOLDED_LANES == neura_profile::STREAMED_DEPTH,
    "a folded tile hands one lane every number of the run a streamed tile walks",
);

pub(crate) const HEAP: &str = "heap";
pub(crate) const STAGED: &str = "staged";
pub(crate) const RUN: &str = "run";
pub(crate) const STAGES_A_RUN: Element = Element::Q8_0;

const _: () = assert!(
    STAGES_A_RUN.stages_a_run(),
    "a run stages the blocks of an element whose blocks the device stages",
);

pub(crate) fn install(compiler: &mut Compiler, weights: &[Element]) {
    device::define(compiler);
    left::define(compiler);
    run::define(compiler);
    panel::define(compiler);
    compiler.constant("RUN_CODES", neura_profile::RUN_CODES);
    compiler.constant("RUN_SCALES", neura_profile::RUN_SCALES);
    compiler.constant("RUN_OFFSET", neura_profile::RUN_OFFSET);
    compiler.constant("RUN_SLOT", neura_profile::RUN_SLOT);
    for generic in [
        "template_dot_element",
        "template_dot_q8_0",
        "template_dot_q4_k",
        "template_dot_q5_k",
        "template_dot_q6_k",
    ] {
        let name = generic.trim_start_matches("template_");
        for family in [HEAP, STAGED] {
            compiler.specialize(generic, &format!("{name}_{family}"), &[]);
        }
    }
    compiler.specialize("template_dot_q8_0_run", "dot_q8_0_run", &[]);
    compiler.select("generic_dot_by_element", "dot_by_element");
    compiler.select("generic_dot_by_element", "dot_staged_by_element");
    compiler.select("generic_dot_by_element", "dot_run_by_element");
    dispatch(compiler, weights, "dot_by_element", HEAP);
    dispatch(compiler, weights, "dot_staged_by_element", STAGED);
    compiler.insert_case(
        "dot_run_by_element",
        ast::Arm {
            pattern: ast::Pattern::Integer(STAGES_A_RUN.code()),
            body: vec![ast::Statement::Return(Some(ast::Expression::call(
                format!("dot_{}_{RUN}", STAGES_A_RUN.name()),
                dot_arguments(),
            )))],
        },
    );
    compiler.select("dot_by_element", "dot");
    compiler.select("dot_staged_by_element", "dot_staged");
    compiler.select("dot_run_by_element", "dot_run");
    compiler.select("generic_stage_by_element", "stage_by_element");
    compiler.insert_case(
        "stage_by_element",
        ast::Arm {
            pattern: ast::Pattern::Integer(STAGES_A_RUN.code()),
            body: vec![ast::Statement::Return(Some(ast::Expression::call(
                format!("stage_{}", STAGES_A_RUN.name()),
                stage_arguments(),
            )))],
        },
    );
    compiler.select("stage_by_element", "stage");
    compiler.select("stage_none", "stage_untouched");
    compiler.select("generic_codes_by_element", "codes_by_element");
    for element in weights.iter().copied().filter(|held| held.packs_codes()) {
        compiler.insert_case(
            "codes_by_element",
            ast::Arm {
                pattern: ast::Pattern::Integer(element.code()),
                body: vec![ast::Statement::Expression(ast::Expression::call(
                    format!("codes_{}", symbol(element)),
                    ["right", "address", "inside", "slot"]
                        .into_iter()
                        .map(ast::Expression::name)
                        .collect(),
                ))],
            },
        );
    }
    compiler.select("codes_by_element", "codes");
    crate::dot_pair::install(compiler, weights);
}

fn dispatch(compiler: &mut Compiler, weights: &[Element], by_element: &str, family: &str) {
    for element in weights.iter().copied() {
        compiler.insert_case(
            by_element,
            ast::Arm {
                pattern: ast::Pattern::Integer(element.code()),
                body: vec![ast::Statement::Return(Some(ast::Expression::call(
                    dot_of(element, family),
                    dot_arguments(),
                )))],
            },
        );
    }
}

fn stage_arguments() -> Vec<ast::Expression> {
    ["right", "run_at", "blocks", "warp", "lane"]
        .into_iter()
        .map(ast::Expression::name)
        .collect()
}

fn dot_arguments() -> Vec<ast::Expression> {
    ["right", "left", "right_at", "left_at", "last", "valid"]
        .into_iter()
        .map(ast::Expression::name)
        .collect()
}

pub(crate) fn symbol(element: Element) -> &'static str {
    match element {
        Element::Q8_0 => "q8_0",
        Element::Q4K => "q4_k",
        Element::Q5K => "q5_k",
        Element::Q6K => "q6_k",
        _ => "element",
    }
}

fn dot_of(element: Element, family: &str) -> String {
    format!("dot_{}_{family}", symbol(element))
}

#[neura_compiler::module]
mod left {
    fn left_heap(left: Value, address: u32) -> f32 {
        return fetch(left, address);
    }

    fn left_staged(left: Value, address: u32) -> f32 {
        return scratch[address];
    }
}

#[neura_compiler::module]
mod run {
    fn left_run(left: Value, address: u32) -> f32 {
        return scratch[address];
    }

    fn stage_none(right: Value, run_at: u32, blocks: u32, warp: u32, lane: u32) -> u32 {
        return run_at;
    }

    fn generic_stage_by_element(
        right: Value,
        run_at: u32,
        blocks: u32,
        warp: u32,
        lane: u32,
    ) -> u32 {
        match right.element {
            _ => {
                refuse(refusal::TENSOR, refusal::ELEMENT, right.element);
                return 0u32;
            }
        }
    }

    fn stage_q8_0(right: Value, run_at: u32, blocks: u32, warp: u32, lane: u32) -> u32 {
        let run = ((run_at / Q8_0_BLOCK) - lane) * Q8_0_BYTES;
        let words = ((blocks * Q8_0_BYTES) + 3u32) / 4u32;
        let slot = SCRATCH_MATMUL_RUN + warp * FOLDED_RUN;
        for round in unroll(
            0u32,
            (FOLDED_RUN + FOLDED_LANES - 1u32) / FOLDED_LANES,
            1u32,
        ) {
            let index = round * FOLDED_LANES + lane;
            if index < FOLDED_RUN {
                let word = select(run >> 2u32, (run >> 2u32) + index, index < words);
                scratch[slot + index] = bitcast_f32(bitcast_u32(peek(word_of(right, word))));
            }
        }
        return 4u32 * slot + (run & 3u32) + Q8_0_BYTES * lane;
    }

    fn template_dot_q8_0_run(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        last: u32,
        valid: bool,
    ) -> f32 {
        let start = right_at >> 2u32;
        let held0 = bitcast_u32(scratch[start]);
        let held1 = bitcast_u32(scratch[start + 1u32]);
        let held2 = bitcast_u32(scratch[start + 2u32]);
        let held3 = bitcast_u32(scratch[start + 3u32]);
        let held4 = bitcast_u32(scratch[start + 4u32]);
        let held5 = bitcast_u32(scratch[start + 5u32]);
        let held6 = bitcast_u32(scratch[start + 6u32]);
        let held7 = bitcast_u32(scratch[start + 7u32]);
        let held8 = bitcast_u32(scratch[start + 8u32]);
        let pair = unpack2x16float(held0);
        let scale = select(pair.x, pair.y, (right_at & 2u32) == 2u32);
        let odd = (right_at & 2u32) == 0u32;
        let payload0 = select(held1, (held0 >> 16u32) | (held1 << 16u32), odd);
        let payload1 = select(held2, (held1 >> 16u32) | (held2 << 16u32), odd);
        let payload2 = select(held3, (held2 >> 16u32) | (held3 << 16u32), odd);
        let payload3 = select(held4, (held3 >> 16u32) | (held4 << 16u32), odd);
        let payload4 = select(held5, (held4 >> 16u32) | (held5 << 16u32), odd);
        let payload5 = select(held6, (held5 >> 16u32) | (held6 << 16u32), odd);
        let payload6 = select(held7, (held6 >> 16u32) | (held7 << 16u32), odd);
        let payload7 = select(held8, (held7 >> 16u32) | (held8 << 16u32), odd);
        let mut total = 0.0;
        for word in unroll(0u32, SUPER_SUB_BLOCK / 4u32, 1u32) {
            let worded = select(
                select(
                    select(
                        select(
                            select(
                                select(
                                    select(payload0, payload1, word == 1u32),
                                    payload2,
                                    word == 2u32,
                                ),
                                payload3,
                                word == 3u32,
                            ),
                            payload4,
                            word == 4u32,
                        ),
                        payload5,
                        word == 5u32,
                    ),
                    payload6,
                    word == 6u32,
                ),
                payload7,
                word == 7u32,
            );
            for lane in unroll(0u32, 4u32, 1u32) {
                let at = word * 4u32 + lane;
                let keep = valid && at < last;
                let byte = (worded >> (lane * 8u32)) & 0xffu32;
                let left_address = select(0u32, left_at + at * left.strides.w, keep);
                total = total
                    + select(0.0, template_left(left, left_address), keep)
                        * select(0.0, ggml_signed(byte) * scale, keep);
            }
        }
        return total;
    }
}

#[neura_compiler::module]
mod panel {
    fn generic_codes_by_element(right: Value, address: u32, inside: bool, slot: u32) {
        match right.element {
            _ => refuse(refusal::TENSOR, refusal::ELEMENT, right.element),
        }
    }

    fn panel_affine(slot: u32, low_scale: f32, high_scale: f32, offset: f32, inside: bool) {
        scratch[slot + RUN_CODES] = select(0.0, low_scale, inside);
        scratch[slot + RUN_CODES + 1u32] = select(0.0, high_scale, inside);
        scratch[slot + RUN_CODES + RUN_SCALES] = select(0.0, offset, inside);
    }

    fn panel_word(right: Value, index: u32, odd: bool) -> u32 {
        let low = ggml_raw(right, index);
        let high = ggml_raw(right, index + 1u32);
        return select(low, (low >> 16u32) | (high << 16u32), odd);
    }

    fn codes_q8_0(right: Value, address: u32, inside: bool, slot: u32) {
        let byte = (address / Q8_0_BLOCK) * Q8_0_BYTES;
        let scale = ggml_half(right, byte);
        for word in unroll(0u32, RUN_CODES, 1u32) {
            let at = byte + 2u32 + word * 4u32;
            let held = panel_word(right, at >> 2u32, (at & 3u32) == 2u32);
            scratch[slot + word] = select(0.0, bitcast_f32(held), inside);
        }
        panel_affine(slot, scale, scale, 0.0, inside);
    }

    fn codes_q4_k(right: Value, address: u32, inside: bool, slot: u32) {
        let base = (address / Q4_K_BLOCK) * Q4_K_BYTES;
        let within = address % Q4_K_BLOCK;
        let group = within / 64u32;
        let seat = (within % 64u32) / SUPER_SUB_BLOCK;
        let sub = group * 2u32 + seat;
        let quantum = ggml_half(right, base);
        let floor = ggml_half(right, base + 2u32);
        let scale = quantum * f32(ggml_k4_scale(right, base + 4u32, sub));
        let offset = 0.0 - floor * f32(ggml_k4_min(right, base + 4u32, sub));
        for word in unroll(0u32, RUN_CODES, 1u32) {
            let packed = ggml_raw(right, base / 4u32 + 4u32 + group * 8u32 + word);
            let mut held = 0u32;
            for lane in unroll(0u32, 4u32, 1u32) {
                held = held | (((packed >> (lane * 8u32 + seat * 4u32)) & 0xfu32) << (lane * 8u32));
            }
            scratch[slot + word] = select(0.0, bitcast_f32(held), inside);
        }
        panel_affine(slot, scale, scale, offset, inside);
    }

    fn codes_q5_k(right: Value, address: u32, inside: bool, slot: u32) {
        let base = (address / Q5_K_BLOCK) * Q5_K_BYTES;
        let within = address % Q5_K_BLOCK;
        let group = within / 64u32;
        let seat = (within % 64u32) / SUPER_SUB_BLOCK;
        let sub = group * 2u32 + seat;
        let fifth_shift = 2u32 * group + seat;
        let quantum = ggml_half(right, base);
        let floor = ggml_half(right, base + 2u32);
        let scale = quantum * f32(ggml_k4_scale(right, base + 4u32, sub));
        let offset = 0.0 - floor * f32(ggml_k4_min(right, base + 4u32, sub));
        for word in unroll(0u32, RUN_CODES, 1u32) {
            let packed = ggml_raw(right, base / 4u32 + 12u32 + group * 8u32 + word);
            let fifth = ggml_raw(right, base / 4u32 + 4u32 + word);
            let mut held = 0u32;
            for lane in unroll(0u32, 4u32, 1u32) {
                let nibble = (packed >> (lane * 8u32 + seat * 4u32)) & 0xfu32;
                let bit = (fifth >> (lane * 8u32 + fifth_shift)) & 1u32;
                held = held | ((nibble | (bit << 4u32)) << (lane * 8u32));
            }
            scratch[slot + word] = select(0.0, bitcast_f32(held), inside);
        }
        panel_affine(slot, scale, scale, offset, inside);
    }

    fn codes_q6_k(right: Value, address: u32, inside: bool, slot: u32) {
        let base = (address / Q6_K_BLOCK) * Q6_K_BYTES;
        let within = address % Q6_K_BLOCK;
        let side = within / 128u32;
        let group = (within % 128u32) / SUPER_SUB_BLOCK;
        let nibble_shift = select(0u32, 4u32, group >= 2u32);
        let bit_shift = 2u32 * group;
        let odd = (base & 2u32) == 2u32;
        let scale_at = base + 192u32 + side * 8u32 + group * 2u32;
        let held_scale = ggml_raw(right, scale_at >> 2u32);
        let scale_half = (scale_at & 2u32) == 2u32;
        let width = ggml_half(right, base + 208u32);
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
        for word in unroll(0u32, RUN_CODES, 1u32) {
            let low = panel_word(right, low_start + word, odd);
            let high = panel_word(right, high_start + word, odd);
            let mut held = 0u32;
            for lane in unroll(0u32, 4u32, 1u32) {
                let nibble = (low >> (lane * 8u32 + nibble_shift)) & 0xfu32;
                let bit = (high >> (lane * 8u32 + bit_shift)) & 3u32;
                let biased = (nibble | (bit << 4u32)) + 224u32;
                held = held | (biased & 0xffu32) << (lane * 8u32);
            }
            scratch[slot + word] = select(0.0, bitcast_f32(held), inside);
        }
        panel_affine(slot, lower, upper, 0.0, inside);
    }

    fn run_codes(slot: u32, within: u32) -> f32 {
        let held = bitcast_u32(scratch[slot + within / 4u32]);
        let code = (held >> ((within % 4u32) * 8u32)) & 0xffu32;
        let scale = scratch[slot + RUN_CODES + within / 16u32];
        return ggml_signed(code) * scale + scratch[slot + RUN_CODES + RUN_SCALES];
    }
}

#[neura_compiler::module]
mod device {

    fn generic_dot_by_element(
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

    fn template_dot_element(
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
                + select(0.0, template_left(left, left_address), keep)
                    * select(0.0, fetch(right, right_address), keep);
        }
        return total;
    }

    fn template_dot_q8_0(
        right: Value,
        left: Value,
        right_at: u32,
        left_at: u32,
        last: u32,
        valid: bool,
    ) -> f32 {
        if right.strides.z == 1u32 && right_at % Q8_0_BLOCK == 0u32 {
            let base = (right_at / Q8_0_BLOCK) * Q8_0_BYTES;
            let start = base >> 2u32;
            let held0 = ggml_raw(right, start);
            let held1 = ggml_raw(right, start + 1u32);
            let held2 = ggml_raw(right, start + 2u32);
            let held3 = ggml_raw(right, start + 3u32);
            let held4 = ggml_raw(right, start + 4u32);
            let held5 = ggml_raw(right, start + 5u32);
            let held6 = ggml_raw(right, start + 6u32);
            let held7 = ggml_raw(right, start + 7u32);
            let held8 = ggml_raw(right, start + 8u32);
            let pair = unpack2x16float(held0);
            let scale = select(pair.x, pair.y, (base & 2u32) == 2u32);
            let odd = (base & 2u32) == 0u32;
            let payload0 = select(held1, (held0 >> 16u32) | (held1 << 16u32), odd);
            let payload1 = select(held2, (held1 >> 16u32) | (held2 << 16u32), odd);
            let payload2 = select(held3, (held2 >> 16u32) | (held3 << 16u32), odd);
            let payload3 = select(held4, (held3 >> 16u32) | (held4 << 16u32), odd);
            let payload4 = select(held5, (held4 >> 16u32) | (held5 << 16u32), odd);
            let payload5 = select(held6, (held5 >> 16u32) | (held6 << 16u32), odd);
            let payload6 = select(held7, (held6 >> 16u32) | (held7 << 16u32), odd);
            let payload7 = select(held8, (held7 >> 16u32) | (held8 << 16u32), odd);
            let mut total = 0.0;
            for word in unroll(0u32, SUPER_SUB_BLOCK / 4u32, 1u32) {
                let worded = select(
                    select(
                        select(
                            select(
                                select(
                                    select(
                                        select(payload0, payload1, word == 1u32),
                                        payload2,
                                        word == 2u32,
                                    ),
                                    payload3,
                                    word == 3u32,
                                ),
                                payload4,
                                word == 4u32,
                            ),
                            payload5,
                            word == 5u32,
                        ),
                        payload6,
                        word == 6u32,
                    ),
                    payload7,
                    word == 7u32,
                );
                for lane in unroll(0u32, 4u32, 1u32) {
                    let at = word * 4u32 + lane;
                    let keep = valid && at < last;
                    let byte = (worded >> (lane * 8u32)) & 0xffu32;
                    let left_address = select(0u32, left_at + at * left.strides.w, keep);
                    total = total
                        + select(0.0, template_left(left, left_address), keep)
                            * select(0.0, ggml_signed(byte) * scale, keep);
                }
            }
            return total;
        }
        return template_dot_element(right, left, right_at, left_at, last, valid);
    }

    fn template_dot_q4_k(
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
                        + select(0.0, template_left(left, left_address), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                }
            }
            return total;
        }
        return template_dot_element(right, left, right_at, left_at, last, valid);
    }

    fn template_dot_q5_k(
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
                        + select(0.0, template_left(left, left_address), keep)
                            * select(0.0, scale * f32(quant) - min, keep);
                }
            }
            return total;
        }
        return template_dot_element(right, left, right_at, left_at, last, valid);
    }

    fn template_dot_q6_k(
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
            let mut total = 0.0;
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
                    total = total
                        + select(0.0, template_left(left, left_address), keep)
                            * select(0.0, weight * (f32(quant) - 32.0), keep);
                }
            }
            return total;
        }
        return template_dot_element(right, left, right_at, left_at, last, valid);
    }
}
