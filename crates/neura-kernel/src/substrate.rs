use crate::Banks;
use neura_compiler::{Compiler, ast};

pub(crate) fn install(compiler: &mut Compiler, banks: Banks, paged: bool) {
    device::define(compiler);
    compiler.select(
        if paged {
            "word_of_paged"
        } else {
            "word_of_direct"
        },
        "word_of",
    );
    if banks.count() == 1 {
        compiler.select("peek_single", "peek");
        compiler.select("poke_single", "poke");
        return;
    }
    let shift = banks.shift();
    let mask = (1u32 << shift) - 1;
    compiler.specialize("template_peek", "peek_by_bank", &[("BANK_SHIFT", shift)]);
    compiler.specialize("template_poke", "poke_by_bank", &[("BANK_SHIFT", shift)]);
    for bank in 0..banks.count() {
        compiler.insert_case("peek_by_bank", peek_case(bank, mask));
        compiler.insert_case("poke_by_bank", poke_case(bank, mask));
    }
    compiler.select("peek_by_bank", "peek");
    compiler.select("poke_by_bank", "poke");
}

fn bank_offset(mask: u32) -> ast::Expression {
    ast::Expression::Binary {
        op: ast::BinaryOperator::BitAnd,
        left: Box::new(ast::Expression::name("address")),
        right: Box::new(ast::Expression::u32(mask)),
    }
}

fn heap_word(bank: u32, mask: u32) -> ast::Expression {
    ast::Expression::Index {
        base: Box::new(ast::Expression::name(format!("heap{bank}"))),
        index: Box::new(bank_offset(mask)),
    }
}

fn peek_case(bank: u32, mask: u32) -> ast::Arm {
    ast::Arm {
        pattern: ast::Pattern::Integer(bank),
        body: vec![ast::Statement::Return(Some(heap_word(bank, mask)))],
    }
}

fn poke_case(bank: u32, mask: u32) -> ast::Arm {
    ast::Arm {
        pattern: ast::Pattern::Integer(bank),
        body: vec![ast::Statement::Assign {
            place: heap_word(bank, mask),
            value: ast::Expression::name("data"),
            operator: None,
        }],
    }
}

#[neura_compiler::module]
mod device {
    fn peek_single(address: u32) -> f32 {
        return heap0[address];
    }

    fn poke_single(address: u32, data: f32) {
        heap0[address] = data;
    }

    fn template_peek(address: u32) -> f32 {
        let bank = address >> BANK_SHIFT;
        match bank {
            _ => {
                refuse(refusal::TENSOR, refusal::INDEX, 0u32);
                return 0.0;
            }
        }
    }

    fn template_poke(address: u32, data: f32) {
        let bank = address >> BANK_SHIFT;
        match bank {
            _ => {
                refuse(refusal::TENSOR, refusal::INDEX, 0u32);
            }
        }
    }

    fn refuse(subject: u32, category: u32, code: u32) {
        if atomic_add(&state[control::REFUSAL], 0u32) == 0u32 {
            atomic_store(
                &state[control::REFUSAL],
                (subject << refusal::KIND_BITS) | (category << refusal::CODE_BITS) | (code + 1u32),
            );
        }
    }

    fn max_identity() -> f32 {
        return -bitcast_f32(0x7f800000u32);
    }

    fn exp_underflow() -> f32 {
        return -104.0;
    }

    fn softmax_exp(argument: f32) -> f32 {
        return exp(max(argument, exp_underflow()));
    }

    fn positive_log(value: f32) -> f32 {
        let mut logarithm = max_identity();
        if value > 0.0 {
            logarithm = log(value);
        }
        return logarithm;
    }

    fn slot_of(packed: u32, axis: u32) -> u32 {
        return (packed >> (axis * 8u32)) & 0xffu32;
    }

    fn coordinates(flat: u32, dims: uvec4) -> uvec4 {
        if dims.x == 0u32 || dims.y == 0u32 || dims.z == 0u32 || dims.w == 0u32 {
            refuse(refusal::TENSOR, refusal::EMPTY, 0u32);
            return uvec4(0u32, 0u32, 0u32, 0u32);
        }
        let w = flat % dims.w;
        let z = flat / dims.w % dims.z;
        let y = flat / (dims.w * dims.z) % dims.y;
        let x = flat / (dims.w * dims.z * dims.y);
        return uvec4(x, y, z, w);
    }

    fn read_address(at: uvec4, strides: uvec4) -> u32 {
        return at.x * strides.x + at.y * strides.y + at.z * strides.z + at.w * strides.w;
    }

    fn walked_at(mode: u32, index: u32, dims: uvec4) -> uvec4 {
        if mode == strategy::INDEX {
            return uvec4(0u32, 0u32, 0u32, index);
        }
        return coordinates(index, dims);
    }

    fn component(at: uvec4, axis: u32) -> u32 {
        return select(
            select(at.x, at.y, axis == 1u32),
            select(at.z, at.w, axis == 3u32),
            axis >= 2u32,
        );
    }

    fn whole_index(value: f32, rows: u32, subject: u32, category: u32) -> u32 {
        if trunc(value) != value || !(value >= 0.0) || !(value < f32(rows)) {
            refuse(subject, category, 0u32);
            return 0u32;
        }
        return u32(value);
    }

    fn axis_start(offsets: Value, plane: u32, bound: u32, subject: u32) -> u32 {
        return whole_index(fetch(offsets, plane), bound, subject, refusal::INDEX);
    }

    fn axis_rows(offsets: Value, plane: u32, bound: u32, subject: u32) -> u32 {
        let start = axis_start(offsets, plane, bound, subject);
        let end = whole_index(
            fetch(offsets, plane + 1u32),
            bound,
            subject,
            refusal::EXTENT,
        );
        if end < start {
            refuse(subject, refusal::EXTENT, 0u32);
            return 0u32;
        }
        return end - start;
    }

    fn base_of(value: Value) -> u32 {
        return select(
            tables[PLACEMENT_FIRST + PLACEMENT_WEIGHTS],
            tables[PLACEMENT_FIRST + PLACEMENT_TENSORS],
            value.store == store::TENSORS,
        );
    }

    fn word_of_direct(value: Value, word: u32) -> u32 {
        return base_of(value) + value.base + word;
    }

    fn word_of_paged(value: Value, word: u32) -> u32 {
        let weights = tables[PLACEMENT_FIRST + PLACEMENT_WEIGHTS];
        let tensors = tables[PLACEMENT_FIRST + PLACEMENT_TENSORS];
        let mut address = value.base + word;
        if value.store == store::WEIGHTS {
            let page = address >> PAGE_SHIFT;
            let slot = pages[page];
            if slot == NO_PAGE {
                refuse(refusal::TENSOR, refusal::PAGE, 0u32);
                return weights;
            }
            address = weights + (slot << PAGE_SHIFT) + (address & PAGE_MASK);
            return address;
        }
        return tensors + address;
    }

    fn publish(value: Value, at: u32, data: f32) {
        poke(word_of(value, at), data);
    }

    fn publish_word(value: Value, word: u32, packed: u32) {
        poke(word_of(value, word), bitcast_f32(packed));
    }
}
