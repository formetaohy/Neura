use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn refuse(subject: u32, category: u32, code: u32) {
        if atomic_add(&refusal[0u32], 0u32) == 0u32 {
            atomic_store(
                &refusal[0u32],
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
            placement.weights,
            placement.tensors,
            value.store == store::TENSORS,
        );
    }

    fn word_of(value: Value, word: u32) -> u32 {
        return base_of(value) + value.base + word;
    }

    fn publish(value: Value, at: u32, data: f32) {
        heap[word_of(value, at)] = data;
    }

    fn publish_word(value: Value, word: u32, packed: u32) {
        heap[word_of(value, word)] = bitcast_f32(packed);
    }
}
