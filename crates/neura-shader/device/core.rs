#[neura_compiler::module]
mod source {
    fn refuse(subject: u32, category: u32, code: u32) {
        atomic_store(
            &refusal[0u32],
            (subject << refusal::KIND_BITS) | (category << refusal::CODE_BITS) | (code + 1u32),
        );
    }

    fn coordinates(flat: u32, dims: uvec4) -> uvec4 {
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

    fn run_task(task: Task, lid: u32) {
        match task.kind {
            _ => refuse(task.kind, refusal::TASK, 0u32),
        }
    }

    #[neura_compiler::kernel]
    fn main(lid: u32, group: uvec3) {
        let segment = segments[bounds.first_segment + group.x];
        for index in stride(segment.first, segment.first + segment.count, 1u32) {
            run_task(tasks[index], lid);
            storage_barrier();
        }
    }
}
