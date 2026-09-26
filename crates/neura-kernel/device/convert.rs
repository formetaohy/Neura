#[neura_compiler::module]
mod source {
    fn half_bits(value: f32) -> u32 {
        let bits = bitcast_u32(value);
        let sign = (bits >> 16u32) & 0x8000u32;
        let magnitude = bits & 0x7fff_ffffu32;
        if magnitude > 0x7f80_0000u32 {
            return sign | 0x7e00u32;
        }
        if magnitude >= 0x4780_0000u32 {
            return sign | 0x7c00u32;
        }
        if magnitude < 0x3300_0000u32 {
            return sign;
        }
        if magnitude < 0x3880_0000u32 {
            let exponent = (bits >> 23u32) & 0xffu32;
            let significand = (bits & 0x007f_ffffu32) | 0x0080_0000u32;
            let shift = 126u32 - exponent;
            let rounded =
                significand + ((1u32 << (shift - 1u32)) - 1u32) + ((significand >> shift) & 1u32);
            return sign | (rounded >> shift);
        }
        let rounded = bits + 0x0fffu32 + ((bits >> 13u32) & 1u32);
        let exponent = (rounded >> 23u32) & 0xffu32;
        if exponent >= 143u32 {
            return sign | 0x7c00u32;
        }
        return sign | ((exponent - 112u32) << 10u32) | ((rounded >> 13u32) & 0x3ffu32);
    }

    fn bfloat16_bits(value: f32) -> u32 {
        let bits = bitcast_u32(value);
        return (bits + 0x7fffu32 + ((bits >> 16u32) & 1u32)) >> 16u32;
    }

    fn convert_at(task: Task, source: Value, at: uvec4) -> f32 {
        return chained(task, at, fetch(source, read_address(at, source.strides)));
    }

    fn run_convert_half(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let dims = output.dims;
        let elements = dims.x * dims.y * dims.z * dims.w;
        for word in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let low = coordinates(2u32 * word, dims);
            let high = coordinates(
                select(
                    2u32 * word,
                    2u32 * word + 1u32,
                    2u32 * word + 1u32 < elements,
                ),
                dims,
            );
            let low_bits = half_bits(convert_at(task, source, low));
            let high_bits = half_bits(convert_at(task, source, high));
            publish_word(output, word, low_bits | (high_bits << 16u32));
        }
    }

    fn run_convert_bfloat16(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let dims = output.dims;
        let elements = dims.x * dims.y * dims.z * dims.w;
        for word in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let low = coordinates(2u32 * word, dims);
            let high = coordinates(
                select(
                    2u32 * word,
                    2u32 * word + 1u32,
                    2u32 * word + 1u32 < elements,
                ),
                dims,
            );
            let low_bits = bfloat16_bits(convert_at(task, source, low));
            let high_bits = bfloat16_bits(convert_at(task, source, high));
            publish_word(output, word, low_bits | (high_bits << 16u32));
        }
    }

    fn run_convert(task: Task, lid: u32) {
        match values[task.out].element {
            _ => refuse(kind::CONVERT, refusal::ELEMENT, values[task.out].element),
        }
    }
}
