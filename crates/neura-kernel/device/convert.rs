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

    fn int8_bits(value: f32, scale: f32) -> u32 {
        let scaled = value / scale;
        let rounded = trunc(scaled + select(-0.5, 0.5, scaled >= 0.0));
        let clamped = min(max(rounded, -127.0), 127.0);
        let magnitude = u32(abs(clamped));
        return select(magnitude, 256u32 - magnitude, clamped < 0.0);
    }

    fn fp8_bits(
        value: f32,
        bias: u32,
        mantissa_bits: u32,
        subnormal_shift: u32,
        nan: u32,
        max: u32,
        ceiling: u32,
    ) -> u32 {
        let bits = bitcast_u32(value);
        let sign = (bits >> 24u32) & 0x80u32;
        let magnitude = bits & 0x7fff_ffffu32;
        if magnitude > 0x7f80_0000u32 {
            return sign | nan;
        }
        if magnitude >= ceiling {
            return sign | max;
        }
        let exponent = i32(magnitude >> 23u32) - 127i32;
        let significand = (magnitude & 0x007f_ffffu32) | 0x0080_0000u32;
        let rounding = 23u32 - mantissa_bits;
        if exponent >= 1i32 - i32(bias) {
            let pinned = (significand + (1u32 << (rounding - 1u32))) >> rounding;
            let mut code = u32(exponent + i32(bias));
            let mut carried = pinned - (1u32 << mantissa_bits);
            if carried > (1u32 << mantissa_bits) - 1u32 {
                carried = 0u32;
                code = code + 1u32;
            }
            if code > (max >> mantissa_bits) {
                return sign | max;
            }
            return sign | (code << mantissa_bits) | carried;
        }
        let shift = i32(subnormal_shift) - exponent;
        if shift >= 32i32 {
            return sign;
        }
        let pinned = (significand + (1u32 << (u32(shift) - 1u32))) >> u32(shift);
        if pinned > (1u32 << mantissa_bits) - 1u32 {
            return sign | (1u32 << mantissa_bits);
        }
        return sign | pinned;
    }

    fn pack_fp8(
        task: Task,
        source: Value,
        output: Value,
        word: u32,
        bias: u32,
        mantissa_bits: u32,
        subnormal_shift: u32,
        nan: u32,
        max: u32,
        ceiling: u32,
    ) -> u32 {
        let dims = output.dims;
        let elements = dims.x * dims.y * dims.z * dims.w;
        let low = fp8_bits(
            convert_at(
                task,
                source,
                walked_at(task.geometry, min(4u32 * word, elements - 1u32), dims),
            ),
            bias,
            mantissa_bits,
            subnormal_shift,
            nan,
            max,
            ceiling,
        );
        let second = fp8_bits(
            convert_at(
                task,
                source,
                walked_at(
                    task.geometry,
                    min(4u32 * word + 1u32, elements - 1u32),
                    dims,
                ),
            ),
            bias,
            mantissa_bits,
            subnormal_shift,
            nan,
            max,
            ceiling,
        );
        let third = fp8_bits(
            convert_at(
                task,
                source,
                walked_at(
                    task.geometry,
                    min(4u32 * word + 2u32, elements - 1u32),
                    dims,
                ),
            ),
            bias,
            mantissa_bits,
            subnormal_shift,
            nan,
            max,
            ceiling,
        );
        let high = fp8_bits(
            convert_at(
                task,
                source,
                walked_at(
                    task.geometry,
                    min(4u32 * word + 3u32, elements - 1u32),
                    dims,
                ),
            ),
            bias,
            mantissa_bits,
            subnormal_shift,
            nan,
            max,
            ceiling,
        );
        return low | (second << 8u32) | (third << 16u32) | (high << 24u32);
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
            let low = walked_at(task.geometry, 2u32 * word, dims);
            let high = walked_at(
                task.geometry,
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
            let low = walked_at(task.geometry, 2u32 * word, dims);
            let high = walked_at(
                task.geometry,
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

    fn run_convert_int8(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let dims = output.dims;
        let elements = dims.x * dims.y * dims.z * dims.w;
        let scale = task.param;
        for word in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let low = int8_bits(
                convert_at(
                    task,
                    source,
                    walked_at(task.geometry, min(4u32 * word, elements - 1u32), dims),
                ),
                scale,
            );
            let second = int8_bits(
                convert_at(
                    task,
                    source,
                    walked_at(
                        task.geometry,
                        min(4u32 * word + 1u32, elements - 1u32),
                        dims,
                    ),
                ),
                scale,
            );
            let third = int8_bits(
                convert_at(
                    task,
                    source,
                    walked_at(
                        task.geometry,
                        min(4u32 * word + 2u32, elements - 1u32),
                        dims,
                    ),
                ),
                scale,
            );
            let high = int8_bits(
                convert_at(
                    task,
                    source,
                    walked_at(
                        task.geometry,
                        min(4u32 * word + 3u32, elements - 1u32),
                        dims,
                    ),
                ),
                scale,
            );
            publish_word(
                output,
                word,
                low | (second << 8u32) | (third << 16u32) | (high << 24u32),
            );
        }
    }

    fn template_run_convert_fp8(task: Task, lid: u32) {
        let output = values[task.out];
        let source = values[task.a];
        for word in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            publish_word(
                output,
                word,
                pack_fp8(
                    task,
                    source,
                    output,
                    word,
                    FP8_BIAS,
                    FP8_MANTISSA_BITS,
                    FP8_SUBNORMAL_SHIFT,
                    FP8_NAN,
                    FP8_MAX,
                    FP8_CEILING,
                ),
            );
        }
    }

    fn run_convert(task: Task, lid: u32) {
        match values[task.out].element {
            _ => refuse(kind::CONVERT, refusal::ELEMENT, values[task.out].element),
        }
    }
}
