use neura_abi::Element;
use neura_compiler::{Compiler, ast};

pub(crate) fn install(compiler: &mut Compiler, elements: &[Element]) {
    device::define(compiler);
    for element in elements.iter().copied().filter(|element| {
        element.narrow() && (element.block().is_none() || element.packs_on_device())
    }) {
        let name = match element.format() {
            Some(format) => {
                let name = format!("run_convert_{}", element.name());
                compiler.specialize(
                    "template_run_convert_fp8",
                    &name,
                    &crate::fetch::format_constants(format),
                );
                name
            }
            None => convert_of(element).to_owned(),
        };
        compiler.insert_case(
            "run_convert",
            ast::Arm {
                pattern: ast::Pattern::Integer(element.code()),
                body: vec![ast::Statement::Expression(ast::Expression::call(
                    name,
                    vec![ast::Expression::name("task"), ast::Expression::name("lid")],
                ))],
            },
        );
    }
}

fn convert_of(element: Element) -> &'static str {
    match element {
        Element::Half => "run_convert_half",
        Element::Bfloat16 => "run_convert_bfloat16",
        Element::Int8 => "run_convert_int8",
        Element::Q8_0 => "run_convert_q8_0",
        Element::Single => panic!("a single precision tensor holds one element per word"),
        Element::Int4 | Element::Q3K | Element::Q4K | Element::Q5K | Element::Q6K => {
            panic!(
                "{} storage packs numbers a device cannot quantize, and its host lays the block out",
                element.name(),
            )
        }
        Element::Fp8E4M3 | Element::Fp8E5M2 => panic!(
            "{} storage walks a float grid, and its convert is specialized from the grid the ABI declares",
            element.name(),
        ),
        Element::Fp4E2M1 => panic!("a block quantized tensor is written by its host"),
    }
}

#[neura_compiler::module]
mod device {
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
        let scale = task.literal;
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

    fn run_convert_q8_0(task: Task, lid: u32) {
        let output = values[task.out];
        let source = values[task.a];
        let dims = output.dims;
        let elements = dims.x * dims.y * dims.z * dims.w;
        let blocks = ceil_div(elements, Q8_0_BLOCK);
        let words = ceil_div(blocks * Q8_0_BYTES, 4u32);
        let held = 2u32 * Q8_0_BYTES / 4u32;
        let pairs = ceil_div(blocks, 2u32);
        let first = task.first / held;
        let last = min(ceil_div(task.first + task.count, held), pairs);
        for pair in stride(first + lid, last, WORKGROUP_SIZE) {
            let lower = q8_0_quantum(task, source, output, 2u32 * pair);
            let upper = select(
                lower,
                q8_0_quantum(task, source, output, 2u32 * pair + 1u32),
                2u32 * pair + 1u32 < blocks,
            );
            let lower_bits = half_bits(lower);
            let upper_bits = half_bits(upper);
            let opening = pair * held;
            let closing = min(opening + held, words);
            for word in stride(opening, closing, 1u32) {
                let mut packed = 0u32;
                for within in unroll(0u32, 4u32, 1u32) {
                    let offset = (word - opening) * 4u32 + within;
                    let second = offset >= Q8_0_BYTES;
                    let inside = offset - select(0u32, Q8_0_BYTES, second);
                    let bits = select(lower_bits, upper_bits, second);
                    let quantum = select(lower, upper, second);
                    let mut byte = (bits >> (inside * 8u32)) & 0xffu32;
                    if inside >= 2u32 {
                        let block = 2u32 * pair + select(0u32, 1u32, second);
                        byte = q8_0_code(
                            q8_0_element(task, source, output, block * Q8_0_BLOCK + inside - 2u32),
                            quantum,
                        );
                    }
                    packed = packed | (byte << (within * 8u32));
                }
                publish_word(output, word, packed);
            }
        }
    }

    fn q8_0_quantum(task: Task, source: Value, output: Value, block: u32) -> f32 {
        let mut size = 0.0;
        for within in stride(0u32, Q8_0_BLOCK, 1u32) {
            size = max(
                size,
                abs(q8_0_element(
                    task,
                    source,
                    output,
                    block * Q8_0_BLOCK + within,
                )),
            );
        }
        return size / 127.0;
    }

    fn q8_0_element(task: Task, source: Value, output: Value, element: u32) -> f32 {
        let dims = output.dims;
        let elements = dims.x * dims.y * dims.z * dims.w;
        let inside = element < elements;
        let at = walked_at(task.geometry, min(element, elements - 1u32), dims);
        return select(0.0, convert_at(task, source, at), inside);
    }

    fn q8_0_code(value: f32, quantum: f32) -> u32 {
        let inverse = select(0.0, 1.0 / quantum, quantum != 0.0);
        let scaled = value * inverse;
        let rounded = trunc(scaled + select(-0.5, 0.5, scaled >= 0.0));
        let clamped = min(max(rounded, -127.0), 127.0);
        let magnitude = u32(abs(clamped));
        return select(magnitude, 256u32 - magnitude, clamped < 0.0);
    }

    fn run_convert(task: Task, lid: u32) {
        match values[task.out].element {
            _ => refuse(kind::CONVERT, refusal::ELEMENT, values[task.out].element),
        }
    }
}
