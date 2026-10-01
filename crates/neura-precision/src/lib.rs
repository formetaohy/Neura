use neura_abi::{Element, FP4_BLOCK, FloatFormat, INT4_BLOCK, WORD_BYTES};

pub fn pack(element: Element, quantum: f32, values: &[f32]) -> Vec<u8> {
    match element {
        Element::Single => bytemuck::cast_slice(values).to_vec(),
        Element::Half => halves(values, |value| half::f16::from_f32(value).to_bits()),
        Element::Bfloat16 => halves(values, bfloat16),
        Element::Int8 => {
            let mut bytes = words(values, |value| i8_bits(value, quantum));
            bytes.extend_from_slice(&quantum.to_ne_bytes());
            bytes
        }
        Element::Int4 => int4(values),
        Element::Fp8E4M3 => words(values, |value| e4m3_bits(value) as u8),
        Element::Fp8E5M2 => words(values, |value| e5m2_bits(value) as u8),
        Element::Fp4E2M1 => fp4(values),
    }
}

pub fn unpack(element: Element, elements: usize, bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() as u64 >= element.storage_words(elements as u64) * WORD_BYTES,
        "unpacking {elements} {} elements out of {} bytes",
        element.name(),
        bytes.len(),
    );
    let mut values = match element {
        Element::Single => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        Element::Half => halves_of(payload(element, elements, bytes), |word| {
            [
                half::f16::from_bits(word as u16).to_f32(),
                half::f16::from_bits((word >> 16) as u16).to_f32(),
            ]
        }),
        Element::Bfloat16 => halves_of(payload(element, elements, bytes), |word| {
            [single(word << 16), single(word & 0xffff_0000)]
        }),
        Element::Int8 => {
            let quantum = quantum_of(element, elements, bytes);
            bytes_to_words(payload(element, elements, bytes))
                .iter()
                .flat_map(|word| {
                    (0..4).map(move |lane| i8_value((word >> (8 * lane)) as u8) * quantum)
                })
                .collect()
        }
        Element::Int4 => int4_values(elements, bytes),
        Element::Fp8E4M3 => bytes_to_words(payload(element, elements, bytes))
            .iter()
            .flat_map(|word| (0..4).map(move |lane| e4m3_value((word >> (8 * lane)) as u8)))
            .collect(),
        Element::Fp8E5M2 => bytes_to_words(payload(element, elements, bytes))
            .iter()
            .flat_map(|word| (0..4).map(move |lane| e5m2_value((word >> (8 * lane)) as u8)))
            .collect(),
        Element::Fp4E2M1 => fp4_values(elements, bytes),
    };
    values.truncate(elements);
    values
}

fn single(word: u32) -> f32 {
    f32::from_bits(word)
}

fn i8_value(byte: u8) -> f32 {
    (byte as i8) as f32
}

fn i8_bits(value: f32, scale: f32) -> u8 {
    assert!(
        scale.is_finite() && scale > 0.0,
        "an int8 tensor of scale {scale} reconstructs nothing",
    );
    (value / scale).round().clamp(-127.0, 127.0) as i8 as u8
}

fn int4(values: &[f32]) -> Vec<u8> {
    let mut table = Vec::with_capacity(values.len().div_ceil(INT4_BLOCK as usize) * 4);
    let mut words = Vec::with_capacity(values.len().div_ceil(8) * 4);
    for block in values.chunks(INT4_BLOCK as usize) {
        let peak = block
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let quantum = peak / 7.0;
        table.extend_from_slice(&quantum.to_ne_bytes());
        for lane in block.chunks(8) {
            let mut word = 0u32;
            for (slot, value) in lane.iter().enumerate() {
                word |= u32::from(int4_bits(*value, quantum)) << (4 * slot as u32);
            }
            words.extend_from_slice(&word.to_ne_bytes());
        }
    }
    words.extend_from_slice(&table);
    words
}

fn int4_bits(value: f32, quantum: f32) -> u8 {
    if quantum == 0.0 {
        return 0;
    }
    ((value / quantum).round().clamp(-8.0, 7.0) as i8 as u8) & 0x0f
}

fn fp4(values: &[f32]) -> Vec<u8> {
    let mut table = Vec::with_capacity(values.len().div_ceil(FP4_BLOCK as usize) * 4);
    let mut words = Vec::with_capacity(values.len().div_ceil(8) * 4);
    for block in values.chunks(FP4_BLOCK as usize) {
        let peak = block
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let quantum = peak / 6.0;
        table.extend_from_slice(&quantum.to_ne_bytes());
        for lane in block.chunks(8) {
            let mut word = 0u32;
            for (slot, value) in lane.iter().enumerate() {
                word |= u32::from(fp4_bits(*value, quantum)) << (4 * slot as u32);
            }
            words.extend_from_slice(&word.to_ne_bytes());
        }
    }
    words.extend_from_slice(&table);
    words
}

fn fp4_bits(value: f32, quantum: f32) -> u8 {
    if quantum == 0.0 {
        return 0;
    }
    let code = fp8_bits(value / quantum, grid(Element::Fp4E2M1));
    (((code & 0x80) >> 4) | (code & 0x07)) as u8
}

fn fp4_values(elements: usize, bytes: &[u8]) -> Vec<f32> {
    let table = bytes_to_words(&bytes[payload_bytes(Element::Fp4E2M1, elements)..]);
    bytes_to_words(payload(Element::Fp4E2M1, elements, bytes))
        .iter()
        .enumerate()
        .flat_map(|(word, packed)| {
            (0..8).map(move |lane| {
                let nibble = (packed >> (4 * lane)) & 0x0f;
                fp4_value(nibble as u8) * single(table[(word * 8 + lane) / FP4_BLOCK as usize])
            })
        })
        .collect()
}

fn fp4_value(nibble: u8) -> f32 {
    fp8_value(
        ((nibble & 0x08) << 4) | (nibble & 0x07),
        grid(Element::Fp4E2M1),
    )
}

fn int4_values(elements: usize, bytes: &[u8]) -> Vec<f32> {
    let table = bytes_to_words(&bytes[payload_bytes(Element::Int4, elements)..]);
    bytes_to_words(payload(Element::Int4, elements, bytes))
        .iter()
        .enumerate()
        .flat_map(|(word, packed)| {
            (0..8).map(move |lane| {
                let nibble = (packed >> (4 * lane)) & 0x0f;
                let code = nibble as i32 - if nibble >= 8 { 16 } else { 0 };
                code as f32 * single(table[(word * 8 + lane) / INT4_BLOCK as usize])
            })
        })
        .collect()
}

fn payload(element: Element, elements: usize, bytes: &[u8]) -> &[u8] {
    &bytes[..(element.payload_words(elements as u64) * WORD_BYTES) as usize]
}

fn payload_bytes(element: Element, elements: usize) -> usize {
    (element.payload_words(elements as u64) * WORD_BYTES) as usize
}

fn quantum_of(element: Element, elements: usize, bytes: &[u8]) -> f32 {
    assert!(
        element.per_tensor(),
        "{} storage carries one quantum of its own per block, so a reader reconstructs through the table its storage holds",
        element.name(),
    );
    let at = payload_bytes(element, elements);
    single(u32::from_ne_bytes(
        bytes[at..at + WORD_BYTES as usize]
            .try_into()
            .expect("a quantum table holds a word"),
    ))
}

fn words(values: &[f32], convert: impl Fn(f32) -> u8) -> Vec<u8> {
    let packed = values
        .chunks(4)
        .map(|lane| {
            lane.iter().enumerate().fold(0u32, |word, (slot, value)| {
                word | (u32::from(convert(*value)) << (8 * slot as u32))
            })
        })
        .collect::<Vec<u32>>();
    bytemuck::cast_slice(&packed).to_vec()
}

fn halves(values: &[f32], convert: impl Fn(f32) -> u16) -> Vec<u8> {
    let words = values
        .chunks(2)
        .map(|pair| {
            let low = u32::from(convert(pair[0]));
            let high = pair.get(1).map_or(0, |value| u32::from(convert(*value)));
            low | (high << 16)
        })
        .collect::<Vec<u32>>();
    bytemuck::cast_slice(&words).to_vec()
}

fn bytes_to_words(bytes: &[u8]) -> &[u32] {
    bytemuck::cast_slice::<u8, u32>(bytes)
}

fn halves_of(bytes: &[u8], split: impl Fn(u32) -> [f32; 2]) -> Vec<f32> {
    bytes_to_words(bytes)
        .iter()
        .flat_map(|word| split(*word))
        .collect()
}

fn grid(element: Element) -> FloatFormat {
    element.format().unwrap_or_else(|| {
        panic!(
            "{} storage carries numbers on a grid the ABI does not declare",
            element.name(),
        )
    })
}

fn e4m3_bits(value: f32) -> u32 {
    fp8_bits(value, grid(Element::Fp8E4M3))
}

fn e5m2_bits(value: f32) -> u32 {
    fp8_bits(value, grid(Element::Fp8E5M2))
}

fn fp8_bits(value: f32, format: FloatFormat) -> u32 {
    let bits = value.to_bits();
    let sign = (bits >> 24) & 0x80;
    let magnitude = bits & 0x7fff_ffff;
    if magnitude > 0x7f80_0000 {
        return sign | format.nan;
    }
    if magnitude >= format.ceiling {
        return sign | format.max;
    }
    let exponent = ((magnitude >> 23) as i32) - 127;
    let significand = (magnitude & 0x007f_ffff) | 0x0080_0000;
    let rounding = 23 - format.mantissa_bits;
    if exponent >= 1 - format.bias {
        let pinned = (significand + (1 << (rounding - 1))) >> rounding;
        let mut code = (exponent + format.bias) as u32;
        let mut carried = pinned - (1 << format.mantissa_bits);
        if carried > (1 << format.mantissa_bits) - 1 {
            carried = 0;
            code += 1;
        }
        if code > (format.max >> format.mantissa_bits) {
            return sign | format.max;
        }
        return sign | (code << format.mantissa_bits) | carried;
    }
    let shift = (format.subnormal_shift - exponent) as u32;
    if shift >= 32 {
        return sign;
    }
    let pinned = (significand + (1 << (shift - 1))) >> shift;
    if pinned > (1 << format.mantissa_bits) - 1 {
        return sign | (1 << format.mantissa_bits);
    }
    sign | pinned
}

fn e4m3_value(byte: u8) -> f32 {
    fp8_value(byte, grid(Element::Fp8E4M3))
}

fn e5m2_value(byte: u8) -> f32 {
    fp8_value(byte, grid(Element::Fp8E5M2))
}

fn fp8_value(byte: u8, format: FloatFormat) -> f32 {
    let code = u32::from(byte);
    let sign = code & 0x80;
    let body = code & 0x7f;
    if body > format.infinity || body == format.nan {
        return f32::from_bits((sign << 24) | 0x7fc0_0000);
    }
    if body == format.infinity {
        return f32::from_bits((sign << 24) | 0x7f80_0000);
    }
    let exponent = body >> format.mantissa_bits;
    let mantissa = body & ((1 << format.mantissa_bits) - 1);
    let magnitude = if exponent == 0 {
        mantissa as f32 * format.smallest
    } else {
        f32::from_bits(
            (((exponent as i32 + 127 - format.bias) as u32) << 23)
                | (mantissa << (23 - format.mantissa_bits)),
        )
    };
    signed(sign, magnitude)
}

fn signed(sign: u32, value: f32) -> f32 {
    if sign == 0 { value } else { -value }
}

fn bfloat16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}
