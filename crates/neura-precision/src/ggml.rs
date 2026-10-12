use neura_abi::{
    Block, Element, QUANT_BLOCK, SUPER_BLOCK, SUPER_SCALE_BYTES, SUPER_SMALL_BLOCK,
    SUPER_SUB_BLOCK, WORD_BYTES,
};

const MAX_BLOCK: usize = SUPER_BLOCK as usize;

pub(super) fn pack(element: Element, values: &[f32]) -> Vec<u8> {
    let layout = block(element);
    let mut bytes =
        Vec::with_capacity(values.len().div_ceil(layout.elements as usize) * layout.bytes as usize);
    for lane in values.chunks(layout.elements as usize) {
        let mut held = [0.0f32; MAX_BLOCK];
        held[..lane.len()].copy_from_slice(lane);
        encode(element, &held[..layout.elements as usize], &mut bytes);
    }
    let words = bytes.len().div_ceil(WORD_BYTES as usize) * WORD_BYTES as usize;
    bytes.resize(words, 0);
    bytes
}

pub(super) fn unpack(element: Element, elements: usize, bytes: &[u8]) -> Vec<f32> {
    let block = block(element);
    let held = (element.payload_words(elements as u64) * WORD_BYTES) as usize;
    assert!(
        bytes.len() >= held,
        "unpacking {elements} {} numbers out of {} bytes where they pack {held}",
        element.name(),
        bytes.len(),
    );
    let mut values = Vec::with_capacity(elements);
    for at in 0..elements.div_ceil(block.elements as usize) {
        let field = &bytes[at * block.bytes as usize..(at + 1) * block.bytes as usize];
        decode(element, field, &mut values);
    }
    values.truncate(elements);
    values
}

fn block(element: Element) -> Block {
    element
        .block()
        .expect("a block element lays out the block it packs")
}

fn encode(element: Element, values: &[f32], out: &mut Vec<u8>) {
    match element {
        Element::Q8_0 => q8_0(values, out),
        Element::Q4K => q4_k(values, out),
        Element::Q5K => q5_k(values, out),
        Element::Q6K => q6_k(values, out),
        _ => panic!(
            "{} storage packs numbers a host quantizes by its element",
            element.name(),
        ),
    }
}

fn decode(element: Element, field: &[u8], out: &mut Vec<f32>) {
    match element {
        Element::Q8_0 => q8_0_values(field, out),
        Element::Q4K => q4_k_values(field, out),
        Element::Q5K => q5_k_values(field, out),
        Element::Q6K => q6_k_values(field, out),
        _ => panic!(
            "{} storage walks numbers a host reads by its element",
            element.name(),
        ),
    }
}

fn half_at(field: &[u8], at: usize) -> f32 {
    half::f16::from_bits(u16::from_le_bytes([field[at], field[at + 1]])).to_f32()
}

fn put_half(field: &mut [u8], at: usize, value: f32) {
    field[at..at + 2].copy_from_slice(&half::f16::from_f32(value).to_bits().to_le_bytes());
}

fn nibble_at(byte: u8, high: bool) -> u32 {
    u32::from(if high { byte >> 4 } else { byte & 0x0f })
}

fn put_nibble(field: &mut [u8], at: usize, high: bool, value: u32) {
    field[at] |= ((value & 0x0f) as u8) << if high { 4 } else { 0 };
}

fn least(value: f32, other: f32) -> f32 {
    if value < other { value } else { other }
}

fn greatest(value: f32, other: f32) -> f32 {
    if value > other { value } else { other }
}

fn peak(values: &[f32]) -> (f32, f32) {
    values
        .iter()
        .fold((0.0f32, 0.0f32), |(size, signed), value| {
            if size < value.abs() {
                (value.abs(), *value)
            } else {
                (size, signed)
            }
        })
}

fn span(values: &[f32]) -> (f32, f32) {
    values.iter().fold((0.0f32, 0.0f32), |(low, high), value| {
        (least(low, *value), greatest(high, *value))
    })
}

fn scale_of(quantum: f32, low: f32) -> f32 {
    if quantum == 0.0 { 0.0 } else { low / quantum }
}

fn quantized(value: f32, quantum: f32, low: f32, high: f32) -> u32 {
    if quantum == 0.0 {
        return 0;
    }
    (value / quantum).round().clamp(low, high) as u32
}

fn scale_min_k4(scales: &[u8], at: usize) -> (u32, u32) {
    if at < 4 {
        (u32::from(scales[at] & 63), u32::from(scales[at + 4] & 63))
    } else {
        (
            u32::from(scales[at + 4] & 0x0f) | u32::from(scales[at - 4] >> 6) << 4,
            u32::from(scales[at + 4] >> 4) | u32::from(scales[at] >> 6) << 4,
        )
    }
}

fn put_scale_min_k4(scales: &mut [u8], at: usize, scale: u32, min: u32) {
    if at < 4 {
        scales[at] = scale as u8;
        scales[at + 4] = min as u8;
    } else {
        scales[at + 4] = ((scale & 0x0f) | (min & 0x0f) << 4) as u8;
        scales[at - 4] |= ((scale >> 4) << 6) as u8;
        scales[at] |= ((min >> 4) << 6) as u8;
    }
}

fn affine_block(element: Element, values: &[f32], out: &mut Vec<u8>, steps: u32, sub: usize) {
    let block = block(element);
    let count = SUPER_BLOCK as usize / sub;
    let mut lows = [0.0f32; MAX_BLOCK];
    let mut quantums = [0.0f32; MAX_BLOCK];
    for at in 0..count {
        let lane = &values[at * sub..at * sub + sub];
        let (low, high) = span(lane);
        lows[at] = least(0.0, low);
        quantums[at] = (high - lows[at]) / steps as f32;
    }
    let scale = quantums[..count].iter().copied().fold(0.0f32, greatest) / 63.0;
    let min = lows[..count].iter().map(|low| -low).fold(0.0f32, greatest) / 63.0;
    let mut codes = [0u8; MAX_BLOCK];
    let mut mins = [0u8; MAX_BLOCK];
    for at in 0..count {
        codes[at] = scale_of(scale, quantums[at]).round().clamp(0.0, 63.0) as u8;
        mins[at] = scale_of(min, -lows[at]).round().clamp(0.0, 63.0) as u8;
    }
    let mut scales = [0u8; SUPER_SCALE_BYTES as usize];
    for at in 0..count {
        put_scale_min_k4(&mut scales, at, u32::from(codes[at]), u32::from(mins[at]));
    }
    let start = out.len();
    out.resize(start + block.bytes as usize, 0);
    let field = &mut out[start..start + block.bytes as usize];
    put_half(field, 0, scale);
    put_half(field, 2, min);
    let (quants, bits) = match element {
        Element::Q5K => (48, Some(16)),
        _ => (16, None),
    };
    field[4..4 + SUPER_SCALE_BYTES as usize].copy_from_slice(&scales);
    let mut high_bits = [0u8; SUPER_BLOCK as usize / 8];
    for at in 0..count {
        let quantum = scale * f32::from(codes[at]);
        let low = min * f32::from(mins[at]);
        for within in 0..sub {
            let code = quantized(values[at * sub + within] + low, quantum, 0.0, steps as f32);
            let group = at / 2;
            let seat = at % 2 == 1;
            put_nibble(field, quants + group * sub + within, seat, code);
            if bits.is_some() && code & 0x10 != 0 {
                high_bits[within] |= 1 << (2 * group + usize::from(seat));
            }
        }
    }
    if let Some(at) = bits {
        field[at..at + SUPER_BLOCK as usize / 8].copy_from_slice(&high_bits);
    }
}

fn q8_0(values: &[f32], out: &mut Vec<u8>) {
    let (size, _) = peak(values);
    let quantum = size / 127.0;
    let inverse = if quantum == 0.0 { 0.0 } else { 1.0 / quantum };
    out.extend_from_slice(&half_bytes(quantum));
    for value in values {
        out.push((value * inverse).round().clamp(-127.0, 127.0) as i8 as u8);
    }
}

fn q8_0_values(field: &[u8], out: &mut Vec<f32>) {
    let quantum = half_at(field, 0);
    for at in 0..QUANT_BLOCK as usize {
        out.push((field[2 + at] as i8) as f32 * quantum);
    }
}

fn q5_k(values: &[f32], out: &mut Vec<u8>) {
    affine_block(Element::Q5K, values, out, 31, SUPER_SUB_BLOCK as usize);
}

fn q4_k(values: &[f32], out: &mut Vec<u8>) {
    affine_block(Element::Q4K, values, out, 15, SUPER_SUB_BLOCK as usize);
}

fn q5_k_values(field: &[u8], out: &mut Vec<f32>) {
    let scale = half_at(field, 0);
    let min = half_at(field, 2);
    for group in 0..4 {
        for seat in 0..2 {
            let (code, offset) = scale_min_k4(&field[4..16], group * 2 + seat);
            let quantum = scale * code as f32;
            let low = min * offset as f32;
            for within in 0..SUPER_SUB_BLOCK as usize {
                let high = (field[16 + within] >> (2 * group + seat)) & 1;
                let quant =
                    nibble_at(field[48 + group * 32 + within], seat == 1) | u32::from(high) << 4;
                out.push(quantum * quant as f32 - low);
            }
        }
    }
}

fn q4_k_values(field: &[u8], out: &mut Vec<f32>) {
    let scale = half_at(field, 0);
    let min = half_at(field, 2);
    for group in 0..4 {
        for seat in 0..2 {
            let (code, offset) = scale_min_k4(&field[4..16], group * 2 + seat);
            let quantum = scale * code as f32;
            let low = min * offset as f32;
            for within in 0..SUPER_SUB_BLOCK as usize {
                let quant = nibble_at(field[16 + group * 32 + within], seat == 1);
                out.push(quantum * quant as f32 - low);
            }
        }
    }
}

fn q6_k(values: &[f32], out: &mut Vec<u8>) {
    let sub = SUPER_SMALL_BLOCK as usize;
    let count = SUPER_BLOCK as usize / sub;
    let mut factors = [0.0f32; MAX_BLOCK];
    for at in 0..count {
        let (size, _) = peak(&values[at * sub..at * sub + sub]);
        factors[at] = size / 32.0;
    }
    let scale = factors[..count]
        .iter()
        .map(|factor| factor.abs())
        .fold(0.0f32, greatest)
        / 127.0;
    let start = out.len();
    out.resize(start + 210, 0);
    let field = &mut out[start..start + 210];
    put_half(field, 208, scale);
    for at in 0..count {
        let code = if scale == 0.0 {
            0
        } else {
            (factors[at] / scale).round().clamp(-127.0, 127.0) as i8
        };
        field[192 + at] = code as u8;
        let factor = scale * f32::from(code);
        for within in 0..sub {
            let element = at * sub + within;
            let half = element / 128;
            let group = (element % 128) / 32;
            let seat = element % 32;
            let quant = quantized(values[element], factor, -32.0, 31.0) + 32;
            put_nibble(
                field,
                half * 64 + (group % 2) * 32 + seat,
                group >= 2,
                quant,
            );
            field[128 + half * 32 + seat] |= (((quant >> 4) & 3) << (2 * group)) as u8;
        }
    }
}

fn q6_k_values(field: &[u8], out: &mut Vec<f32>) {
    let scale = half_at(field, 208);
    for seat in 0..2 {
        for group in 0..4 {
            for within in 0..SUPER_SUB_BLOCK as usize {
                let index = group * 2 + within / 16;
                let sub = field[192 + seat * 8 + index] as i8;
                let quant = nibble_at(field[seat * 64 + (group % 2) * 32 + within], group >= 2)
                    | u32::from((field[128 + seat * 32 + within] >> (2 * group)) & 3) << 4;
                out.push(scale * f32::from(sub) * (quant as f32 - 32.0));
            }
        }
    }
}

fn half_bytes(value: f32) -> [u8; 2] {
    half::f16::from_f32(value).to_bits().to_le_bytes()
}
