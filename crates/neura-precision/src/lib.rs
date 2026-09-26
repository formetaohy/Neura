use neura_abi::{Element, WORD_BYTES};

pub fn pack(element: Element, values: &[f32]) -> Vec<u8> {
    match element {
        Element::Single => bytemuck::cast_slice(values).to_vec(),
        Element::Half => halves(values, |value| half::f16::from_f32(value).to_bits()),
        Element::Bfloat16 => halves(values, bfloat16),
    }
}

pub fn unpack(element: Element, elements: usize, bytes: &[u8]) -> Vec<f32> {
    assert!(
        bytes.len() as u64 >= element.words(elements as u64) * WORD_BYTES,
        "unpacking {elements} {} elements out of {} bytes",
        element.name(),
        bytes.len(),
    );
    let mut values = match element {
        Element::Single => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        Element::Half => halves_of(bytes, |word| {
            [
                half::f16::from_bits(word as u16).to_f32(),
                half::f16::from_bits((word >> 16) as u16).to_f32(),
            ]
        }),
        Element::Bfloat16 => halves_of(bytes, |word| {
            [single(word << 16), single(word & 0xffff_0000)]
        }),
    };
    values.truncate(elements);
    values
}

fn single(word: u32) -> f32 {
    f32::from_bits(word)
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

fn halves_of(bytes: &[u8], split: impl Fn(u32) -> [f32; 2]) -> Vec<f32> {
    bytemuck::cast_slice::<u8, u32>(bytes)
        .iter()
        .flat_map(|word| split(*word))
        .collect()
}

fn bfloat16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}
