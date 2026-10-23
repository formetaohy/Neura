use neura_abi::{Element, FP4_BLOCK, INT4_BLOCK};
use neura_precision::{pack, unpack};

fn round_trip(element: Element, values: &[f32]) -> Vec<f32> {
    let bytes = pack(element, 1.0, values);
    unpack(element, values.len(), &bytes)
}

#[test]
fn an_e4m3_word_holds_four_elements() {
    let values = [1.0f32, -0.5, 448.0, 0.0];
    let bytes = pack(Element::Fp8E4M3, 1.0, &values);
    assert_eq!(bytes.len(), 4);
    assert_eq!(bytes, vec![0x38, 0xb0, 0x7e, 0x00]);
    assert_eq!(
        round_trip(Element::Fp8E4M3, &values),
        vec![1.0, -0.5, 448.0, 0.0]
    );
}

#[test]
fn an_e5m2_word_holds_four_elements() {
    let values = [1.0f32, -0.5, 57344.0, 0.0];
    let bytes = pack(Element::Fp8E5M2, 1.0, &values);
    assert_eq!(bytes.len(), 4);
    assert_eq!(bytes, vec![0x3c, 0xb8, 0x7b, 0x00]);
    assert_eq!(
        round_trip(Element::Fp8E5M2, &values),
        vec![1.0, -0.5, 57344.0, 0.0]
    );
}

#[test]
fn a_subnormal_keeps_the_smallest_step_it_can_hold() {
    let e4m3 = round_trip(
        Element::Fp8E4M3,
        &[1.0 / 512.0, 1.0 / 1024.0, 3.0 / 512.0, 1.0 / 4096.0],
    );
    assert_eq!(e4m3, vec![1.0 / 512.0, 1.0 / 512.0, 3.0 / 512.0, 0.0]);
    let e5m2 = round_trip(
        Element::Fp8E5M2,
        &[1.0 / 65536.0, 1.0 / 131072.0, 3.0 / 65536.0, 1.0 / 524288.0],
    );
    assert_eq!(e5m2, vec![1.0 / 65536.0, 1.0 / 65536.0, 3.0 / 65536.0, 0.0]);
}

#[test]
fn a_value_above_the_midpoint_reaches_the_step_it_is_closest_to() {
    assert_eq!(round_trip(Element::Fp8E4M3, &[1.0625]), vec![1.125]);
    assert_eq!(round_trip(Element::Fp8E4M3, &[1.125]), vec![1.125]);
    assert_eq!(round_trip(Element::Fp8E4M3, &[1.1875]), vec![1.25]);
    assert_eq!(round_trip(Element::Fp8E4M3, &[1.15625]), vec![1.125]);
    assert_eq!(round_trip(Element::Fp8E5M2, &[1.125]), vec![1.25]);
    assert_eq!(round_trip(Element::Fp8E5M2, &[1.25]), vec![1.25]);
    assert_eq!(round_trip(Element::Fp8E5M2, &[1.375]), vec![1.5]);
}

#[test]
fn an_overflow_saturates_and_a_nan_stays_a_nan() {
    let e4m3 = round_trip(Element::Fp8E4M3, &[700.0, -700.0, 448.0, f32::NAN]);
    assert_eq!(e4m3[0], 448.0);
    assert_eq!(e4m3[1], -448.0);
    assert_eq!(e4m3[2], 448.0);
    assert!(e4m3[3].is_nan());
    let e5m2 = round_trip(Element::Fp8E5M2, &[70000.0, -70000.0, 57344.0, f32::NAN]);
    assert_eq!(e5m2[0], 57344.0);
    assert_eq!(e5m2[1], -57344.0);
    assert_eq!(e5m2[2], 57344.0);
    assert!(e5m2[3].is_nan());
}

#[test]
fn every_lane_of_a_word_keeps_its_own_element() {
    let values = [1.0f32, 2.0, 4.0, 8.0, -1.0, -2.0, -4.0, -8.0];
    assert_eq!(round_trip(Element::Fp8E4M3, &values), values);
    assert_eq!(round_trip(Element::Fp8E5M2, &values), values);
}

#[test]
fn a_four_bit_word_holds_eight_numbers_beside_the_quantum_their_block_shares() {
    let values = (0..200)
        .map(|index| (index as f32 / 200.0) * 2.0 - 1.0)
        .collect::<Vec<_>>();
    let bytes = pack(Element::Int4, 1.0, &values);
    assert_eq!(
        bytes.len() as u64,
        Element::Int4.storage_words(200) * 4,
        "a block of {INT4_BLOCK} int4 numbers packs its quantum beside the words they share",
    );
    let quantized = unpack(Element::Int4, values.len(), &bytes);
    for (index, (value, quantized)) in values.iter().zip(&quantized).enumerate() {
        let block = &values[index / INT4_BLOCK as usize * INT4_BLOCK as usize..];
        let peak = block[..block.len().min(INT4_BLOCK as usize)]
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        assert!(
            (value - quantized).abs() <= peak / 14.0 + f32::EPSILON,
            "a number below the peak of its block lands within half a step of it",
        );
    }
    assert_eq!(quantized[199], values[199]);
    let saturated = unpack(
        Element::Int4,
        8,
        &pack(
            Element::Int4,
            1.0,
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0],
        ),
    );
    assert_eq!(saturated, vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
}

#[test]
fn a_four_bit_float_word_holds_eight_numbers_beside_the_quantum_their_block_shares() {
    let values = (0..200)
        .map(|index| (index as f32 / 200.0) * 6.0 - 3.0)
        .collect::<Vec<_>>();
    let bytes = pack(Element::Fp4E2M1, 1.0, &values);
    assert_eq!(
        bytes.len() as u64,
        Element::Fp4E2M1.storage_words(200) * 4,
        "a block of {FP4_BLOCK} four bit floats packs its quantum beside the words they share",
    );
    let quantized = unpack(Element::Fp4E2M1, values.len(), &bytes);
    for (index, (value, quantized)) in values.iter().zip(&quantized).enumerate() {
        let block = &values[index / FP4_BLOCK as usize * FP4_BLOCK as usize..];
        let peak = block[..block.len().min(FP4_BLOCK as usize)]
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        assert!(
            (value - quantized).abs() <= peak / 6.0 + f32::EPSILON,
            "a number below the peak of its block lands within the step its four bits hold",
        );
    }
}

#[test]
fn a_four_bit_float_walks_the_grid_its_exponent_and_mantissa_name() {
    let grid = [
        0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    assert_eq!(
        round_trip(Element::Fp4E2M1, &grid),
        grid,
        "the sixteen codes of a four bit float are its own numbers",
    );
    let rounded = unpack(
        Element::Fp4E2M1,
        8,
        &pack(
            Element::Fp4E2M1,
            1.0,
            &[0.2, 0.6, 1.2, 5.7, -0.2, -0.6, 2.9, 6.0],
        ),
    );
    assert_eq!(rounded, vec![0.0, 0.5, 1.0, 6.0, 0.0, -0.5, 3.0, 6.0]);
}

#[test]
fn a_block_of_zeros_quantizes_to_zero_beside_a_block_that_does_not() {
    let mut values = vec![0.0f32; INT4_BLOCK as usize];
    values.extend((0..16).map(|index| (index as f32 - 8.0) / 2.0));
    let quantized = round_trip(Element::Int4, &values);
    assert_eq!(
        &quantized[..INT4_BLOCK as usize],
        vec![0.0f32; INT4_BLOCK as usize].as_slice(),
        "a block of zeros reconstructs as zeros rather than through the quantum of its neighbour",
    );
    assert_eq!(quantized[INT4_BLOCK as usize], -4.0);
}

fn q3_k_scales(field: &[u8]) -> [i8; 16] {
    let mut aux = [0u8; 16];
    aux[..12].copy_from_slice(&field[96..108]);
    let words = std::array::from_fn::<u32, 4, _>(|at| {
        u32::from_le_bytes(aux[at * 4..at * 4 + 4].try_into().expect("a word"))
    });
    let tmp = words[2];
    let masks = (0x0303_0303u32, 0x0f0f_0f0fu32);
    let packed = [
        (words[0] & masks.1) | ((tmp & masks.0) << 4),
        (words[1] & masks.1) | (((tmp >> 2) & masks.0) << 4),
        ((words[0] >> 4) & masks.1) | (((tmp >> 4) & masks.0) << 4),
        ((words[1] >> 4) & masks.1) | (((tmp >> 6) & masks.0) << 4),
    ];
    std::array::from_fn(|at| (packed[at / 4] >> ((at % 4) * 8)) as u8 as i8 - 32)
}

fn q3_k_reference(field: &[u8]) -> Vec<f32> {
    let d = half::f16::from_bits(u16::from_le_bytes([field[108], field[109]])).to_f32();
    let scales = q3_k_scales(field);
    let mut out = Vec::with_capacity(256);
    let mut is = 0usize;
    let mut m = 1u8;
    for half in 0..2 {
        let q = &field[32 + half * 32..32 + half * 32 + 32];
        let mut shift = 0u32;
        for _ in 0..4 {
            let low = d * f32::from(scales[is]);
            is += 1;
            for l in 0..16 {
                let quant = ((q[l] >> shift) & 3) as i32 - if field[l] & m == 0 { 4 } else { 0 };
                out.push(low * quant as f32);
            }
            let high = d * f32::from(scales[is]);
            is += 1;
            for l in 0..16 {
                let quant =
                    ((q[l + 16] >> shift) & 3) as i32 - if field[l + 16] & m == 0 { 4 } else { 0 };
                out.push(high * quant as f32);
            }
            shift += 2;
            m <<= 1;
        }
    }
    out
}

#[test]
fn a_three_bit_super_block_walks_the_high_bits_and_scales_a_gguf_packs() {
    let mut field = vec![0u8; 112];
    for (at, byte) in field[..32].iter_mut().enumerate() {
        *byte = (at as u8).wrapping_mul(37).wrapping_add(11);
    }
    for (at, byte) in field[32..96].iter_mut().enumerate() {
        *byte = (at as u8).wrapping_mul(53).wrapping_add(7);
    }
    for (at, byte) in field[96..108].iter_mut().enumerate() {
        *byte = (at as u8).wrapping_mul(19).wrapping_add(3);
    }
    field[108..110].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
    let walked = unpack(Element::Q3K, 256, &field);
    assert_eq!(walked, q3_k_reference(&field));

    let values = (0..256)
        .map(|at| (at as f32 * 0.031).sin() * 0.75)
        .collect::<Vec<_>>();
    let quantized = round_trip(Element::Q3K, &values);
    let worst = values
        .iter()
        .zip(&quantized)
        .map(|(value, quantized)| (value - quantized).abs())
        .fold(0.0f32, f32::max);
    assert!(worst <= 0.25, "a three bit block returned {worst} away");
}
