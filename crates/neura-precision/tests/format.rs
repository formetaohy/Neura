use neura_abi::{Element, INT4_BLOCK};
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
