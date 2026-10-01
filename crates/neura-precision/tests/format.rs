use neura_abi::Element;
use neura_precision::{pack, unpack};

fn round_trip(element: Element, values: &[f32]) -> Vec<f32> {
    let bytes = pack(element, 1.0, values);
    unpack(element, 1.0, values.len(), &bytes)
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
