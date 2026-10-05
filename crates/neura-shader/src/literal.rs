pub(crate) fn float(value: f32) -> String {
    assert!(
        value.is_finite(),
        "a device program carries the non-finite literal {value}"
    );
    let digits = if value == value.trunc() && value.abs() < 1e7 {
        format!("{value:.1}")
    } else {
        format!("{value:e}")
    };
    format!("{digits}f")
}
