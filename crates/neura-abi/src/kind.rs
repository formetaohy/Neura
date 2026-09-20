use std::fmt::Write as _;

macro_rules! kinds {
    (@declare $code:expr;) => {};
    (@declare $code:expr; $name:ident = $label:literal pointwise $pointwise:literal chainable $chainable:literal; $($rest:tt)*) => {
        pub const $name: u32 = $code;
        kinds!(@declare $code + 1u32; $($rest)*);
    };
    ($($name:ident = $label:literal pointwise $pointwise:literal chainable $chainable:literal;)+) => {
        kinds!(@declare 0u32; $($name = $label pointwise $pointwise chainable $chainable;)+);

        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        pub struct Kind {
            pub code: u32,
            pub constant: &'static str,
            pub name: &'static str,
            pub pointwise: bool,
            pub chainable: bool,
        }

        pub const KINDS: &[Kind] = &[$(Kind {
            code: $name,
            constant: stringify!($name),
            name: $label,
            pointwise: $pointwise,
            chainable: $chainable,
        }),+];

        pub const COUNT: u32 = KINDS.len() as u32;
    };
}

kinds! {
    MATMUL = "matmul" pointwise false chainable false;
    BINARY = "binary" pointwise true chainable true;
    UNARY = "unary" pointwise true chainable true;
    PARTIAL = "partial" pointwise true chainable false;
    FILL = "fill" pointwise true chainable false;
    BROADCAST = "broadcast" pointwise true chainable false;
    SUM_CHUNK = "sum_chunk" pointwise false chainable false;
    SUM_TO = "sum_to" pointwise true chainable false;
    SOFTMAX = "softmax" pointwise false chainable false;
    SOFTMAX_GRAD = "softmax_grad" pointwise false chainable false;
    LOG_SOFTMAX = "log_softmax" pointwise false chainable false;
    LOG_SOFTMAX_GRAD = "log_softmax_grad" pointwise false chainable false;
}

pub fn of(code: u32) -> &'static Kind {
    KINDS
        .get(code as usize)
        .filter(|kind| kind.code == code)
        .unwrap_or_else(|| panic!("kind {code} is not a declared task kind"))
}

pub fn constant(code: u32) -> &'static str {
    of(code).constant
}

pub fn name(code: u32) -> &'static str {
    of(code).name
}

pub fn pointwise(code: u32) -> bool {
    of(code).pointwise
}

pub fn declarations() -> String {
    let mut out = String::new();
    for kind in KINDS {
        writeln!(out, "const {}: u32 = {}u;", kind.constant, kind.code).unwrap();
    }
    out
}
