use std::fmt::Write as _;

macro_rules! kinds {
    ($($variant:ident = $label:literal pointwise $pointwise:literal chainable $chainable:literal;)+) => {
        #[repr(u32)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
        pub enum Kind {
            $($variant),+
        }

        impl Kind {
            pub const ALL: &'static [Kind] = &[$(Kind::$variant),+];
            pub const COUNT: u32 = Self::ALL.len() as u32;

            pub const fn code(self) -> u32 {
                self as u32
            }

            pub fn of(code: u32) -> Self {
                *Self::ALL.get(code as usize).unwrap_or_else(|| {
                    panic!("kind {code} is not a declared task kind")
                })
            }

            pub const fn constant(self) -> &'static str {
                match self {
                    $(Self::$variant => stringify!($variant)),+
                }
            }

            pub const fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $label),+
                }
            }

            pub const fn pointwise(self) -> bool {
                match self {
                    $(Self::$variant => $pointwise),+
                }
            }

            pub const fn chainable(self) -> bool {
                match self {
                    $(Self::$variant => $chainable),+
                }
            }
        }
    };
}

kinds! {
    Matmul = "matmul" pointwise false chainable false;
    MatmulFold = "matmul_fold" pointwise false chainable false;
    Binary = "binary" pointwise true chainable true;
    Unary = "unary" pointwise true chainable true;
    Partial = "partial" pointwise true chainable false;
    Fill = "fill" pointwise true chainable false;
    Broadcast = "broadcast" pointwise true chainable false;
    SumChunk = "sum_chunk" pointwise false chainable false;
    SumAxis = "sum_axis" pointwise false chainable false;
    Concat = "concat" pointwise false chainable false;
    Accumulate = "accumulate" pointwise false chainable false;
    Softmax = "softmax" pointwise false chainable false;
    SoftmaxGrad = "softmax_grad" pointwise false chainable false;
    LogSoftmax = "log_softmax" pointwise false chainable false;
    LogSoftmaxGrad = "log_softmax_grad" pointwise false chainable false;
    Argmax = "argmax" pointwise false chainable false;
    Categorical = "categorical" pointwise false chainable false;
    OneHot = "one_hot" pointwise false chainable false;
    Gather = "gather" pointwise false chainable false;
    Scatter = "scatter" pointwise false chainable false;
    Conv2d = "conv2d" pointwise false chainable false;
    Conv2dInputGrad = "conv2d_input_grad" pointwise false chainable false;
    Conv2dWeightGrad = "conv2d_weight_grad" pointwise false chainable false;
}

pub fn declarations() -> String {
    let mut out = String::new();
    for kind in Kind::ALL {
        writeln!(out, "const {}: u32 = {}u;", kind.constant(), kind.code()).unwrap();
    }
    out
}
