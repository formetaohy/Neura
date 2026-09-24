use std::fmt::Write as _;

macro_rules! kinds {
    ($($variant:ident = $label:literal;)+) => {
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

        }
    };
}

kinds! {
    Matmul = "matmul";
    MatmulFold = "matmul_fold";
    Binary = "binary";
    Unary = "unary";
    Partial = "partial";
    Fill = "fill";
    Broadcast = "broadcast";
    SumChunk = "sum_chunk";
    SumAxis = "sum_axis";
    Softmax = "softmax";
    SoftmaxGrad = "softmax_grad";
    LogSoftmax = "log_softmax";
    LogSoftmaxGrad = "log_softmax_grad";
    Argmax = "argmax";
    Categorical = "categorical";
    OneHot = "one_hot";
    Gather = "gather";
    Scatter = "scatter";
    Conv2d = "conv2d";
    Conv2dInputGrad = "conv2d_input_grad";
    Conv2dWeightGrad = "conv2d_weight_grad";
}

pub fn declarations() -> String {
    let mut out = String::new();
    for kind in Kind::ALL {
        writeln!(out, "const {}: u32 = {}u;", kind.constant(), kind.code()).unwrap();
    }
    out
}
