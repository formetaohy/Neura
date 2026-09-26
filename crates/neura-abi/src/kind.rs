macro_rules! kinds {
    ($($variant:ident $symbol:ident = $label:literal;)+) => {
        #[repr(u32)]
        #[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
        pub enum Kind {
            $($variant),+
        }

        $(pub const $symbol: u32 = Kind::$variant as u32;)+

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

            pub const fn symbol(self) -> &'static str {
                match self {
                    $(Self::$variant => concat!("kind::", stringify!($symbol))),+
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
    Matmul MATMUL = "matmul";
    MatmulFold MATMUL_FOLD = "matmul_fold";
    Attention ATTENTION = "attention";
    AttentionQueryGrad ATTENTION_QUERY_GRAD = "attention_query_grad";
    AttentionKeyGrad ATTENTION_KEY_GRAD = "attention_key_grad";
    AttentionValueGrad ATTENTION_VALUE_GRAD = "attention_value_grad";
    Binary BINARY = "binary";
    Unary UNARY = "unary";
    Partial PARTIAL = "partial";
    Fill FILL = "fill";
    Broadcast BROADCAST = "broadcast";
    SumChunk SUM_CHUNK = "sum_chunk";
    SumAxis SUM_AXIS = "sum_axis";
    Softmax SOFTMAX = "softmax";
    SoftmaxGrad SOFTMAX_GRAD = "softmax_grad";
    LogSoftmax LOG_SOFTMAX = "log_softmax";
    LogSoftmaxGrad LOG_SOFTMAX_GRAD = "log_softmax_grad";
    Argmax ARGMAX = "argmax";
    Categorical CATEGORICAL = "categorical";
    OneHot ONE_HOT = "one_hot";
    Gather GATHER = "gather";
    Scatter SCATTER = "scatter";
    Pack PACK = "pack";
    Conv2d CONV2D = "conv2d";
    Conv2dInputGrad CONV2D_INPUT_GRAD = "conv2d_input_grad";
    Conv2dWeightGrad CONV2D_WEIGHT_GRAD = "conv2d_weight_grad";
    PoolMax2d POOL_MAX2D = "pool_max2d";
    PoolMax2dInputGrad POOL_MAX2D_INPUT_GRAD = "pool_max2d_input_grad";
    PoolMean2d POOL_MEAN2D = "pool_mean2d";
    PoolMean2dInputGrad POOL_MEAN2D_INPUT_GRAD = "pool_mean2d_input_grad";
}

impl Kind {
    pub const fn takes_prelude(self) -> bool {
        matches!(
            self,
            Self::SumChunk | Self::SumAxis | Self::Argmax | Self::Categorical
        )
    }
}
