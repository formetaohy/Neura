#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum DeviceModule {
    Matmul,
    MatmulTiles,
    Attention,
    Reduce,
    Softmax,
    Choice,
    Select,
    Conv,
    Scatter,
    Layout,
    Pool,
    Convert,
    Scan,
    Rows,
    MatmulWeight,
}

impl DeviceModule {
    pub const ALL: &'static [DeviceModule] = &[
        Self::Matmul,
        Self::MatmulTiles,
        Self::Attention,
        Self::Reduce,
        Self::Softmax,
        Self::Choice,
        Self::Select,
        Self::Conv,
        Self::Scatter,
        Self::Layout,
        Self::Pool,
        Self::Convert,
        Self::Scan,
        Self::Rows,
        Self::MatmulWeight,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Matmul => "matmul",
            Self::MatmulTiles => "matmul_tiles",
            Self::Attention => "attention",
            Self::Reduce => "reduce",
            Self::Softmax => "softmax",
            Self::Choice => "choice",
            Self::Select => "select",
            Self::Conv => "conv",
            Self::Scatter => "scatter",
            Self::Layout => "layout",
            Self::Pool => "pool",
            Self::Convert => "convert",
            Self::Scan => "scan",
            Self::Rows => "rows",
            Self::MatmulWeight => "matmul_weight",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Geometry {
    None,
    Product,
    Attention,
    Strategy,
    Access,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct KindInfo {
    pub kind: Kind,
    pub symbol: &'static str,
    pub name: &'static str,
    pub entry: &'static str,
    pub modules: &'static [DeviceModule],
    pub geometry: Geometry,
    pub prelude: bool,
    pub chain: bool,
    pub origin: bool,
}

impl KindInfo {
    pub fn carries(self, module: DeviceModule) -> bool {
        self.modules.contains(&module)
    }
}

macro_rules! kinds {
    ($(
        $variant:ident $symbol:ident = $label:literal {
            entry: $entry:literal,
            modules: [$($module:ident),* $(,)?],
            geometry: $geometry:ident,
            prelude: $prelude:literal,
            chain: $chain:literal,
            origin: $origin:literal $(,)?
        }
    );+ $(;)?) => {
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

            pub fn info(self) -> &'static KindInfo {
                &KINDS[self as usize]
            }

            pub fn symbol(self) -> &'static str {
                self.info().symbol
            }

            pub fn name(self) -> &'static str {
                self.info().name
            }

            pub fn entry(self) -> &'static str {
                self.info().entry
            }

            pub fn carries(self, module: DeviceModule) -> bool {
                self.info().carries(module)
            }

            pub fn geometry(self) -> Geometry {
                self.info().geometry
            }

            pub fn takes_prelude(self) -> bool {
                self.info().prelude
            }

            pub fn takes_chain(self) -> bool {
                self.info().chain
            }

            pub fn reads_origin(self) -> bool {
                self.info().origin
            }
        }

        pub const KINDS: &[KindInfo] = &[$(KindInfo {
            kind: Kind::$variant,
            symbol: concat!("kind::", stringify!($symbol)),
            name: $label,
            entry: $entry,
            modules: &[$(DeviceModule::$module),*],
            geometry: Geometry::$geometry,
            prelude: $prelude,
            chain: $chain,
            origin: $origin,
        }),+];
    };
}

kinds! {
    Matmul MATMUL = "matmul" {
        entry: "run_matmul",
        modules: [Matmul, MatmulTiles],
        geometry: Product,
        prelude: false,
        chain: true,
        origin: false,
    };
    MatmulFold MATMUL_FOLD = "matmul_fold" {
        entry: "run_matmul_fold",
        modules: [Matmul],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    Attention ATTENTION = "attention" {
        entry: "run_attention",
        modules: [Attention],
        geometry: Attention,
        prelude: false,
        chain: true,
        origin: true,
    };
    AttentionQueryGrad ATTENTION_QUERY_GRAD = "attention_query_grad" {
        entry: "run_attention_query_grad",
        modules: [Attention],
        geometry: Attention,
        prelude: false,
        chain: true,
        origin: true,
    };
    AttentionKeyGrad ATTENTION_KEY_GRAD = "attention_key_grad" {
        entry: "run_attention_key_grad",
        modules: [Attention],
        geometry: Attention,
        prelude: false,
        chain: true,
        origin: true,
    };
    AttentionValueGrad ATTENTION_VALUE_GRAD = "attention_value_grad" {
        entry: "run_attention_value_grad",
        modules: [Attention],
        geometry: Attention,
        prelude: false,
        chain: true,
        origin: true,
    };
    Binary BINARY = "binary" {
        entry: "run_binary",
        modules: [],
        geometry: Access,
        prelude: false,
        chain: true,
        origin: false,
    };
    Unary UNARY = "unary" {
        entry: "run_unary",
        modules: [],
        geometry: Access,
        prelude: false,
        chain: true,
        origin: false,
    };
    Partial PARTIAL = "partial" {
        entry: "run_partial",
        modules: [],
        geometry: Access,
        prelude: false,
        chain: true,
        origin: false,
    };
    PrefixChunk PREFIX_CHUNK = "prefix_chunk" {
        entry: "run_prefix_chunk",
        modules: [Scan],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    PrefixScan PREFIX_SCAN = "prefix_scan" {
        entry: "run_prefix_scan",
        modules: [Scan],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    PrefixClose PREFIX_CLOSE = "prefix_close" {
        entry: "run_prefix_close",
        modules: [Scan],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    Fill FILL = "fill" {
        entry: "run_fill",
        modules: [],
        geometry: Access,
        prelude: false,
        chain: true,
        origin: false,
    };
    Broadcast BROADCAST = "broadcast" {
        entry: "run_broadcast",
        modules: [],
        geometry: Access,
        prelude: false,
        chain: true,
        origin: false,
    };
    Layout LAYOUT = "layout" {
        entry: "run_layout",
        modules: [Layout],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    Extend EXTEND = "extend" {
        entry: "run_extend",
        modules: [Layout],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    SumChunk SUM_CHUNK = "sum_chunk" {
        entry: "run_sum_chunk",
        modules: [Reduce],
        geometry: None,
        prelude: true,
        chain: false,
        origin: false,
    };
    SumAxis SUM_AXIS = "sum_axis" {
        entry: "run_sum_axis",
        modules: [Reduce],
        geometry: Strategy,
        prelude: true,
        chain: true,
        origin: false,
    };
    Softmax SOFTMAX = "softmax" {
        entry: "run_softmax",
        modules: [Reduce, Softmax],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    SoftmaxGrad SOFTMAX_GRAD = "softmax_grad" {
        entry: "run_softmax_grad",
        modules: [Reduce, Softmax],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    LogSoftmax LOG_SOFTMAX = "log_softmax" {
        entry: "run_log_softmax",
        modules: [Reduce, Softmax],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    LogSoftmaxGrad LOG_SOFTMAX_GRAD = "log_softmax_grad" {
        entry: "run_log_softmax_grad",
        modules: [Reduce, Softmax],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    Argmax ARGMAX = "argmax" {
        entry: "run_argmax",
        modules: [Reduce, Choice],
        geometry: Strategy,
        prelude: true,
        chain: true,
        origin: false,
    };
    Categorical CATEGORICAL = "categorical" {
        entry: "run_categorical",
        modules: [Reduce, Choice],
        geometry: Strategy,
        prelude: true,
        chain: true,
        origin: false,
    };
    OneHot ONE_HOT = "one_hot" {
        entry: "run_one_hot",
        modules: [Select],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    Gather GATHER = "gather" {
        entry: "run_gather",
        modules: [Select],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    Scatter SCATTER = "scatter" {
        entry: "run_scatter",
        modules: [Scatter],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    ScatterWrite SCATTER_WRITE = "scatter_write" {
        entry: "run_scatter_write",
        modules: [Scatter],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    Compact COMPACT = "compact" {
        entry: "run_compact",
        modules: [Scatter],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    Convert CONVERT = "convert" {
        entry: "run_convert",
        modules: [Convert],
        geometry: Access,
        prelude: false,
        chain: true,
        origin: false,
    };
    Conv2d CONV2D = "conv2d" {
        entry: "run_conv2d",
        modules: [Conv],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    Conv2dInputGrad CONV2D_INPUT_GRAD = "conv2d_input_grad" {
        entry: "run_conv2d_input_grad",
        modules: [Conv],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    Conv2dWeightGrad CONV2D_WEIGHT_GRAD = "conv2d_weight_grad" {
        entry: "run_conv2d_weight_grad",
        modules: [Conv],
        geometry: Strategy,
        prelude: false,
        chain: true,
        origin: false,
    };
    PoolMax2d POOL_MAX2D = "pool_max2d" {
        entry: "run_pool2d",
        modules: [Pool],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    PoolMax2dInputGrad POOL_MAX2D_INPUT_GRAD = "pool_max2d_input_grad" {
        entry: "run_pool2d_input_grad",
        modules: [Pool],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    PoolMean2d POOL_MEAN2D = "pool_mean2d" {
        entry: "run_pool2d",
        modules: [Pool],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    PoolMean2dInputGrad POOL_MEAN2D_INPUT_GRAD = "pool_mean2d_input_grad" {
        entry: "run_pool2d_input_grad",
        modules: [Pool],
        geometry: None,
        prelude: false,
        chain: true,
        origin: false,
    };
    Concat CONCAT = "concat" {
        entry: "run_concat",
        modules: [Layout],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    Slice SLICE = "slice" {
        entry: "run_slice",
        modules: [Layout],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    Rope ROPE = "rope" {
        entry: "run_rope",
        modules: [],
        geometry: None,
        prelude: false,
        chain: true,
        origin: true,
    };
    RopeGrad ROPE_GRAD = "rope_grad" {
        entry: "run_rope_grad",
        modules: [],
        geometry: None,
        prelude: false,
        chain: true,
        origin: true,
    };
    Rows ROWS = "rows" {
        entry: "run_rows",
        modules: [Rows],
        geometry: None,
        prelude: false,
        chain: false,
        origin: false,
    };
    MatmulWeightGrad MATMUL_WEIGHT_GRAD = "matmul_weight_grad" {
        entry: "run_matmul_weight_grad",
        modules: [MatmulWeight],
        geometry: Product,
        prelude: false,
        chain: false,
        origin: false,
    };
    Select SELECT = "select" {
        entry: "run_select",
        modules: [],
        geometry: Access,
        prelude: false,
        chain: true,
        origin: false,
    };
    SegmentSum SEGMENT_SUM = "segment_sum" {
        entry: "run_segment_sum",
        modules: [Reduce],
        geometry: None,
        prelude: true,
        chain: false,
        origin: false,
    };
}
