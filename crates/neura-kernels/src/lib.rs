mod matmul;

use neura_abi::{
    KIND_BINARY, KIND_BROADCAST, KIND_COUNT, KIND_FILL, KIND_MATMUL, KIND_SOFTMAX,
    KIND_SOFTMAX_GRAD, KIND_SUM_CHUNK, KIND_SUM_TO, KIND_UNARY, KIND_UNARY_GRAD, Schedule,
};

pub const INDEX: &str = include_str!("../shaders/index.wgsl");
pub const CHAIN: &str = include_str!("../shaders/chain.wgsl");
pub const ELEMENTWISE: &str = include_str!("../shaders/elementwise.wgsl");
pub const REDUCE: &str = include_str!("../shaders/reduce.wgsl");
pub const SOFTMAX: &str = include_str!("../shaders/softmax.wgsl");

pub const CHAIN_OPS: &[&str] = &[
    "CHAIN_ADD",
    "CHAIN_MUL",
    "CHAIN_RELU",
    "CHAIN_SQRT",
    "CHAIN_RECIP",
];

pub struct Kernel {
    pub kind: u32,
    pub constant: &'static str,
    pub body: &'static str,
}

struct Reduction {
    function: &'static str,
    combine: &'static str,
}

const REDUCTIONS: &[Reduction] = &[
    Reduction {
        function: "reduce_chunk_sum",
        combine: "{left} + {right}",
    },
    Reduction {
        function: "softmax_row_sum",
        combine: "{left} + {right}",
    },
    Reduction {
        function: "softmax_row_max",
        combine: "max({left}, {right})",
    },
];

pub const KERNELS: &[Kernel] = &[
    Kernel {
        kind: KIND_MATMUL,
        constant: "KIND_MATMUL",
        body: "run_matmul",
    },
    Kernel {
        kind: KIND_BINARY,
        constant: "KIND_BINARY",
        body: "run_binary",
    },
    Kernel {
        kind: KIND_UNARY,
        constant: "KIND_UNARY",
        body: "run_unary",
    },
    Kernel {
        kind: KIND_UNARY_GRAD,
        constant: "KIND_UNARY_GRAD",
        body: "run_unary_grad",
    },
    Kernel {
        kind: KIND_FILL,
        constant: "KIND_FILL",
        body: "run_fill",
    },
    Kernel {
        kind: KIND_BROADCAST,
        constant: "KIND_BROADCAST",
        body: "run_broadcast",
    },
    Kernel {
        kind: KIND_SUM_CHUNK,
        constant: "KIND_SUM_CHUNK",
        body: "run_sum_chunk",
    },
    Kernel {
        kind: KIND_SUM_TO,
        constant: "KIND_SUM_TO",
        body: "run_sum_to",
    },
    Kernel {
        kind: KIND_SOFTMAX,
        constant: "KIND_SOFTMAX",
        body: "run_softmax",
    },
    Kernel {
        kind: KIND_SOFTMAX_GRAD,
        constant: "KIND_SOFTMAX_GRAD",
        body: "run_softmax_grad",
    },
];

pub fn fragments(schedule: Schedule) -> Vec<String> {
    vec![
        INDEX.to_owned(),
        CHAIN.to_owned(),
        matmul::body(schedule.matmul()),
        ELEMENTWISE.to_owned(),
        REDUCE.to_owned(),
        SOFTMAX.to_owned(),
    ]
}

pub fn kernel(kind: u32) -> &'static Kernel {
    KERNELS
        .iter()
        .find(|kernel| kernel.kind == kind)
        .unwrap_or_else(|| panic!("kind {kind} has no kernel body"))
}

pub fn kind_count() -> u32 {
    KIND_COUNT
}

pub fn reductions() -> String {
    let mut source = String::from(
        "var<workgroup> reduction_scratch: array<f32, WORKGROUP_SIZE>;

",
    );
    for reduction in REDUCTIONS {
        let combine = reduction
            .combine
            .replace("{left}", "reduction_scratch[lid]")
            .replace("{right}", "reduction_scratch[lid + stride]");
        source.push_str(&format!(
            "
fn {function}(lid: u32, start: f32) -> f32 {{
    reduction_scratch[lid] = start;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid < stride) {{
            reduction_scratch[lid] = {combine};
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let total = reduction_scratch[0];
    workgroupBarrier();
    return total;
}}
",
            function = reduction.function,
            combine = combine,
        ));
    }
    source
}
