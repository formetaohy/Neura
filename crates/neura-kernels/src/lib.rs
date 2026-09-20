use neura_abi::{
    KIND_BINARY, KIND_BROADCAST, KIND_COUNT, KIND_EXPAND, KIND_FILL, KIND_MATMUL, KIND_SOFTMAX,
    KIND_SOFTMAX_GRAD, KIND_SUM_CHUNK, KIND_SUM_TO, KIND_UNARY, KIND_UNARY_GRAD, MATMUL_COL_TILE,
    MATMUL_DEPTH_TILE, MATMUL_ROW_TILE, WORKGROUP_SIZE,
};

pub const INDEX: &str = include_str!("../shaders/index.wgsl");
pub const MATMUL: &str = include_str!("../shaders/matmul.wgsl");
pub const ELEMENTWISE: &str = include_str!("../shaders/elementwise.wgsl");
pub const REDUCE: &str = include_str!("../shaders/reduce.wgsl");
pub const SOFTMAX: &str = include_str!("../shaders/softmax.wgsl");

pub const FRAGMENTS: &[&str] = &[INDEX, MATMUL, ELEMENTWISE, REDUCE, SOFTMAX];

pub struct Kernel {
    pub kind: u32,
    pub constant: &'static str,
    pub body: &'static str,
}

struct Reduction {
    scratch: &'static str,
    function: &'static str,
    combine: &'static str,
}

const REDUCTIONS: &[Reduction] = &[
    Reduction {
        scratch: "reduce_scan",
        function: "reduce_chunk_sum",
        combine: "{left} + {right}",
    },
    Reduction {
        scratch: "softmax_scan",
        function: "softmax_row_sum",
        combine: "{left} + {right}",
    },
    Reduction {
        scratch: "softmax_scan",
        function: "softmax_row_max",
        combine: "max({left}, {right})",
    },
];

const _: () = {
    assert!(WORKGROUP_SIZE.is_power_of_two());
    assert!(MATMUL_ROW_TILE.is_multiple_of(2) && MATMUL_COL_TILE.is_multiple_of(2));
    assert!(MATMUL_ROW_TILE * MATMUL_COL_TILE / 4 == WORKGROUP_SIZE);
    assert!((MATMUL_ROW_TILE * MATMUL_DEPTH_TILE).is_multiple_of(WORKGROUP_SIZE));
    assert!((MATMUL_DEPTH_TILE * MATMUL_COL_TILE).is_multiple_of(WORKGROUP_SIZE));
};

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
        kind: KIND_EXPAND,
        constant: "KIND_EXPAND",
        body: "run_expand",
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
    let mut source = String::new();
    let mut declared = Vec::new();
    for reduction in REDUCTIONS {
        if !declared.contains(&reduction.scratch) {
            source.push_str(&format!(
                "var<workgroup> {}: array<f32, WORKGROUP_SIZE>;
",
                reduction.scratch,
            ));
            declared.push(reduction.scratch);
        }
        let combine = reduction
            .combine
            .replace("{left}", &format!("{}[lid]", reduction.scratch))
            .replace("{right}", &format!("{}[lid + stride]", reduction.scratch));
        source.push_str(&format!(
            "
fn {function}(lid: u32, start: f32) -> f32 {{
    {scratch}[lid] = start;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
    loop {{
        if (stride == 0u) {{ break; }}
        if (lid < stride) {{
            {scratch}[lid] = {combine};
        }}
        workgroupBarrier();
        stride = stride / 2u;
    }}
    let total = {scratch}[0];
    workgroupBarrier();
    return total;
}}
",
            function = reduction.function,
            scratch = reduction.scratch,
            combine = combine,
        ));
    }
    source
}
