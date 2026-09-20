mod matmul;
mod ops;

use neura_abi::{Geometry, kind};

pub const REFUSE: &str = include_str!("../shaders/refuse.wgsl");
pub const POINTWISE: &str = include_str!("../shaders/pointwise.wgsl");
pub const REDUCE: &str = include_str!("../shaders/reduce.wgsl");
pub const SOFTMAX: &str = include_str!("../shaders/softmax.wgsl");

pub struct Body {
    pub kind: u32,
    pub function: &'static str,
}

pub const BODIES: &[Body] = &[
    Body {
        kind: kind::MATMUL,
        function: "run_matmul",
    },
    Body {
        kind: kind::BINARY,
        function: "run_binary",
    },
    Body {
        kind: kind::UNARY,
        function: "run_unary",
    },
    Body {
        kind: kind::PARTIAL,
        function: "run_partial",
    },
    Body {
        kind: kind::FILL,
        function: "run_fill",
    },
    Body {
        kind: kind::BROADCAST,
        function: "run_broadcast",
    },
    Body {
        kind: kind::SUM_CHUNK,
        function: "run_sum_chunk",
    },
    Body {
        kind: kind::SUM_TO,
        function: "run_sum_to",
    },
    Body {
        kind: kind::SOFTMAX,
        function: "run_softmax",
    },
    Body {
        kind: kind::SOFTMAX_GRAD,
        function: "run_softmax_grad",
    },
    Body {
        kind: kind::LOG_SOFTMAX,
        function: "run_log_softmax",
    },
    Body {
        kind: kind::LOG_SOFTMAX_GRAD,
        function: "run_log_softmax_grad",
    },
];

pub fn body(code: u32) -> &'static str {
    BODIES
        .iter()
        .find(|body| body.kind == code)
        .unwrap_or_else(|| panic!("no device body runs the {} task", kind::name(code)))
        .function
}

pub fn fragments(geometry: Geometry) -> Vec<String> {
    vec![
        REFUSE.to_owned(),
        ops::fragment(),
        POINTWISE.to_owned(),
        matmul::family(geometry),
        REDUCE.to_owned(),
        SOFTMAX.to_owned(),
        reductions(),
    ]
}

struct Reduction {
    function: &'static str,
    combine: &'static str,
}

const REDUCTIONS: &[Reduction] = &[
    Reduction {
        function: "workgroup_sum",
        combine: "{left} + {right}",
    },
    Reduction {
        function: "workgroup_max",
        combine: "max({left}, {right})",
    },
];

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
