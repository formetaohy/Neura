mod matmul;
mod ops;

use neura_abi::{Geometry, Kind, Placement, Precision};

pub const REFUSE: &str = include_str!("../shaders/refuse.wgsl");
pub const READ: &str = include_str!("../shaders/read.wgsl");
pub const POINTWISE: &str = include_str!("../shaders/pointwise.wgsl");
pub const REDUCE: &str = include_str!("../shaders/reduce.wgsl");
pub const SOFTMAX: &str = include_str!("../shaders/softmax.wgsl");
pub const CHOICE: &str = include_str!("../shaders/choice.wgsl");
pub const SELECT: &str = include_str!("../shaders/select.wgsl");

pub fn body(kind: Kind) -> &'static str {
    match kind {
        Kind::Matmul => "run_matmul",
        Kind::Binary => "run_binary",
        Kind::Unary => "run_unary",
        Kind::Partial => "run_partial",
        Kind::Fill => "run_fill",
        Kind::Broadcast => "run_broadcast",
        Kind::SumChunk => "run_sum_chunk",
        Kind::SumTo => "run_sum_to",
        Kind::Softmax => "run_softmax",
        Kind::SoftmaxGrad => "run_softmax_grad",
        Kind::LogSoftmax => "run_log_softmax",
        Kind::LogSoftmaxGrad => "run_log_softmax_grad",
        Kind::Argmax => "run_argmax",
        Kind::Categorical => "run_categorical",
        Kind::OneHot => "run_one_hot",
        Kind::Gather => "run_gather",
    }
}

pub fn fragments(geometry: Geometry, weights: Precision, placement: Placement) -> Vec<String> {
    vec![
        REFUSE.to_owned(),
        READ.to_owned(),
        ops::fragment(),
        storage(weights, placement),
        POINTWISE.to_owned(),
        matmul::family(geometry),
        REDUCE.to_owned(),
        reductions(),
        SOFTMAX.to_owned(),
        CHOICE.to_owned(),
        SELECT.to_owned(),
    ]
}

fn storage(weights: Precision, placement: Placement) -> String {
    let mut source = String::new();
    source.push_str(
        "
fn publish(base: u32, offset: u32, data: f32) {
    heap[base + offset] = data;
}
",
    );
    match weights {
        Precision::Single => source.push_str(
            "
fn fetch(base: u32, offset: u32) -> f32 {
    return heap[base + offset];
}
",
        ),
        Precision::Half => {
            source.push_str(&format!(
                "
const HEAP_WORDS: u32 = {}u;
const WEIGHT_WORDS: u32 = {}u;

fn fetch(base: u32, offset: u32) -> f32 {{
    let address = base + offset;
    if (address < HEAP_WORDS) {{
        return heap[address];
    }}
    let element = address - HEAP_WORDS;
    let pair = unpack2x16float(bitcast<u32>(heap[WEIGHT_WORDS + (element >> 1u)]));
    return select(pair.x, pair.y, (element & 1u) == 1u);
}}
",
                placement.heap(),
                placement.weights(),
            ));
        }
    }
    source
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
