mod matmul;
mod op;

use neura_abi::{Geometry, Kind, Precision};

pub const REFUSE: &str = include_str!("../shader/refuse.wgsl");
pub const READ: &str = include_str!("../shader/read.wgsl");
pub const POINTWISE: &str = include_str!("../shader/pointwise.wgsl");
pub const REDUCE: &str = include_str!("../shader/reduce.wgsl");
pub const SOFTMAX: &str = include_str!("../shader/softmax.wgsl");
pub const CHOICE: &str = include_str!("../shader/choice.wgsl");
pub const SELECT: &str = include_str!("../shader/select.wgsl");
pub const CONV: &str = include_str!("../shader/conv.wgsl");
pub const SCATTER: &str = include_str!("../shader/scatter.wgsl");

pub fn body(kind: Kind) -> &'static str {
    match kind {
        Kind::Matmul => "run_matmul",
        Kind::MatmulFold => "run_matmul_fold",
        Kind::Binary => "run_binary",
        Kind::Unary => "run_unary",
        Kind::Partial => "run_partial",
        Kind::Fill => "run_fill",
        Kind::Broadcast => "run_broadcast",
        Kind::SumChunk => "run_sum_chunk",
        Kind::SumAxis => "run_sum_axis",
        Kind::Softmax => "run_softmax",
        Kind::SoftmaxGrad => "run_softmax_grad",
        Kind::LogSoftmax => "run_log_softmax",
        Kind::LogSoftmaxGrad => "run_log_softmax_grad",
        Kind::Argmax => "run_argmax",
        Kind::Categorical => "run_categorical",
        Kind::OneHot => "run_one_hot",
        Kind::Gather => "run_gather",
        Kind::Scatter => "run_scatter",
        Kind::Conv2d => "run_conv2d",
        Kind::Conv2dInputGrad => "run_conv2d_input_grad",
        Kind::Conv2dWeightGrad => "run_conv2d_weight_grad",
    }
}

pub fn fragments(kinds: &[Kind], geometry: Geometry, weights: Precision) -> Vec<String> {
    let mut fragments = vec![
        REFUSE.to_owned(),
        READ.to_owned(),
        op::fragment(),
        storage(weights),
        POINTWISE.to_owned(),
    ];
    if kinds
        .iter()
        .any(|kind| matches!(kind, Kind::Matmul | Kind::MatmulFold))
    {
        fragments.push(matmul::family(geometry));
    }
    if kinds
        .iter()
        .any(|kind| matches!(kind, Kind::SumChunk | Kind::SumAxis))
    {
        fragments.push(REDUCE.to_owned());
    }
    if kinds.iter().any(|kind| {
        matches!(
            kind,
            Kind::SumChunk
                | Kind::SumAxis
                | Kind::Softmax
                | Kind::SoftmaxGrad
                | Kind::LogSoftmax
                | Kind::LogSoftmaxGrad
                | Kind::Argmax
                | Kind::Categorical
        )
    }) {
        fragments.push(reductions());
    }
    if kinds.iter().any(|kind| {
        matches!(
            kind,
            Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad
        )
    }) {
        fragments.push(SOFTMAX.to_owned());
    }
    if kinds
        .iter()
        .any(|kind| matches!(kind, Kind::Argmax | Kind::Categorical))
    {
        fragments.push(CHOICE.to_owned());
    }
    if kinds
        .iter()
        .any(|kind| matches!(kind, Kind::OneHot | Kind::Gather))
    {
        fragments.push(SELECT.to_owned());
    }
    if kinds.iter().any(|kind| {
        matches!(
            kind,
            Kind::Conv2d | Kind::Conv2dInputGrad | Kind::Conv2dWeightGrad
        )
    }) {
        fragments.push(CONV.to_owned());
    }
    if kinds.contains(&Kind::Scatter) {
        fragments.push(SCATTER.to_owned());
    }
    fragments
}

fn storage(weights: Precision) -> String {
    let mut source = String::from(
        "
fn base_of(value: Value) -> u32 {
    return select(placement.weights, placement.tensors, value.store == STORE_TENSORS);
}

fn publish(value: Value, offset: u32, data: f32) {
    heap[base_of(value) + value.base + offset] = data;
}
",
    );
    match weights {
        Precision::Single => source.push_str(
            "
fn fetch(value: Value, offset: u32) -> f32 {
    return heap[base_of(value) + value.base + offset];
}
",
        ),
        Precision::Half => source.push_str(
            "
fn fetch(value: Value, offset: u32) -> f32 {
    if (value.store == STORE_WEIGHTS) {
        let element = value.base + offset;
        let pair = unpack2x16float(bitcast<u32>(heap[placement.weights + (element >> 1u)]));
        return select(pair.x, pair.y, (element & 1u) == 1u);
    }
    return heap[placement.tensors + value.base + offset];
}
",
        ),
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
