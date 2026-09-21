mod matmul;
mod ops;

use neura_abi::{Geometry, Kind, Precision};

pub const REFUSE: &str = include_str!("../shaders/refuse.wgsl");
pub const READ: &str = include_str!("../shaders/read.wgsl");
pub const POINTWISE: &str = include_str!("../shaders/pointwise.wgsl");
pub const REDUCE: &str = include_str!("../shaders/reduce.wgsl");
pub const SOFTMAX: &str = include_str!("../shaders/softmax.wgsl");
pub const CHOICE: &str = include_str!("../shaders/choice.wgsl");
pub const SELECT: &str = include_str!("../shaders/select.wgsl");
pub const CONV: &str = include_str!("../shaders/conv.wgsl");

pub fn body(kind: Kind) -> &'static str {
    match kind {
        Kind::Matmul => "run_matmul",
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
        Kind::Conv2d => "run_conv2d",
        Kind::Conv2dInputGrad => "run_conv2d_input_grad",
        Kind::Conv2dWeightGrad => "run_conv2d_weight_grad",
    }
}

pub fn fragments(geometry: Geometry, weights: Precision) -> Vec<String> {
    vec![
        REFUSE.to_owned(),
        READ.to_owned(),
        ops::fragment(),
        storage(weights),
        POINTWISE.to_owned(),
        matmul::family(geometry),
        REDUCE.to_owned(),
        reductions(),
        SOFTMAX.to_owned(),
        CHOICE.to_owned(),
        SELECT.to_owned(),
        CONV.to_owned(),
    ]
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
