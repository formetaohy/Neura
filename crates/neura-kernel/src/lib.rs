#[path = "../device/choice.rs"]
mod choice;
#[path = "../device/conv.rs"]
mod conv;
mod element;
mod matmul;
#[path = "../device/matmul.rs"]
mod matmul_device;
mod op;
#[path = "../device/op.rs"]
mod op_device;
mod pack;
#[path = "../device/pack.rs"]
mod pack_device;
#[path = "../device/pointwise.rs"]
mod pointwise;
#[path = "../device/reduce.rs"]
mod reduce;
#[path = "../device/scatter.rs"]
mod scatter;
#[path = "../device/select.rs"]
mod select;
#[path = "../device/softmax.rs"]
mod softmax;

use neura_abi::{Element, Kind};
use neura_compiler::Compiler;
use neura_profile::Geometry;

pub fn define(compiler: &mut Compiler, kinds: &[Kind], elements: &[Element], geometry: &Geometry) {
    element::define(compiler, elements);
    pointwise::define(compiler);
    op::define(compiler);
    if kinds
        .iter()
        .any(|kind| matches!(kind, Kind::Matmul | Kind::MatmulFold))
    {
        matmul_device::define(compiler);
    }
    if kinds.contains(&Kind::Matmul) {
        let (left, right) = geometry.stage_lengths();
        compiler.workgroup("matmul_left", "f32", left);
        compiler.workgroup("matmul_right", "f32", right);
        matmul::specialize(compiler, geometry);
    }
    let reduction = kinds.iter().any(|kind| {
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
    });
    if reduction {
        compiler.workgroup("reduction_scratch", "f32", geometry.workgroup());
        reduce::define(compiler);
    }
    if kinds.iter().any(|kind| {
        matches!(
            kind,
            Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad
        )
    }) {
        softmax::define(compiler);
    }
    if kinds
        .iter()
        .any(|kind| matches!(kind, Kind::Argmax | Kind::Categorical))
    {
        compiler.workgroup("choice_index", "u32", geometry.workgroup());
        choice::define(compiler);
    }
    if kinds
        .iter()
        .any(|kind| matches!(kind, Kind::OneHot | Kind::Gather))
    {
        select::define(compiler);
    }
    if kinds.iter().any(|kind| {
        matches!(
            kind,
            Kind::Conv2d | Kind::Conv2dInputGrad | Kind::Conv2dWeightGrad
        )
    }) {
        conv::define(compiler);
    }
    if kinds.contains(&Kind::Scatter) {
        scatter::define(compiler);
    }
    if kinds.contains(&Kind::Pack) {
        pack::define(compiler, elements);
    }
}
