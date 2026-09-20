mod constants {
    include!(concat!(env!("OUT_DIR"), "/constants.rs"));
}
mod records {
    include!(concat!(env!("OUT_DIR"), "/records.rs"));
}

pub const TAPE_WGSL: &str = include_str!("../abi/program.wgsl");

pub use constants::*;
pub use records::{BoundsRecord, TaskRecord, ValueRecord};

pub const WORD_BYTES: u64 = 4;
pub const BOUNDS_BYTES: u64 = BOUNDS_WORDS as u64 * WORD_BYTES;
pub const CURSOR_BYTES: u64 = CURSOR_WORDS as u64 * WORD_BYTES;

const _: () = assert!(CURSOR_WORDS == MAX_WAVES + CURSOR_WAVE_BASE);

pub const BINARY_OPS: &[u32] = &[BINARY_ADD, BINARY_MUL];
pub const UNARY_OPS: &[u32] = &[UNARY_RELU, UNARY_SQRT, UNARY_RECIP];

pub fn slot_offset(slot: u32) -> u64 {
    slot as u64 * WORD_BYTES
}

pub fn kind_name(kind: u32) -> &'static str {
    match kind {
        KIND_MATMUL => "matmul",
        KIND_BINARY => "binary",
        KIND_UNARY => "unary",
        KIND_UNARY_GRAD => "unary_grad",
        KIND_FILL => "fill",
        KIND_BROADCAST => "broadcast",
        KIND_SUM_CHUNK => "sum_chunk",
        KIND_EXPAND => "expand",
        KIND_SUM_TO => "sum_to",
        KIND_SOFTMAX => "softmax",
        KIND_SOFTMAX_GRAD => "softmax_grad",
        other => panic!("kind {other} is not a declared task kind"),
    }
}

pub fn pointwise(kind: u32) -> bool {
    match kind {
        KIND_BINARY | KIND_UNARY | KIND_UNARY_GRAD | KIND_FILL | KIND_BROADCAST | KIND_EXPAND
        | KIND_SUM_TO => true,
        KIND_MATMUL | KIND_SUM_CHUNK | KIND_SOFTMAX | KIND_SOFTMAX_GRAD => false,
        other => panic!("kind {other} is not a declared task kind"),
    }
}

pub fn binary_name(op: u32) -> &'static str {
    match op {
        BINARY_ADD => "add",
        BINARY_MUL => "mul",
        other => panic!("binary code {other} is not a declared binary op"),
    }
}

pub fn unary_name(op: u32) -> &'static str {
    match op {
        UNARY_RELU => "relu",
        UNARY_SQRT => "sqrt",
        UNARY_RECIP => "recip",
        other => panic!("unary code {other} is not a declared unary op"),
    }
}
