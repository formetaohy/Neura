use neura_abi::{
    BINARY_ADD, BINARY_MUL, BoundsRecord, CHAIN_ADD, CHAIN_COUNT, CHAIN_MUL, CHAIN_RELU,
    CHAIN_SQRT, KIND_COUNT, KIND_MATMUL, KIND_SOFTMAX, KIND_UNARY, StepRecord, TaskRecord,
    UNARY_RECIP, UNARY_RELU, UNARY_SQRT, ValueRecord,
};
use std::mem::{align_of, offset_of, size_of};

#[test]
fn records_follow_the_shader_layout() {
    assert_eq!(size_of::<ValueRecord>(), 48);
    assert_eq!(offset_of!(ValueRecord, base), 0);
    assert_eq!(offset_of!(ValueRecord, dims), 16);
    assert_eq!(offset_of!(ValueRecord, strides), 32);
    assert_eq!(size_of::<TaskRecord>(), 48);
    assert_eq!(offset_of!(TaskRecord, count), 12);
    assert_eq!(offset_of!(TaskRecord, a), 24);
    assert_eq!(offset_of!(TaskRecord, param), 36);
    assert_eq!(offset_of!(TaskRecord, chain), 40);
    assert_eq!(offset_of!(TaskRecord, steps), 44);
    assert_eq!(size_of::<StepRecord>(), 8);
    assert_eq!(offset_of!(StepRecord, op), 0);
    assert_eq!(offset_of!(StepRecord, operand), 4);
    assert_eq!(size_of::<BoundsRecord>(), 12);
    assert_eq!(offset_of!(BoundsRecord, first_task), 0);
    assert_eq!(offset_of!(BoundsRecord, task_count), 4);
    assert_eq!(offset_of!(BoundsRecord, wave), 8);
    assert_eq!(align_of::<ValueRecord>(), 4);
    assert_eq!(align_of::<TaskRecord>(), 4);
    assert_eq!(align_of::<StepRecord>(), 4);
}

#[test]
fn every_task_kind_is_named() {
    assert_eq!(KIND_COUNT, 11);
    for kind in 0..KIND_COUNT {
        assert!(!neura_abi::kind_name(kind).is_empty());
    }
    assert_eq!(neura_abi::kind_name(KIND_MATMUL), "matmul");
    assert_eq!(neura_abi::kind_name(KIND_SOFTMAX), "softmax");
    assert_eq!(neura_abi::kind_name(neura_abi::KIND_EXPAND), "expand");
}

#[test]
fn every_op_code_is_named() {
    for op in neura_abi::BINARY_OPS {
        let _ = neura_abi::binary_name(*op);
    }
    for op in neura_abi::UNARY_OPS {
        let _ = neura_abi::unary_name(*op);
    }
    for op in neura_abi::CHAIN_OPS {
        let _ = neura_abi::chain_name(*op);
    }
    assert_eq!(neura_abi::binary_name(BINARY_ADD), "add");
    assert_eq!(neura_abi::binary_name(BINARY_MUL), "mul");
    assert_eq!(neura_abi::unary_name(UNARY_RELU), "relu");
    assert_eq!(neura_abi::unary_name(UNARY_SQRT), "sqrt");
    assert_eq!(neura_abi::unary_name(UNARY_RECIP), "recip");
    assert_eq!(neura_abi::chain_name(CHAIN_ADD), "add");
    assert_eq!(neura_abi::chain_name(CHAIN_MUL), "mul");
    assert_eq!(neura_abi::chain_name(CHAIN_RELU), "relu");
    assert_eq!(neura_abi::chain_name(CHAIN_SQRT), "sqrt");
    assert_eq!(neura_abi::chain_name(neura_abi::CHAIN_RECIP), "recip");
}

#[test]
fn every_elementwise_op_reaches_a_chain_op() {
    let mut reached = Vec::new();
    for op in neura_abi::BINARY_OPS {
        reached.push(neura_abi::chain_op(neura_abi::KIND_BINARY, *op));
    }
    for op in neura_abi::UNARY_OPS {
        reached.push(neura_abi::chain_op(KIND_UNARY, *op));
    }
    reached.sort_unstable();
    reached.dedup();
    let mut declared = neura_abi::CHAIN_OPS.to_vec();
    declared.sort_unstable();
    assert_eq!(
        reached, declared,
        "every elementwise op must land on a chain op the device can run",
    );
    assert_eq!(declared.len() as u32, CHAIN_COUNT);
}

#[test]
fn a_record_declares_what_the_device_reads() {
    let mut task: TaskRecord = bytemuck::Zeroable::zeroed();
    task.kind = KIND_MATMUL;
    task.first = 3;
    task.count = 5;
    task.a = 2;
    task.b = 3;
    task.c = 4;
    task.out = 1;
    task.param = 0.5;
    task.chain = 7;
    task.steps = 2;
    let bytes = bytemuck::bytes_of(&task);
    assert_eq!(bytes.len(), 48);
    assert_eq!(
        u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
        KIND_MATMUL
    );
    assert_eq!(u32::from_ne_bytes(bytes[12..16].try_into().unwrap()), 5);
    assert_eq!(f32::from_ne_bytes(bytes[36..40].try_into().unwrap()), 0.5);
    assert_eq!(u32::from_ne_bytes(bytes[40..44].try_into().unwrap()), 7);
    assert_eq!(u32::from_ne_bytes(bytes[44..48].try_into().unwrap()), 2);
}

#[test]
fn a_step_declares_what_the_device_applies() {
    let step = StepRecord {
        op: CHAIN_RELU,
        operand: 9,
    };
    let bytes = bytemuck::bytes_of(&step);
    assert_eq!(bytes.len(), 8);
    assert_eq!(
        u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
        CHAIN_RELU
    );
    assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 9);
}
