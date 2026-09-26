use bytemuck::Zeroable;
use neura_abi::{
    BoundsRecord, Element, Kind, PlacementRecord, SegmentRecord, StepFields, StepRecord, Store,
    TaskFields, TaskRecord, ValueFields, ValueRecord,
};
use std::mem::{align_of, offset_of, size_of};

fn refuses(action: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    std::panic::catch_unwind(action).is_err()
}

#[test]
fn records_follow_the_shader_layout() {
    assert_eq!(size_of::<ValueRecord>(), 48);
    assert_eq!(offset_of!(ValueRecord, base), 0);
    assert_eq!(offset_of!(ValueRecord, store), 4);
    assert_eq!(offset_of!(ValueRecord, dims), 16);
    assert_eq!(offset_of!(ValueRecord, strides), 32);
    assert_eq!(size_of::<TaskRecord>(), 104);
    assert_eq!(offset_of!(TaskRecord, op), 4);
    assert_eq!(offset_of!(TaskRecord, geometry), 8);
    assert_eq!(offset_of!(TaskRecord, count), 16);
    assert_eq!(offset_of!(TaskRecord, splits), 24);
    assert_eq!(offset_of!(TaskRecord, out), 28);
    assert_eq!(offset_of!(TaskRecord, extra), 32);
    assert_eq!(offset_of!(TaskRecord, a), 36);
    assert_eq!(offset_of!(TaskRecord, b), 40);
    assert_eq!(offset_of!(TaskRecord, c), 44);
    assert_eq!(offset_of!(TaskRecord, d), 48);
    assert_eq!(offset_of!(TaskRecord, e), 52);
    assert_eq!(offset_of!(TaskRecord, f), 56);
    assert_eq!(offset_of!(TaskRecord, param), 60);
    assert_eq!(offset_of!(TaskRecord, prelude), 64);
    assert_eq!(offset_of!(TaskRecord, prelude_steps), 68);
    assert_eq!(offset_of!(TaskRecord, chain), 72);
    assert_eq!(offset_of!(TaskRecord, steps), 76);
    assert_eq!(offset_of!(TaskRecord, reach_rows), 80);
    assert_eq!(offset_of!(TaskRecord, reach_columns), 84);
    assert_eq!(offset_of!(TaskRecord, stride_rows), 88);
    assert_eq!(offset_of!(TaskRecord, stride_columns), 92);
    assert_eq!(offset_of!(TaskRecord, pad_rows), 96);
    assert_eq!(offset_of!(TaskRecord, pad_columns), 100);
    assert_eq!(size_of::<StepRecord>(), 12);
    assert_eq!(offset_of!(StepRecord, op), 0);
    assert_eq!(offset_of!(StepRecord, operand), 4);
    assert_eq!(offset_of!(StepRecord, swapped), 8);
    assert_eq!(size_of::<BoundsRecord>(), 4);
    assert_eq!(offset_of!(BoundsRecord, first_segment), 0);
    assert_eq!(size_of::<SegmentRecord>(), 8);
    assert_eq!(offset_of!(SegmentRecord, first), 0);
    assert_eq!(offset_of!(SegmentRecord, count), 4);
    assert_eq!(align_of::<ValueRecord>(), 4);
    assert_eq!(align_of::<TaskRecord>(), 4);
    assert_eq!(align_of::<StepRecord>(), 4);
    assert_eq!(size_of::<PlacementRecord>(), 8);
    assert_eq!(offset_of!(PlacementRecord, tensors), 0);
    assert_eq!(offset_of!(PlacementRecord, weights), 4);
}

#[test]
fn every_task_kind_is_declared_once() {
    assert_eq!(Kind::COUNT as usize, Kind::ALL.len());
    for (code, kind) in Kind::ALL.iter().enumerate() {
        assert_eq!(
            kind.code(),
            code as u32,
            "the {} kind leaves a hole",
            kind.name()
        );
        assert_eq!(Kind::of(kind.code()), *kind);
        assert!(!kind.name().is_empty());
        assert!(!kind.symbol().is_empty());
    }
    let mut names = Kind::ALL.iter().map(|kind| kind.name()).collect::<Vec<_>>();
    let mut symbols = Kind::ALL
        .iter()
        .map(|kind| kind.symbol())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    symbols.sort_unstable();
    symbols.dedup();
    assert_eq!(names.len(), Kind::ALL.len(), "two kinds share a name");
    assert_eq!(
        symbols.len(),
        Kind::ALL.len(),
        "two kinds share a Rust constant"
    );
    assert_eq!(neura_abi::kind::MATMUL, Kind::Matmul.code());
    assert_eq!(
        neura_abi::kind::CONV2D_WEIGHT_GRAD,
        Kind::Conv2dWeightGrad.code()
    );
    assert_eq!(Kind::Matmul.name(), "matmul");
    assert_eq!(Kind::LogSoftmax.name(), "log_softmax");
    assert_eq!(Kind::Conv2d.name(), "conv2d");
    assert!(refuses(|| {
        let _ = Kind::of(Kind::COUNT);
    }));
}

#[test]
fn only_a_reduction_that_reads_its_source_once_opens_a_prelude() {
    let opening = Kind::ALL
        .iter()
        .copied()
        .filter(|kind| kind.takes_prelude())
        .collect::<Vec<_>>();
    assert_eq!(
        opening,
        vec![
            Kind::SumChunk,
            Kind::SumAxis,
            Kind::Argmax,
            Kind::Categorical,
        ],
        "a prelude replaces the tensor a task would have read, so only a task that reads it once opens one",
    );
}

#[test]
fn a_record_declares_what_the_device_reads() {
    let value = ValueRecord::of(ValueFields {
        base: 6,
        store: Store::Weights.code(),
        element: Element::Half.code(),
        dims: [1, 2, 3, 4],
        strides: [12, 6, 2, 1],
    });
    let bytes = bytemuck::bytes_of(&value);
    assert_eq!(bytes.len(), 48);
    assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 6);
    assert_eq!(
        u32::from_ne_bytes(bytes[4..8].try_into().unwrap()),
        Store::Weights.code()
    );
    assert_eq!(
        u32::from_ne_bytes(bytes[8..12].try_into().unwrap()),
        Element::Half.code()
    );
    assert_eq!(u32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 1);
    assert_eq!(u32::from_ne_bytes(bytes[20..24].try_into().unwrap()), 2);
    let task = TaskRecord::of(TaskFields {
        kind: Kind::Matmul.code(),
        op: 2,
        geometry: 2,
        first: 3,
        count: 5,
        slot: 0,
        splits: 6,
        out: 1,
        extra: 2,
        a: 2,
        b: 3,
        c: 4,
        d: 5,
        e: 6,
        f: 7,
        param: 0.5,
        prelude: 5,
        prelude_steps: 3,
        chain: 7,
        steps: 2,
        reach_rows: 1,
        reach_columns: 2,
        stride_rows: 1,
        stride_columns: 2,
        pad_rows: 3,
        pad_columns: 4,
    });
    let bytes = bytemuck::bytes_of(&task);
    assert_eq!(bytes.len(), 104);
    assert_eq!(
        u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
        Kind::Matmul.code()
    );
    assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[8..12].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 5);
    assert_eq!(u32::from_ne_bytes(bytes[24..28].try_into().unwrap()), 6);
    assert_eq!(f32::from_ne_bytes(bytes[60..64].try_into().unwrap()), 0.5);
    assert_eq!(u32::from_ne_bytes(bytes[64..68].try_into().unwrap()), 5);
    assert_eq!(u32::from_ne_bytes(bytes[68..72].try_into().unwrap()), 3);
    assert_eq!(u32::from_ne_bytes(bytes[72..76].try_into().unwrap()), 7);
    assert_eq!(u32::from_ne_bytes(bytes[76..80].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[80..84].try_into().unwrap()), 1);
    assert_eq!(u32::from_ne_bytes(bytes[84..88].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[88..92].try_into().unwrap()), 1);
    assert_eq!(u32::from_ne_bytes(bytes[92..96].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[96..100].try_into().unwrap()), 3);
    assert_eq!(u32::from_ne_bytes(bytes[100..104].try_into().unwrap()), 4);
}

#[test]
fn a_step_declares_what_the_device_applies() {
    let step = StepRecord::of(StepFields {
        op: 6,
        operand: 9,
        swapped: 1,
    });
    let bytes = bytemuck::bytes_of(&step);
    assert_eq!(bytes.len(), 12);
    assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 6);
    assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 9);
    assert_eq!(u32::from_ne_bytes(bytes[8..12].try_into().unwrap()), 1);
}

#[test]
fn every_value_lives_in_one_declared_store() {
    for (code, store) in Store::ALL.iter().enumerate() {
        assert_eq!(
            store.code(),
            code as u32,
            "the {store:?} store leaves a hole"
        );
    }
    assert_eq!(Store::Tensors.code(), neura_abi::store::TENSORS);
    assert_eq!(Store::Weights.code(), neura_abi::store::WEIGHTS);
    assert_ne!(neura_abi::store::TENSORS, neura_abi::store::WEIGHTS);
}

#[test]
fn a_zeroed_record_holds_no_number_the_shader_could_read() {
    let bounds = BoundsRecord::zeroed();
    let placement = PlacementRecord::zeroed();
    let segment = SegmentRecord::zeroed();
    let step = StepRecord::zeroed();
    let task = TaskRecord::zeroed();
    let value = ValueRecord::zeroed();
    let records: [&[u8]; 6] = [
        bytemuck::bytes_of(&bounds),
        bytemuck::bytes_of(&placement),
        bytemuck::bytes_of(&segment),
        bytemuck::bytes_of(&step),
        bytemuck::bytes_of(&task),
        bytemuck::bytes_of(&value),
    ];
    for bytes in records {
        assert!(
            bytes.iter().all(|byte| *byte == 0),
            "a zeroed record carries {bytes:?} where the shader reads every byte of it",
        );
    }
}
