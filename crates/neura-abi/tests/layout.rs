use bytemuck::Zeroable;
use neura_abi::{
    Element, Kind, PlacementRecord, SegmentRecord, StepFields, StepRecord, Store, TaskFields,
    TaskRecord, ValueFields, ValueRecord,
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
    assert_eq!(offset_of!(ValueRecord, table), 12);
    assert_eq!(offset_of!(ValueRecord, dims), 16);
    assert_eq!(offset_of!(ValueRecord, strides), 32);
    assert_eq!(size_of::<TaskRecord>(), 120);
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
    assert_eq!(offset_of!(TaskRecord, origin), 60);
    assert_eq!(offset_of!(TaskRecord, param), 64);
    assert_eq!(offset_of!(TaskRecord, prelude), 68);
    assert_eq!(offset_of!(TaskRecord, prelude_steps), 72);
    assert_eq!(offset_of!(TaskRecord, chain), 76);
    assert_eq!(offset_of!(TaskRecord, steps), 80);
    assert_eq!(offset_of!(TaskRecord, reach_rows), 84);
    assert_eq!(offset_of!(TaskRecord, reach_columns), 88);
    assert_eq!(offset_of!(TaskRecord, stride_rows), 92);
    assert_eq!(offset_of!(TaskRecord, stride_columns), 96);
    assert_eq!(offset_of!(TaskRecord, pad_rows), 100);
    assert_eq!(offset_of!(TaskRecord, pad_columns), 104);
    assert_eq!(offset_of!(TaskRecord, axis), 108);
    assert_eq!(offset_of!(TaskRecord, offset), 112);
    assert_eq!(offset_of!(TaskRecord, wave), 116);
    assert_eq!(size_of::<StepRecord>(), 12);
    assert_eq!(offset_of!(StepRecord, op), 0);
    assert_eq!(offset_of!(StepRecord, operand), 4);
    assert_eq!(offset_of!(StepRecord, swapped), 8);
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
fn every_numeric_format_declares_how_it_packs_into_a_word() {
    assert_eq!(Element::COUNT as usize, Element::ALL.len());
    for (code, element) in Element::ALL.iter().enumerate() {
        assert_eq!(
            element.code(),
            code as u32,
            "the {} format leaves a hole",
            element.name()
        );
        assert_eq!(Element::of(element.code()), *element);
        assert!(!element.name().is_empty() && !element.symbol().is_empty());
        assert!(
            element.payload_words(4) * element.elements_per_word() >= 4,
            "four {} numbers outrun the word they pack into",
            element.name(),
        );
    }
    let mut names = Element::ALL
        .iter()
        .map(|element| element.name())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), Element::ALL.len(), "two elements share a name");
    assert_eq!(Element::Single.payload_words(5), 5);
    assert_eq!(Element::Half.payload_words(5), 3);
    assert_eq!(Element::Int8.payload_words(5), 2);
    assert_eq!(Element::Int4.payload_words(5), 1);
    assert_eq!(Element::Fp4E2M1.payload_words(5), 1);
    assert_eq!(Element::Fp4E2M1.storage_words(200), 32);
    assert!(Element::Int8.narrow() && Element::Int8.quantized());
    assert!(Element::Int4.narrow() && Element::Int4.quantized());
    assert!(Element::Fp4E2M1.narrow() && Element::Fp4E2M1.quantized());
    assert!(Element::Fp4E2M1.per_block() && !Element::Fp4E2M1.per_tensor());
    assert_eq!(Element::Fp4E2M1.block(), neura_abi::FP4_BLOCK);
    assert!(!Element::Single.narrow() && !Element::Single.quantized());
    assert_eq!(Element::Int8.storage_words(5), 3);
    assert_eq!(Element::Int4.storage_words(200), 27);
    assert_eq!(Element::Int8.promote(Element::Half), Element::Half);
    assert_eq!(Element::Half.promote(Element::Int8), Element::Half);
    assert_eq!(
        Element::Int8.promote(Element::Int8),
        Element::Single,
        "a word of int8 numbers computes as single precision",
    );
    assert_eq!(Element::Half.promote(Element::Bfloat16), Element::Single);
}

#[test]
fn every_float_grid_is_declared_once_for_the_host_and_the_device() {
    for element in Element::ALL {
        match element {
            Element::Fp8E4M3 | Element::Fp8E5M2 | Element::Fp4E2M1 => {
                let format = element.format().unwrap_or_else(|| {
                    panic!(
                        "the {} element walks a float grid the ABI does not declare",
                        element.name(),
                    )
                });
                assert!(
                    format.bias > 0,
                    "the {} grid biases its exponent by nothing",
                    element.name(),
                );
                assert!(
                    format.mantissa_bits > 0 && format.mantissa_bits < 8,
                    "the {} grid holds {} mantissa bits in a word of at most eight",
                    element.name(),
                    format.mantissa_bits,
                );
                assert!(
                    format.smallest > 0.0 && format.smallest.is_finite(),
                    "the {} grid steps its subnormals by {}",
                    element.name(),
                    format.smallest,
                );
                let ceiling = f32::from_bits(format.ceiling);
                assert!(
                    ceiling.is_finite() && ceiling > 0.0,
                    "the {} grid saturates from a ceiling of {ceiling}",
                    element.name(),
                );
                assert!(
                    format.max & 0x80 == 0,
                    "the {} grid keeps the sign beside the largest finite code it declares",
                    element.name(),
                );
            }
            Element::Single | Element::Half | Element::Bfloat16 | Element::Int8 | Element::Int4 => {
                assert!(
                    element.format().is_none(),
                    "{} storage carries no exponent and mantissa of its own",
                    element.name(),
                );
            }
        }
    }
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
fn every_task_kind_declares_the_device_code_it_runs() {
    for kind in Kind::ALL {
        let info = kind.info();
        assert_eq!(info.kind, *kind);
        assert_eq!(kind.name(), info.name);
        assert!(
            info.entry.starts_with("run_"),
            "the {} kind names its device body {}",
            kind.name(),
            info.entry,
        );
        let mut modules = info.modules.to_vec();
        let declared = modules.len();
        modules.sort_unstable_by_key(|module| module.name());
        modules.dedup();
        assert_eq!(
            modules.len(),
            declared,
            "the {} kind installs the same device module twice",
            kind.name(),
        );
    }
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
        table: 7,
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
    assert_eq!(u32::from_ne_bytes(bytes[12..16].try_into().unwrap()), 7);
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
        origin: 8,
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
        axis: 2,
        offset: 9,
        wave: 4,
    });
    let bytes = bytemuck::bytes_of(&task);
    assert_eq!(bytes.len(), 120);
    assert_eq!(
        u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
        Kind::Matmul.code()
    );
    assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[8..12].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[16..20].try_into().unwrap()), 5);
    assert_eq!(u32::from_ne_bytes(bytes[24..28].try_into().unwrap()), 6);
    assert_eq!(u32::from_ne_bytes(bytes[60..64].try_into().unwrap()), 8);
    assert_eq!(f32::from_ne_bytes(bytes[64..68].try_into().unwrap()), 0.5);
    assert_eq!(u32::from_ne_bytes(bytes[68..72].try_into().unwrap()), 5);
    assert_eq!(u32::from_ne_bytes(bytes[72..76].try_into().unwrap()), 3);
    assert_eq!(u32::from_ne_bytes(bytes[76..80].try_into().unwrap()), 7);
    assert_eq!(u32::from_ne_bytes(bytes[80..84].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[84..88].try_into().unwrap()), 1);
    assert_eq!(u32::from_ne_bytes(bytes[88..92].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[92..96].try_into().unwrap()), 1);
    assert_eq!(u32::from_ne_bytes(bytes[96..100].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[100..104].try_into().unwrap()), 3);
    assert_eq!(u32::from_ne_bytes(bytes[104..108].try_into().unwrap()), 4);
    assert_eq!(u32::from_ne_bytes(bytes[108..112].try_into().unwrap()), 2);
    assert_eq!(u32::from_ne_bytes(bytes[112..116].try_into().unwrap()), 9);
    assert_eq!(u32::from_ne_bytes(bytes[116..120].try_into().unwrap()), 4);
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
    let placement = PlacementRecord::zeroed();
    let segment = SegmentRecord::zeroed();
    let step = StepRecord::zeroed();
    let task = TaskRecord::zeroed();
    let value = ValueRecord::zeroed();
    let records: [&[u8]; 5] = [
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

#[test]
fn only_a_positional_task_walks_a_cursor() {
    let reading = Kind::ALL
        .iter()
        .copied()
        .filter(|kind| kind.reads_origin())
        .collect::<Vec<_>>();
    assert_eq!(
        reading,
        vec![
            Kind::Attention,
            Kind::AttentionQueryGrad,
            Kind::AttentionKeyGrad,
            Kind::AttentionValueGrad,
            Kind::Rope,
            Kind::RopeGrad,
        ],
        "a cursor names the position a block of rows starts from, and only a task that places those rows walks one",
    );
}

#[test]
fn a_refusal_carries_the_subject_it_names_and_the_reason_it_refused() {
    use neura_abi::refusal::{CODE_LIMIT, KIND_BITS};
    use neura_abi::{Refusal, TENSOR};
    assert_eq!(TENSOR, Kind::COUNT);
    for reason in Refusal::ALL {
        for subject in [0u32, Kind::Attention.code(), TENSOR] {
            assert_eq!(
                Refusal::read(reason.word(subject, 9)),
                (subject, *reason, 9),
                "the {} refusal leaves the subject it names",
                reason.name(),
            );
        }
        assert!(refuses(|| {
            let _ = reason.word(Kind::Attention.code(), CODE_LIMIT);
        }));
        assert!(refuses(|| {
            let _ = reason.word(1 << KIND_BITS, 0);
        }));
    }
    let mut names = Refusal::ALL
        .iter()
        .map(|reason| reason.name())
        .collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), Refusal::ALL.len(), "two reasons share a name");
    for (code, reason) in Refusal::ALL.iter().enumerate() {
        assert_eq!(reason.code(), code as u32, "the reasons leave a hole");
        assert_eq!(Refusal::of(code as u32), *reason);
    }
    assert!(refuses(|| {
        let _ = Refusal::of(Refusal::COUNT);
    }));
    assert!(refuses(|| {
        let _ = Refusal::read(0);
    }));
}
