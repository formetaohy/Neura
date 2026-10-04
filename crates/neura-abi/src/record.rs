use crate::NO_SLOT;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    U32,
    I32,
    F32,
    U32x4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldLayout {
    pub name: &'static str,
    pub offset: u32,
    pub ty: FieldType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordLayout {
    pub name: &'static str,
    pub size: u32,
    pub fields: &'static [FieldLayout],
}

macro_rules! record {
    ($record:ident, $fields:ident, $layout:ident, $name:literal {
        $($field:ident: $ty:ty => $kind:ident),+ $(,)?
    }) => {
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
        pub struct $record {
            $(pub $field: $ty,)+
        }

        #[derive(Clone, Copy, Debug, PartialEq, Default)]
        pub struct $fields {
            $(pub $field: $ty,)+
        }

        impl $record {
            pub fn of(fields: $fields) -> Self {
                Self { $($field: fields.$field,)+ }
            }
        }

        pub const $layout: RecordLayout = RecordLayout {
            name: $name,
            size: std::mem::size_of::<$record>() as u32,
            fields: &[$(FieldLayout {
                name: stringify!($field),
                offset: std::mem::offset_of!($record, $field) as u32,
                ty: FieldType::$kind,
            },)+],
        };
    };
}

record!(PlacementRecord, PlacementFields, PLACEMENT, "Placement" {
    tensors: u32 => U32,
    weights: u32 => U32,
});

record!(SegmentRecord, SegmentFields, SEGMENT, "Segment" {
    first: u32 => U32,
    count: u32 => U32,
    wave: u32 => U32,
});

record!(StepRecord, StepFields, STEP, "Step" {
    op: u32 => U32,
    operand: u32 => U32,
    swapped: u32 => U32,
});

record!(TaskRecord, TaskFields, TASK, "Task" {
    kind: u32 => U32,
    op: u32 => U32,
    geometry: u32 => U32,
    first: u32 => U32,
    count: u32 => U32,
    slot: u32 => U32,
    splits: u32 => U32,
    out: u32 => U32,
    extra: u32 => U32,
    a: u32 => U32,
    b: u32 => U32,
    c: u32 => U32,
    d: u32 => U32,
    e: u32 => U32,
    f: u32 => U32,
    origin: u32 => U32,
    param: f32 => F32,
    prelude: u32 => U32,
    prelude_steps: u32 => U32,
    chain: u32 => U32,
    steps: u32 => U32,
    reach_rows: u32 => U32,
    reach_columns: u32 => U32,
    stride_rows: u32 => U32,
    stride_columns: u32 => U32,
    pad_rows: u32 => U32,
    pad_columns: u32 => U32,
    axis: u32 => U32,
    offset: u32 => U32,
    wave: u32 => U32,
    split: u32 => U32,
    measure: u32 => U32,
    index: u32 => U32,
    group: u32 => U32,
    planes: u32 => U32,
    plane: u32 => U32,
    patch: u32 => U32,
    segment: u32 => U32,
    keys: u32 => U32,
    reach: u32 => U32,
});

record!(MeasureRecord, MeasureFields, MEASURE, "Measure" {
    kind: u32 => U32,
    value: u32 => U32,
    rows: u32 => U32,
    columns: u32 => U32,
});

record!(PatchRecord, PatchFields, PATCH, "Patch" {
    slots: u32 => U32,
    slots_count: u32 => U32,
    count: u32 => U32,
    segment: u32 => U32,
    values: u32 => U32,
    values_count: u32 => U32,
    tasks: u32 => U32,
    tasks_count: u32 => U32,
});

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod)]
pub struct ValueRecord {
    pub base: u32,
    pub store: u32,
    pub element: u32,
    pub table: u32,
    pub storage: u32,
    pub free: u32,
    pub source: u32,
    pub pad: u32,
    pub bounds: [u32; 4],
    pub dims: [u32; 4],
    pub strides: [u32; 4],
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct ValueFields {
    pub base: u32,
    pub store: u32,
    pub element: u32,
    pub table: u32,
    pub storage: u32,
    pub free: [u32; 4],
    pub source: [u32; 4],
    pub bounds: [u32; 4],
    pub dims: [u32; 4],
    pub strides: [u32; 4],
}

impl ValueRecord {
    pub fn of(fields: ValueFields) -> Self {
        Self {
            base: fields.base,
            store: fields.store,
            element: fields.element,
            table: fields.table,
            storage: fields.storage,
            free: pack_slots(fields.free),
            source: pack_slots(fields.source),
            pad: 0,
            bounds: fields.bounds,
            dims: fields.dims,
            strides: fields.strides,
        }
    }
}

fn pack_slots(slots: [u32; 4]) -> u32 {
    let mut packed = 0u32;
    for (axis, slot) in slots.iter().enumerate() {
        assert!(
            *slot == NO_SLOT || *slot < 1 << 8,
            "a free extent of slot {slot} outruns the byte axis {axis} of a value fills",
        );
        packed |= (*slot & 0xff) << (axis * 8);
    }
    packed
}

unsafe impl bytemuck::Zeroable for ValueRecord {
    fn zeroed() -> Self {
        Self::of(ValueFields::default())
    }
}

pub const VALUE: RecordLayout = RecordLayout {
    name: "Value",
    size: std::mem::size_of::<ValueRecord>() as u32,
    fields: &[
        FieldLayout {
            name: "base",
            offset: std::mem::offset_of!(ValueRecord, base) as u32,
            ty: FieldType::U32,
        },
        FieldLayout {
            name: "store",
            offset: std::mem::offset_of!(ValueRecord, store) as u32,
            ty: FieldType::U32,
        },
        FieldLayout {
            name: "element",
            offset: std::mem::offset_of!(ValueRecord, element) as u32,
            ty: FieldType::U32,
        },
        FieldLayout {
            name: "table",
            offset: std::mem::offset_of!(ValueRecord, table) as u32,
            ty: FieldType::U32,
        },
        FieldLayout {
            name: "storage",
            offset: std::mem::offset_of!(ValueRecord, storage) as u32,
            ty: FieldType::U32,
        },
        FieldLayout {
            name: "free",
            offset: std::mem::offset_of!(ValueRecord, free) as u32,
            ty: FieldType::U32,
        },
        FieldLayout {
            name: "source",
            offset: std::mem::offset_of!(ValueRecord, source) as u32,
            ty: FieldType::U32,
        },
        FieldLayout {
            name: "pad",
            offset: std::mem::offset_of!(ValueRecord, pad) as u32,
            ty: FieldType::U32,
        },
        FieldLayout {
            name: "bounds",
            offset: std::mem::offset_of!(ValueRecord, bounds) as u32,
            ty: FieldType::U32x4,
        },
        FieldLayout {
            name: "dims",
            offset: std::mem::offset_of!(ValueRecord, dims) as u32,
            ty: FieldType::U32x4,
        },
        FieldLayout {
            name: "strides",
            offset: std::mem::offset_of!(ValueRecord, strides) as u32,
            ty: FieldType::U32x4,
        },
    ],
};

pub const RECORDS: &[RecordLayout] = &[PLACEMENT, SEGMENT, STEP, TASK, MEASURE, PATCH, VALUE];
