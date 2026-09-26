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

record!(BoundsRecord, BoundsFields, BOUNDS, "Bounds" {
    first_segment: u32 => U32,
});

record!(PlacementRecord, PlacementFields, PLACEMENT, "Placement" {
    tensors: u32 => U32,
    weights: u32 => U32,
});

record!(SegmentRecord, SegmentFields, SEGMENT, "Segment" {
    first: u32 => U32,
    count: u32 => U32,
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
});

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, bytemuck::Pod)]
pub struct ValueRecord {
    pub base: u32,
    pub store: u32,
    pub element: u32,
    padding: [u8; 4],
    pub dims: [u32; 4],
    pub strides: [u32; 4],
}

#[derive(Clone, Copy, PartialEq, Debug, Default)]
pub struct ValueFields {
    pub base: u32,
    pub store: u32,
    pub element: u32,
    pub dims: [u32; 4],
    pub strides: [u32; 4],
}

impl ValueRecord {
    pub fn of(fields: ValueFields) -> Self {
        Self {
            base: fields.base,
            store: fields.store,
            element: fields.element,
            padding: [0; 4],
            dims: fields.dims,
            strides: fields.strides,
        }
    }
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

pub const RECORDS: &[RecordLayout] = &[BOUNDS, PLACEMENT, SEGMENT, STEP, TASK, VALUE];
