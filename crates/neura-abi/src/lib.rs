pub mod control;
pub mod digest;
pub mod element;
pub mod kind;
pub mod noise;
pub mod page;
mod placement;
mod record;
pub mod refusal;
pub mod span;
pub mod store;
pub mod strategy;

pub use digest::Digest;
pub use element::{
    Block, DELTA_WIDTH, Element, FP4_BLOCK, FloatFormat, INT4_BLOCK, QUANT_BLOCK, SUPER_BLOCK,
    SUPER_SCALE_BYTES, SUPER_SMALL_BLOCK, SUPER_SUB_BLOCK,
};
pub use kind::{DeviceModule, Geometry, KINDS, Kind, KindInfo};
pub use noise::Distribution;
pub use page::{NO_PAGE, PAGE_MASK, PAGE_SHIFT, PAGE_WORDS, pages_of};
pub use placement::Placement;
pub use record::{
    FieldLayout, FieldType, MEASURE, MeasureFields, MeasureRecord, PATCH, PLACEMENT, PatchFields,
    PatchRecord, PlacementFields, PlacementRecord, RECORDS, RecordLayout, SegmentFields,
    SegmentRecord, StepFields, StepRecord, TASK, TaskFields, TaskRecord, VALUE, ValueFields,
    ValueRecord,
};
pub use refusal::{Refusal, TENSOR};
pub use span::{measure, split};
pub use store::Store;

pub const MAX_RANK: u32 = 4;
pub const EXACT_WALK_LIMIT: u32 = 1 << 24;
pub const REFUSAL_WORDS: u32 = 1;
pub const NO_VALUE: u32 = u32::MAX;
pub const NO_SLOT: u32 = u8::MAX as u32;
pub const WORD_BYTES: u64 = 4;
pub const CANDIDATES: u32 = 64;
pub const CHOICE_ROUND: u32 = 4;
pub const CHOICE_SECTIONS: u32 = 64;
pub const CHOICE_STREAM: u32 = 4;
pub const SOFTMAX_STREAM: u32 = 4;
pub const CHOICE_MERGE_WORDS: u32 = 5 * CHOICE_SECTIONS + 3 * CANDIDATES;

const _: () = assert!(
    CHOICE_ROUND == MAX_RANK,
    "a round of the choice family hands back the lanes of one device vector",
);
