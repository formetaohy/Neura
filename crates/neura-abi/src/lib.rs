pub mod kind;
mod placement;
mod record;
pub mod store;
pub mod strategy;

pub use kind::Kind;
pub use placement::Placement;
pub use record::{
    BoundsFields, BoundsRecord, FieldLayout, FieldType, PlacementFields, PlacementRecord, RECORDS,
    RecordLayout, SegmentFields, SegmentRecord, StepFields, StepRecord, TaskFields, TaskRecord,
    ValueFields, ValueRecord,
};
pub use store::Store;

pub const MAX_RANK: u32 = 4;
pub const MAX_DISPATCH_SEGMENTS: u32 = 65_535;
pub const REFUSAL_WORDS: u32 = 1;
pub const NO_VALUE: u32 = u32::MAX;
pub const WORD_BYTES: u64 = 4;
pub const REFUSAL_BYTES: u64 = REFUSAL_WORDS as u64 * WORD_BYTES;
