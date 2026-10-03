pub mod element;
pub mod kind;
mod placement;
pub mod progress;
mod record;
pub mod refusal;
pub mod store;
pub mod strategy;

pub use element::{Element, FP4_BLOCK, FloatFormat, INT4_BLOCK};
pub use kind::{DeviceModule, Geometry, KINDS, Kind, KindInfo};
pub use placement::Placement;
pub use record::{
    FieldLayout, FieldType, PlacementFields, PlacementRecord, RECORDS, RecordLayout, SegmentFields,
    SegmentRecord, StepFields, StepRecord, TaskFields, TaskRecord, ValueFields, ValueRecord,
};
pub use refusal::{Refusal, TENSOR};
pub use store::Store;

pub const MAX_RANK: u32 = 4;
pub const REFUSAL_WORDS: u32 = 1;
pub const NO_VALUE: u32 = u32::MAX;
pub const WORD_BYTES: u64 = 4;
pub const REFUSAL_BYTES: u64 = REFUSAL_WORDS as u64 * WORD_BYTES;
