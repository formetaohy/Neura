mod constants {
    include!(concat!(env!("OUT_DIR"), "/constants.rs"));
}
mod records {
    include!(concat!(env!("OUT_DIR"), "/records.rs"));
}
pub mod kind;
mod placement;
pub mod store;
pub mod strategy;

pub const TAPE_WGSL: &str = include_str!("../abi/program.wgsl");

pub use constants::*;
pub use kind::Kind;
pub use placement::Placement;
pub use records::{
    BoundsFields, BoundsRecord, PlacementFields, PlacementRecord, SegmentFields, SegmentRecord,
    StepFields, StepRecord, TaskFields, TaskRecord, ValueFields, ValueRecord,
};
pub use store::Store;

pub const WORD_BYTES: u64 = 4;
pub const REFUSAL_BYTES: u64 = REFUSAL_WORDS as u64 * WORD_BYTES;
