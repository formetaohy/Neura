mod constants {
    include!(concat!(env!("OUT_DIR"), "/constants.rs"));
}
mod records {
    include!(concat!(env!("OUT_DIR"), "/records.rs"));
}
pub mod kind;
pub mod op;
mod placement;
mod precision;
mod profile;
pub mod store;
pub mod strategy;
mod window;

pub const TAPE_WGSL: &str = include_str!("../abi/program.wgsl");

pub use constants::*;
pub use kind::Kind;
pub use placement::Placement;
pub use precision::Precision;
pub use profile::{Geometry, MEDIUM, MatmulTile, NARROW, PROFILES, Profile, WIDE};
pub use records::{
    BoundsRecord, PlacementRecord, SegmentRecord, StepRecord, TaskRecord, ValueRecord,
};
pub use store::Store;
pub use window::Window;

pub const WORD_BYTES: u64 = 4;
pub const REFUSAL_BYTES: u64 = REFUSAL_WORDS as u64 * WORD_BYTES;
