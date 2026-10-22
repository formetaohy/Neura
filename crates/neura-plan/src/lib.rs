include!(concat!(env!("OUT_DIR"), "/plan_build.rs"));

mod access;
mod authored;
mod domain;
mod flow;
mod fuse;
mod hazard;
mod layout;
mod lower;
mod pages;
mod plan;
mod product;
mod record;
mod region;
mod remembered;
mod schedule;
mod span;

pub use layout::{Arena, Entry, Layout, Seed};
pub use pages::WeightPages;
pub use plan::{Encoding, Plan, Quantum, Span};
pub use product::{Contiguous, Product, Weight};
pub use region::TableRows;
pub use remembered::DEFAULT_ENCODING_BYTES;
