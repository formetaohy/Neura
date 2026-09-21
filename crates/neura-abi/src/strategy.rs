use std::fmt::Write as _;

pub const THREAD_ROW: u32 = 0;
pub const WORKGROUP_ROW: u32 = 1;
pub const THREAD_ELEMENT: u32 = 2;

pub fn declarations() -> String {
    let mut out = String::new();
    writeln!(out, "const THREAD_ROW: u32 = {THREAD_ROW}u;").unwrap();
    writeln!(out, "const WORKGROUP_ROW: u32 = {WORKGROUP_ROW}u;").unwrap();
    writeln!(out, "const THREAD_ELEMENT: u32 = {THREAD_ELEMENT}u;").unwrap();
    out
}
