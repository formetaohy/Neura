pub const PAGE_SHIFT: u32 = 12;
pub const PAGE_WORDS: u64 = 1 << PAGE_SHIFT;
pub const PAGE_MASK: u64 = PAGE_WORDS - 1;
pub const NO_PAGE: u32 = u32::MAX;

pub const fn pages_of(words: u64) -> u64 {
    words.div_ceil(PAGE_WORDS)
}
