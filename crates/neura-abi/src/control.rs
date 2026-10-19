pub const REFUSAL: u32 = 0;
pub const CURSOR: u32 = crate::REFUSAL_WORDS;
pub const TAIL: u32 = CURSOR + 1;
pub const COUNT: u32 = TAIL + 1;
pub const FIRST: u32 = COUNT + 1;
pub const QUEUE: u32 = FIRST + 1;
pub const GENERATION: u32 = QUEUE + 1;
pub const WAVES: u32 = GENERATION + 1;
pub const WAVE_TABLE: u32 = WAVES + 1;
pub const PREFIX: u32 = WAVE_TABLE + 1;
pub const HEADER_WORDS: u32 = PREFIX - CURSOR + 1;
pub const COUNTERS: u32 = PREFIX + 1;

pub const ENTRY_SHIFT: u32 = 24;
pub const ENTRY_MASK: u32 = (1 << ENTRY_SHIFT) - 1;
pub const MAX_SEGMENTS: u32 = ENTRY_MASK;

const _: () = assert!(
    CURSOR == crate::REFUSAL_WORDS && COUNTERS == HEADER_WORDS + crate::REFUSAL_WORDS,
    "the control block opens on the refusal word and carries the scheduler header",
);

pub const fn remaining(segment: u32) -> u32 {
    COUNTERS + segment
}

pub const fn queue(capacity: u32, slot: u32) -> u32 {
    COUNTERS + capacity + slot
}

pub const fn wave(capacity: u32, wave: u32) -> u32 {
    COUNTERS + 2 * capacity + wave
}

pub const fn offset_bytes() -> u64 {
    CURSOR as u64 * crate::WORD_BYTES
}

pub fn entry(generation: u32, segment: u32) -> u32 {
    assert!(
        segment < MAX_SEGMENTS,
        "segment {segment} outruns the {} a queue entry names",
        MAX_SEGMENTS,
    );
    ((generation & 0xff) << ENTRY_SHIFT) | segment
}

pub const fn generation_of(entry: u32) -> u32 {
    entry >> ENTRY_SHIFT
}

pub const fn segment_of(entry: u32) -> u32 {
    entry & ENTRY_MASK
}

pub fn header(
    generation: u32,
    tail: u32,
    count: u32,
    first: u32,
    queue: u32,
    waves: u32,
    wave_table: u32,
) -> Vec<u32> {
    vec![
        0,
        tail,
        count,
        first,
        queue,
        generation & 0xff,
        waves,
        wave_table,
        0,
    ]
}

pub fn bytes(capacity: u32) -> u64 {
    (COUNTERS + 3 * capacity) as u64 * crate::WORD_BYTES
}
