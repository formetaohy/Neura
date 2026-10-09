pub const REFUSAL: u32 = 0;
pub const CURSOR: u32 = crate::REFUSAL_WORDS;
pub const FRONTIER: u32 = CURSOR + 1;
pub const SEGMENTS: u32 = FRONTIER + 1;
pub const WAVES: u32 = SEGMENTS + 1;
pub const FIRST_TASK: u32 = WAVES + 1;
pub const LAST_TASK: u32 = FIRST_TASK + 1;
pub const HEADER_WORDS: u32 = LAST_TASK - CURSOR + 1;
pub const WAVE_STRIDE: u32 = 2;
pub const COUNTERS: u32 = LAST_TASK + 1;

const _: () = assert!(
    CURSOR == crate::REFUSAL_WORDS && COUNTERS == HEADER_WORDS + crate::REFUSAL_WORDS,
    "the control block opens on the refusal word and carries the scheduler header",
);

pub const fn remaining(wave: u32) -> u32 {
    COUNTERS + WAVE_STRIDE * wave
}

pub const fn total(wave: u32) -> u32 {
    remaining(wave) + 1
}

pub const fn offset_bytes() -> u64 {
    CURSOR as u64 * crate::WORD_BYTES
}

pub fn header(segments: u32, waves: u32) -> Vec<u8> {
    header_words(0, 0, segments, waves, 0, u32::MAX)
}

pub fn window(
    first_segment: u32,
    segments: u32,
    first_wave: u32,
    waves: u32,
    first_task: u32,
    last_task: u32,
) -> Vec<u8> {
    header_words(
        first_segment,
        first_wave,
        first_segment + segments,
        waves,
        first_task,
        last_task,
    )
}

fn header_words(
    cursor: u32,
    frontier: u32,
    segments: u32,
    waves: u32,
    first_task: u32,
    last_task: u32,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HEADER_WORDS as usize * crate::WORD_BYTES as usize);
    for word in [cursor, frontier, segments, waves, first_task, last_task] {
        bytes.extend_from_slice(&word.to_ne_bytes());
    }
    bytes
}

pub fn words(segments: u32, wave_tasks: &[u32]) -> Vec<u32> {
    let mut words = vec![0u32; (REFUSAL + 1) as usize];
    words.extend(progress(segments, wave_tasks));
    words
}

pub fn progress(segments: u32, wave_tasks: &[u32]) -> Vec<u32> {
    let mut words =
        vec![0u32; (COUNTERS - CURSOR + WAVE_STRIDE * wave_tasks.len() as u32) as usize];
    words[(SEGMENTS - CURSOR) as usize] = segments;
    words[(WAVES - CURSOR) as usize] = wave_tasks.len() as u32;
    for (wave, tasks) in wave_tasks.iter().enumerate() {
        assert!(
            *tasks > 0,
            "wave {wave} holds no task a device program could gate on",
        );
        let wave = wave as u32;
        words[(remaining(wave) - CURSOR) as usize] = *tasks;
        words[(total(wave) - CURSOR) as usize] = *tasks;
    }
    words
}

pub const fn bytes(waves: u32) -> u64 {
    (COUNTERS + WAVE_STRIDE * waves) as u64 * crate::WORD_BYTES
}
