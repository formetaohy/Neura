pub const CURSOR: u32 = 0;
pub const FRONTIER: u32 = 1;
pub const SEGMENTS: u32 = 2;
pub const WAVES: u32 = 3;
pub const FIRST_TASK: u32 = 4;
pub const LAST_TASK: u32 = 5;
pub const WAVE_STRIDE: u32 = 2;
pub const COUNTERS: u32 = 6;

pub const fn remaining(wave: u32) -> u32 {
    COUNTERS + WAVE_STRIDE * wave
}

pub const fn total(wave: u32) -> u32 {
    remaining(wave) + 1
}

pub const fn header_bytes() -> u64 {
    COUNTERS as u64 * crate::WORD_BYTES
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
    let mut bytes = Vec::with_capacity(header_bytes() as usize);
    for word in [cursor, frontier, segments, waves, first_task, last_task] {
        bytes.extend_from_slice(&word.to_ne_bytes());
    }
    bytes
}

pub fn words(segments: u32, wave_tasks: &[u32]) -> Vec<u32> {
    let mut words = vec![0u32; (COUNTERS + WAVE_STRIDE * wave_tasks.len() as u32) as usize];
    words[SEGMENTS as usize] = segments;
    words[WAVES as usize] = wave_tasks.len() as u32;
    for (wave, tasks) in wave_tasks.iter().enumerate() {
        assert!(
            *tasks > 0,
            "wave {wave} holds no task a device program could gate on",
        );
        words[remaining(wave as u32) as usize] = *tasks;
        words[total(wave as u32) as usize] = *tasks;
    }
    words
}

pub const fn bytes(waves: u32) -> u64 {
    (COUNTERS + WAVE_STRIDE * waves) as u64 * crate::WORD_BYTES
}
