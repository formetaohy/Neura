use neura_abi::WORD_BYTES;
use neura_gpu::{BindGroup, GpuBuffer};
use neura_program::{Encoding, Value};

pub struct Program {
    pub(crate) encoding: Encoding,
    pub(crate) group: BindGroup,
    pub(crate) cursor: GpuBuffer,
    pub(crate) arena: GpuBuffer,
    pub(crate) tape: GpuBuffer,
    pub(crate) values: GpuBuffer,
    pub(crate) bounds: GpuBuffer,
    pub(crate) steps: GpuBuffer,
}

impl Program {
    pub fn arena(&self) -> &GpuBuffer {
        &self.arena
    }

    pub fn arena_bytes(&self) -> u64 {
        self.encoding.arena_bytes()
    }

    pub fn device_bytes(&self) -> u64 {
        self.arena.size()
            + self.tape.size()
            + self.values.size()
            + self.bounds.size()
            + self.steps.size()
            + self.cursor.size()
    }

    pub fn task_count(&self) -> u32 {
        self.encoding.task_count()
    }

    pub fn step_count(&self) -> u32 {
        self.encoding.step_count()
    }

    pub fn wave_count(&self) -> u32 {
        self.encoding.wave_count()
    }

    pub fn value_count(&self) -> u32 {
        self.encoding.value_count()
    }

    pub fn work(&self) -> u64 {
        self.encoding.work()
    }

    pub fn readable(&self, value: Value) -> bool {
        self.encoding.readable(value)
    }

    pub fn span(&self, value: Value) -> WordSpan {
        let span = self.encoding.span(value);
        WordSpan {
            offset: span.offset,
            elements: (span.bytes / WORD_BYTES) as u32,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct WordSpan {
    pub offset: u64,
    pub elements: u32,
}
