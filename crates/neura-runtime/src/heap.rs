use neura_abi::WORD_BYTES;
use neura_gpu::{BufferUsages, GpuBuffer, GpuContext};
use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Block {
    pub(crate) words: u64,
    pub(crate) word: u64,
}

struct Free {
    blocks: Vec<Block>,
}

pub struct Heap {
    buffer: GpuBuffer,
    words: u64,
    stride: u64,
    free: Mutex<Free>,
}

impl Heap {
    pub(crate) fn new(context: &GpuContext, bytes: u64) -> Self {
        let words = bytes / WORD_BYTES;
        assert!(
            words > 0,
            "a device heap of {bytes} bytes holds no word the tape can address",
        );
        let stride = (context.binding_alignment() / WORD_BYTES).max(1);
        let buffer = GpuBuffer::new(
            context.device(),
            "neura heap",
            words * WORD_BYTES,
            BufferUsages::STORAGE
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::VERTEX
                | BufferUsages::INDIRECT,
        );
        Self {
            buffer,
            words,
            stride,
            free: Mutex::new(Free {
                blocks: vec![Block { word: 0, words }],
            }),
        }
    }

    pub fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn words(&self) -> u64 {
        self.words
    }

    pub fn bytes(&self) -> u64 {
        self.words * WORD_BYTES
    }

    pub(crate) fn reserve(&self, words: u64) -> Block {
        let wanted = words.max(1).next_multiple_of(self.stride);
        let mut free = self.free.lock().expect("a device heap is never poisoned");
        let index = free
            .blocks
            .iter()
            .position(|block| block.words >= wanted)
            .unwrap_or_else(|| {
                panic!(
                    "the device heap of {} bytes holds no room for {wanted} more words",
                    self.bytes(),
                )
            });
        let block = free.blocks[index];
        if block.words == wanted {
            free.blocks.remove(index);
        } else {
            free.blocks[index] = Block {
                word: block.word + wanted,
                words: block.words - wanted,
            };
        }
        Block {
            word: block.word,
            words: wanted,
        }
    }

    pub(crate) fn shrink(&self, block: Block, words: u64) -> Block {
        let kept = words.max(1).next_multiple_of(self.stride).min(block.words);
        if kept < block.words {
            self.release(Block {
                word: block.word + kept,
                words: block.words - kept,
            });
        }
        Block {
            word: block.word,
            words: kept,
        }
    }

    pub(crate) fn release(&self, block: Block) {
        if block.words == 0 {
            return;
        }
        let mut free = self.free.lock().expect("a device heap is never poisoned");
        free.blocks.push(block);
        free.blocks.sort_by_key(|block| block.word);
        let mut merged: Vec<Block> = Vec::with_capacity(free.blocks.len());
        for block in free.blocks.drain(..) {
            match merged.last_mut() {
                Some(last) if last.word + last.words >= block.word => {
                    last.words = last.words.max(block.word + block.words - last.word);
                }
                _ => merged.push(block),
            }
        }
        free.blocks = merged;
    }
}
