use neura_abi::WORD_BYTES;
use neura_gpu::{BufferUsages, GpuBuffer, GpuContext};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
struct Block {
    word: u64,
    words: u64,
}

pub(crate) struct Heap {
    buffer: GpuBuffer,
    words: u64,
    stride: u64,
    free: Mutex<Vec<Block>>,
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
            free: Mutex::new(vec![Block { word: 0, words }]),
        }
    }

    pub(crate) fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub(crate) fn words(&self) -> u64 {
        self.words
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.words * WORD_BYTES
    }

    pub(crate) fn allocate(self: &Arc<Self>, words: u64) -> Allocation {
        let wanted = words.max(1).next_multiple_of(self.stride);
        let claimed = {
            let mut free = self.free.lock().expect("a device heap is never poisoned");
            free.iter()
                .position(|block| block.words >= wanted)
                .map(|index| {
                    let block = free[index];
                    if block.words == wanted {
                        free.remove(index);
                    } else {
                        free[index] = Block {
                            word: block.word + wanted,
                            words: block.words - wanted,
                        };
                    }
                    Block {
                        word: block.word,
                        words: wanted,
                    }
                })
                .ok_or_else(|| free.iter().map(|block| block.words).sum::<u64>())
        };
        let block = claimed.unwrap_or_else(|free| {
            panic!(
                "the device heap of {} bytes holds no room for {wanted} words beside the {} it has already handed out",
                self.bytes(),
                self.words - free,
            )
        });
        Allocation {
            lease: Arc::new(Lease {
                heap: self.clone(),
                block,
            }),
        }
    }

    fn release(&self, block: Block) {
        if block.words == 0 {
            return;
        }
        let mut free = self.free.lock().expect("a device heap is never poisoned");
        free.push(block);
        free.sort_by_key(|block| block.word);
        let mut merged: Vec<Block> = Vec::with_capacity(free.len());
        for block in free.drain(..) {
            match merged.last_mut() {
                Some(last) if last.word + last.words >= block.word => {
                    last.words = last.words.max(block.word + block.words - last.word);
                }
                _ => merged.push(block),
            }
        }
        *free = merged;
    }
}

pub(crate) struct Allocation {
    lease: Arc<Lease>,
}

struct Lease {
    heap: Arc<Heap>,
    block: Block,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.heap.release(self.block);
    }
}

impl Clone for Allocation {
    fn clone(&self) -> Self {
        Self {
            lease: self.lease.clone(),
        }
    }
}

impl Allocation {
    pub(crate) fn word(&self) -> u64 {
        self.lease.block.word
    }

    pub(crate) fn words(&self) -> u64 {
        self.lease.block.words
    }

    pub(crate) fn offset(&self) -> u64 {
        self.word() * WORD_BYTES
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.words() * WORD_BYTES
    }

    pub(crate) fn heap(&self) -> &Heap {
        &self.lease.heap
    }

    pub(crate) fn buffer(&self) -> &GpuBuffer {
        self.lease.heap.buffer()
    }

    pub(crate) fn lives_on(&self, heap: &Arc<Heap>) -> bool {
        Arc::ptr_eq(&self.lease.heap, heap)
    }
}
