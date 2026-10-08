use neura_abi::WORD_BYTES;
use neura_gpu::{BufferUsages, GpuBuffer, GpuContext};
use neura_kernel::Banks;
use std::sync::{Arc, Mutex};

pub(crate) fn default_bank_bytes(binding_bytes: u64) -> u64 {
    assert!(
        binding_bytes > 0,
        "a device binding of no bytes splits no device heap",
    );
    1 << (63 - binding_bytes.leading_zeros())
}

#[derive(Clone, Copy)]
struct Block {
    word: u64,
    words: u64,
}

pub(crate) struct Heap {
    buffer: GpuBuffer,
    words: u64,
    bank_bytes: u64,
    banks: Banks,
    stride: u64,
    free: Mutex<Vec<Block>>,
}

impl Heap {
    pub(crate) fn new(context: &GpuContext, bytes: u64, bank_bytes: u64) -> Self {
        let words = bytes / WORD_BYTES;
        assert!(
            words > 0,
            "a device heap of {bytes} bytes holds no word a plan can address",
        );
        let alignment = context.binding_alignment();
        assert!(
            bank_bytes.is_power_of_two(),
            "a heap bank of {bank_bytes} bytes splits no device word address",
        );
        assert!(
            bank_bytes.is_multiple_of(alignment),
            "a heap bank of {bank_bytes} bytes is no multiple of the {alignment} bytes a storage binding demands",
        );
        assert!(
            bank_bytes <= context.limits().max_storage_buffer_binding_size,
            "a heap bank of {bank_bytes} bytes outruns the {} bytes one storage binding of this device holds",
            context.limits().max_storage_buffer_binding_size,
        );
        let bank_words = bank_bytes / WORD_BYTES;
        let banks = Banks::of(
            u32::try_from(words.div_ceil(bank_words)).unwrap_or_else(|_| {
                panic!("a heap of {bytes} bytes spans more banks than a device address holds")
            }),
            bank_words.trailing_zeros(),
        );
        let stride = (alignment / WORD_BYTES).max(1);
        let buffer = GpuBuffer::new(
            context.device(),
            "neura heap",
            words * WORD_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        Self {
            buffer,
            words,
            bank_bytes,
            banks,
            stride,
            free: Mutex::new(vec![Block { word: 0, words }]),
        }
    }

    pub(crate) fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub(crate) fn banks(&self) -> Banks {
        self.banks
    }

    pub(crate) fn bank_bytes(&self) -> u64 {
        self.bank_bytes
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.words * WORD_BYTES
    }

    pub(crate) fn holds(&self, words: u64) -> bool {
        let wanted = words.max(1).next_multiple_of(self.stride);
        self.free
            .lock()
            .expect("a device heap is never poisoned")
            .iter()
            .any(|block| block.words >= wanted)
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
