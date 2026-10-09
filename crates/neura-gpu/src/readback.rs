use crate::{BufferUsages, Device, GpuBuffer};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Readback {
    slots: Vec<GpuBuffer>,
    free: Mutex<Vec<usize>>,
}

impl Readback {
    pub fn new(device: &Device, bytes: u64, slots: usize) -> Self {
        assert!(
            slots > 0 && bytes > 0 && bytes.is_multiple_of(4),
            "a readback needs aligned storage and at least one slot"
        );
        Self {
            slots: (0..slots)
                .map(|_| {
                    GpuBuffer::new(
                        device,
                        "neura readback",
                        bytes,
                        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                    )
                })
                .collect(),
            free: Mutex::new((0..slots).rev().collect()),
        }
    }

    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    pub fn capacity(&self) -> u64 {
        self.slots[0].size()
    }

    pub fn lease(self: &Arc<Self>) -> ReadbackLease {
        let slot = self
            .free
            .lock()
            .expect("a readback pool is never poisoned")
            .pop();
        let slot = slot.unwrap_or_else(|| {
            panic!(
                "all {} readbacks of this runtime are held, each holding at most {} bytes; collect or drop one before pulling another, or open the runtime with more readback slots",
                self.slots.len(),
                self.capacity(),
            )
        });
        ReadbackLease {
            readback: self.clone(),
            slot,
        }
    }

    fn release(&self, slot: usize) {
        self.free
            .lock()
            .expect("a readback pool is never poisoned")
            .push(slot);
    }
}

pub struct ReadbackLease {
    readback: Arc<Readback>,
    slot: usize,
}

impl ReadbackLease {
    pub fn buffer(&self) -> &GpuBuffer {
        &self.readback.slots[self.slot]
    }
}

impl Drop for ReadbackLease {
    fn drop(&mut self) {
        self.readback.release(self.slot);
    }
}
