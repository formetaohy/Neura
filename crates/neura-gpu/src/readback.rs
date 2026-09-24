use crate::{BufferUsages, Device, GpuBuffer, Queue, SubmissionIndex};
use std::sync::Mutex;
use std::time::Duration;

pub const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Readback {
    slots: Vec<GpuBuffer>,
    free: Mutex<Vec<usize>>,
}

impl Readback {
    pub fn new(device: &Device, bytes: u64, slots: u64) -> Self {
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
            free: Mutex::new((0..slots as usize).rev().collect()),
        }
    }

    pub fn slots(&self) -> usize {
        self.slots.len()
    }

    pub fn capacity(&self) -> u64 {
        self.slots[0].size()
    }

    pub fn claim(&self) -> usize {
        let claimed = self
            .free
            .lock()
            .expect("a readback pool is never poisoned")
            .pop();
        claimed.unwrap_or_else(|| {
            panic!(
                "all {} readbacks of this runtime are in flight; collect one before pulling another, and map at most {} bytes into each",
                self.slots.len(), self.capacity(),
            )
        })
    }

    pub fn staging(&self, slot: usize) -> &GpuBuffer {
        &self.slots[slot]
    }

    pub fn finish(
        &self,
        queue: &Queue,
        slot: usize,
        submission: SubmissionIndex,
        bytes: u64,
    ) -> Vec<u8> {
        assert!(bytes > 0 && bytes.is_multiple_of(4) && bytes <= self.capacity());
        let result = self.staging(slot).read(queue, submission, bytes);
        self.free
            .lock()
            .expect("a readback pool is never poisoned")
            .push(slot);
        result
    }
}
