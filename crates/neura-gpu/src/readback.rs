use crate::buffer::GpuBuffer;
use crate::{BufferUsages, Device};
use std::sync::Mutex;
use std::sync::mpsc;
use std::time::Duration;
use wgpu::{BufferAsyncError, PollType, SubmissionIndex};

pub const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Readback {
    slots: Vec<GpuBuffer>,
    free: Mutex<Vec<usize>>,
}

impl Readback {
    pub fn new(device: &Device, bytes: u64, slots: u64) -> Self {
        assert!(
            slots > 0,
            "a readback pool of {slots} slots holds nothing the host can map",
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
                self.slots.len(),
                self.capacity(),
            )
        })
    }

    pub fn staging(&self, slot: usize) -> &GpuBuffer {
        &self.slots[slot]
    }

    pub fn finish(
        &self,
        device: &Device,
        slot: usize,
        submission: SubmissionIndex,
        completed: mpsc::Receiver<Result<(), BufferAsyncError>>,
        bytes: u64,
    ) -> Vec<u8> {
        assert!(
            bytes > 0 && bytes.is_multiple_of(4) && bytes <= self.capacity(),
            "collecting {bytes} bytes through a readback of {} bytes",
            self.capacity(),
        );
        device
            .poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(READBACK_TIMEOUT),
            })
            .unwrap_or_else(|error| panic!("waiting for the readback submission failed: {error}"));
        completed
            .recv_timeout(READBACK_TIMEOUT)
            .unwrap_or_else(|error| panic!("the readback callback never fired: {error}"))
            .unwrap_or_else(|error: BufferAsyncError| {
                panic!("mapping the readback failed: {error}")
            });
        let staging = self.staging(slot);
        let bytes = staging
            .buffer()
            .slice(..bytes)
            .get_mapped_range()
            .expect("a finished readback is mapped")
            .to_vec();
        staging.buffer().unmap();
        self.free
            .lock()
            .expect("a readback pool is never poisoned")
            .push(slot);
        bytes
    }
}
