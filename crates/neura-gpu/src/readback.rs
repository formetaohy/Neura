use crate::buffer::GpuBuffer;
use crate::submission::Submission;
use crate::{BufferUsages, Device, Queue};
use std::sync::mpsc;
use std::time::Duration;
use wgpu::{BufferAsyncError, MapMode, PollType};

pub const READBACK_TIMEOUT: Duration = Duration::from_secs(30);

pub struct Readback {
    staging: GpuBuffer,
}

impl Readback {
    pub fn new(device: &Device, bytes: u64) -> Self {
        Self {
            staging: GpuBuffer::new(
                device,
                "neura readback",
                bytes,
                BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            ),
        }
    }

    pub fn staging(&self) -> &GpuBuffer {
        &self.staging
    }

    pub fn capacity(&self) -> u64 {
        self.staging.size()
    }

    pub fn collect(
        &self,
        device: &Device,
        queue: &Queue,
        submission: Submission,
        bytes: u64,
    ) -> Vec<u8> {
        assert!(
            bytes > 0 && bytes.is_multiple_of(4) && bytes <= self.staging.size(),
            "collecting {bytes} bytes through a staging buffer of {} bytes",
            self.staging.size(),
        );
        let (sender, completion) = mpsc::channel();
        submission.map_buffer_on_submit(
            self.staging.buffer(),
            MapMode::Read,
            ..bytes,
            move |result| {
                let _ = sender.send(result);
            },
        );
        let index = submission.submit(queue);
        device
            .poll(PollType::Wait {
                submission_index: Some(index),
                timeout: Some(READBACK_TIMEOUT),
            })
            .unwrap_or_else(|error| panic!("waiting for the readback submission failed: {error}"));
        completion
            .recv_timeout(READBACK_TIMEOUT)
            .unwrap_or_else(|error| panic!("the readback callback never fired: {error}"))
            .unwrap_or_else(|error: BufferAsyncError| {
                panic!("mapping the readback failed: {error}")
            });
        let bytes = self
            .staging
            .buffer()
            .slice(..bytes)
            .get_mapped_range()
            .expect("a finished readback is mapped")
            .to_vec();
        self.staging.buffer().unmap();
        bytes
    }
}
