use std::ops::{Deref, DerefMut};
use wgpu::{CommandEncoder, CommandEncoderDescriptor, Device, Queue, SubmissionIndex};

pub struct Submission {
    encoder: CommandEncoder,
}

impl Submission {
    pub fn new(device: &Device, label: &str) -> Self {
        Self {
            encoder: device
                .create_command_encoder(&CommandEncoderDescriptor { label: Some(label) }),
        }
    }

    pub fn submit(self, queue: &Queue) -> SubmissionIndex {
        queue.submit([self.encoder.finish()])
    }
}

impl Deref for Submission {
    type Target = CommandEncoder;

    fn deref(&self) -> &Self::Target {
        &self.encoder
    }
}

impl DerefMut for Submission {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.encoder
    }
}
