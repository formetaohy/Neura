use crate::buffer::GpuBuffer;
use crate::context::{Device, Queue};
use crate::native::NativeBuffer;
use crate::pipeline::{BindGroup, PipelineHandle};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct SubmissionIndex(pub(crate) u64);

pub(crate) struct Write {
    pub(crate) buffer: NativeBuffer,
    pub(crate) offset: u64,
    pub(crate) bytes: Vec<u8>,
}

#[derive(Clone)]
pub(crate) enum Command {
    Copy {
        source: GpuBuffer,
        source_offset: u64,
        target: GpuBuffer,
        target_offset: u64,
        bytes: u64,
    },
    Clear {
        buffer: GpuBuffer,
        offset: u64,
        bytes: u64,
    },
    Dispatch {
        pipeline: PipelineHandle,
        group: BindGroup,
        offsets: Vec<u32>,
        groups: [u32; 3],
    },
}

pub struct Submission {
    device: Device,
    commands: Vec<Command>,
}

impl Submission {
    pub fn new(device: &Device, _label: &str) -> Self {
        Self {
            device: device.clone(),
            commands: Vec::new(),
        }
    }

    pub fn copy(
        &mut self,
        source: &GpuBuffer,
        source_offset: u64,
        target: &GpuBuffer,
        target_offset: u64,
        bytes: u64,
    ) {
        use crate::BufferUsages;
        assert!(
            source.device().same(&self.device) && target.device().same(&self.device),
            "a transfer crosses compute devices"
        );
        assert!(
            source.usage().contains(BufferUsages::COPY_SRC)
                && target.usage().contains(BufferUsages::COPY_DST),
            "a transfer requires a copy source and destination"
        );
        assert!(
            bytes > 0
                && bytes.is_multiple_of(4)
                && source_offset.is_multiple_of(4)
                && target_offset.is_multiple_of(4)
        );
        assert!(
            source_offset
                .checked_add(bytes)
                .is_some_and(|end| end <= source.size())
        );
        assert!(
            target_offset
                .checked_add(bytes)
                .is_some_and(|end| end <= target.size())
        );
        assert!(
            source.allocation() != target.allocation()
                || source_offset + bytes <= target_offset
                || target_offset + bytes <= source_offset,
            "a copy cannot overlap itself"
        );
        self.commands.push(Command::Copy {
            source: source.clone(),
            source_offset,
            target: target.clone(),
            target_offset,
            bytes,
        });
    }

    pub fn clear(&mut self, buffer: &GpuBuffer, offset: u64, bytes: u64) {
        assert!(
            buffer.device().same(&self.device),
            "a clear crosses compute devices"
        );
        assert!(
            buffer.usage().contains(crate::BufferUsages::COPY_DST),
            "a clear requires a copy destination"
        );
        assert!(offset.is_multiple_of(4) && bytes > 0 && bytes.is_multiple_of(4));
        assert!(
            offset
                .checked_add(bytes)
                .is_some_and(|end| end <= buffer.size())
        );
        self.commands.push(Command::Clear {
            buffer: buffer.clone(),
            offset,
            bytes,
        });
    }

    pub fn dispatch(
        &mut self,
        pipeline: &PipelineHandle,
        group: &BindGroup,
        offsets: &[u32],
        groups: [u32; 3],
    ) {
        assert!(
            pipeline.slot.device.same(&self.device),
            "a pipeline crosses compute devices"
        );
        assert!(
            Arc::ptr_eq(&pipeline.slot, &group.slot),
            "a group belongs to another pipeline"
        );
        let limits = self.device.limits();
        assert!(
            groups
                .iter()
                .all(|count| *count > 0 && *count <= limits.max_compute_workgroups_per_dimension),
            "a dispatch exceeds the workgroup grid"
        );
        assert_eq!(
            offsets.len(),
            group
                .buffers
                .iter()
                .filter(|binding| binding.dynamic)
                .count()
        );
        for (binding, offset) in group
            .buffers
            .iter()
            .filter(|binding| binding.dynamic)
            .zip(offsets)
        {
            let offset = u64::from(*offset);
            assert!(
                offset.is_multiple_of(limits.min_storage_buffer_offset_alignment),
                "a dynamic binding is misaligned"
            );
            assert!(
                binding
                    .offset
                    .checked_add(offset)
                    .and_then(|start| start.checked_add(binding.size))
                    .is_some_and(|end| end <= binding.buffer.size()),
                "a dynamic binding outruns its buffer"
            );
        }
        pipeline.compile();
        self.commands.push(Command::Dispatch {
            pipeline: pipeline.clone(),
            group: group.clone(),
            offsets: offsets.to_vec(),
            groups,
        });
    }

    pub fn submit(self, queue: &Queue) -> SubmissionIndex {
        assert!(
            self.device.same(queue.device()),
            "a submission crosses compute devices"
        );
        queue.submit(&self.commands)
    }
}
