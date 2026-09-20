use neura_abi::{
    BoundsRecord, CURSOR_BYTES, Geometry, MatmulTile, Profile, StepRecord, WORD_BYTES,
};
use neura_gpu::{BindGroup, BindGroupEntry, BufferUsages, GpuBuffer, GpuContext, PipelineHandle};
use neura_program::{Encoding, Value};
use neura_shader::{ARENA, BOUNDS, CURSOR, Megakernel, STEPS, TASKS, VALUES};
use std::mem::size_of;

pub struct Program {
    pub(crate) encoding: Encoding,
    pub(crate) kernel: PipelineHandle,
    pub(crate) group: BindGroup,
    pub(crate) cursor: GpuBuffer,
    pub(crate) arena: GpuBuffer,
    pub(crate) tape: GpuBuffer,
    pub(crate) values: GpuBuffer,
    pub(crate) bounds: GpuBuffer,
    pub(crate) steps: GpuBuffer,
}

impl Program {
    pub(crate) fn build(context: &GpuContext, encoding: Encoding) -> Self {
        let profile = encoding.profile();
        let limits = context.limits();
        let arena_bytes = encoding.arena_bytes();
        assert!(
            arena_bytes <= limits.max_storage_buffer_binding_size,
            "a tape of {} tasks lays out {arena_bytes} bytes of arena, and the device binds at most {} bytes of storage",
            encoding.task_count(),
            limits.max_storage_buffer_binding_size,
        );
        assert!(
            arena_bytes <= limits.max_buffer_size,
            "a tape of {} tasks lays out {arena_bytes} bytes of arena, and the device holds buffers of at most {} bytes",
            encoding.task_count(),
            limits.max_buffer_size,
        );
        let device = context.device();
        let arena = GpuBuffer::new(
            device,
            "neura arena",
            arena_bytes,
            BufferUsages::STORAGE
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::VERTEX
                | BufferUsages::INDIRECT,
        );
        let tape = GpuBuffer::new(
            device,
            "neura tape",
            encoding.tasks().len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let values = GpuBuffer::new(
            device,
            "neura values",
            encoding.values().len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let bounds = GpuBuffer::new(
            device,
            "neura bounds",
            encoding.bounds().len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let cursor = GpuBuffer::new(
            device,
            "neura cursor",
            CURSOR_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let steps = GpuBuffer::new(
            device,
            "neura steps",
            (encoding.steps().len() as u64).max(size_of::<StepRecord>() as u64),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let queue = context.queue();
        tape.write(queue, encoding.tasks());
        values.write(queue, encoding.values());
        bounds.write(queue, encoding.bounds());
        if !encoding.steps().is_empty() {
            steps.write(queue, encoding.steps());
        }
        for (offset, data) in encoding.initial() {
            arena.write_at(queue, *offset, bytemuck::cast_slice(data));
        }
        let geometry = Geometry::of(profile, encoding.tiles());
        let kernel = context.declare(Megakernel::assemble(geometry).program());
        let group = kernel.bind_group(&[
            BindGroupEntry {
                binding: TASKS,
                resource: tape.resource(0, tape.size()),
            },
            BindGroupEntry {
                binding: VALUES,
                resource: values.resource(0, values.size()),
            },
            BindGroupEntry {
                binding: ARENA,
                resource: arena.resource(0, arena.size()),
            },
            BindGroupEntry {
                binding: CURSOR,
                resource: cursor.resource(0, cursor.size()),
            },
            BindGroupEntry {
                binding: BOUNDS,
                resource: bounds.resource(0, size_of::<BoundsRecord>() as u64),
            },
            BindGroupEntry {
                binding: STEPS,
                resource: steps.resource(0, steps.size()),
            },
        ]);
        Self {
            encoding,
            kernel,
            arena,
            tape,
            values,
            bounds,
            steps,
            cursor,
            group,
        }
    }

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

    pub fn profile(&self) -> Profile {
        self.encoding.profile()
    }

    pub fn matmul_geometries(&self) -> Vec<(MatmulTile, u32)> {
        self.encoding.matmul_geometries()
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        self.encoding.tiles()
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
