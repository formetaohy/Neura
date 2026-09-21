use crate::heap::{Block, Heap};
use neura_abi::{
    BoundsRecord, CURSOR_BYTES, Geometry, MatmulTile, Placement, Precision, Profile, StepRecord,
    WORD_BYTES,
};
use neura_gpu::{
    BindGroup, BindGroupEntry, BufferUsages, GpuBuffer, GpuContext, PipelineHandle, Submission,
};
use neura_program::{Encoding, Region, Store, Value};
use neura_shader::{BOUNDS, CURSOR, HEAP, Megakernel, STEPS, TASKS, VALUES};
use std::mem::size_of;
use std::sync::Arc;

#[derive(Clone)]
pub struct Weights {
    heap: Arc<Heap>,
    block: Block,
    region: Region,
    precision: Precision,
}

impl Weights {
    pub(crate) fn new(heap: Arc<Heap>, block: Block, region: Region, precision: Precision) -> Self {
        Self {
            heap,
            block,
            region,
            precision,
        }
    }

    pub fn buffer(&self) -> &GpuBuffer {
        self.heap.buffer()
    }

    pub fn offset(&self) -> u64 {
        self.block.word * WORD_BYTES
    }

    pub fn words(&self) -> u64 {
        self.block.words
    }

    pub fn bytes(&self) -> u64 {
        self.block.words * WORD_BYTES
    }

    pub fn tensors(&self) -> usize {
        self.region.tensors()
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    pub(crate) fn region(&self) -> &Region {
        &self.region
    }

    pub(crate) fn lives_on(&self, heap: &Arc<Heap>) -> bool {
        Arc::ptr_eq(&self.heap, heap)
    }
}

pub struct Program {
    pub(crate) encoding: Encoding,
    pub(crate) kernel: PipelineHandle,
    pub(crate) group: BindGroup,
    pub(crate) cursor: GpuBuffer,
    pub(crate) heap: Arc<Heap>,
    pub(crate) tensors: Block,
    pub(crate) weights: Weights,
    pub(crate) tape: GpuBuffer,
    pub(crate) values: GpuBuffer,
    pub(crate) bounds: GpuBuffer,
    pub(crate) steps: GpuBuffer,
}

impl Program {
    pub(crate) fn build(
        context: &GpuContext,
        encoding: Encoding,
        heap: Arc<Heap>,
        tensors: Block,
        weights: Weights,
    ) -> Self {
        let limits = context.limits();
        for (name, bytes) in [
            ("device heap", heap.bytes()),
            ("tape", encoding.tasks().len() as u64),
            ("values", encoding.values().len() as u64),
        ] {
            assert!(
                bytes <= limits.max_storage_buffer_binding_size,
                "a tape of {} tasks binds {bytes} bytes of {name}, and the device binds at most {} bytes of storage",
                encoding.task_count(),
                limits.max_storage_buffer_binding_size,
            );
            assert!(
                bytes <= limits.max_buffer_size,
                "a tape of {} tasks binds {bytes} bytes of {name}, and the device holds buffers of at most {} bytes",
                encoding.task_count(),
                limits.max_buffer_size,
            );
        }
        let device = context.device();
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
        let mut clearing = Submission::new(device, "neura tensors");
        clearing.clear_buffer(
            heap.buffer().buffer(),
            tensors.word * WORD_BYTES,
            Some(tensors.words * WORD_BYTES),
        );
        clearing.submit(queue);
        tape.write(queue, encoding.tasks());
        values.write(queue, encoding.values());
        bounds.write(queue, encoding.bounds());
        if !encoding.steps().is_empty() {
            steps.write(queue, encoding.steps());
        }
        let geometry = Geometry::of(encoding.profile(), encoding.tiles());
        let placement = Placement::new(heap.words(), weights.offset() / WORD_BYTES, tensors.word);
        let kernel = context
            .declare(Megakernel::assemble(geometry, weights.precision(), placement).program());
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
                binding: HEAP,
                resource: heap.buffer().resource(0, heap.buffer().size()),
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
            heap,
            tensors,
            weights,
            tape,
            values,
            bounds,
            steps,
            cursor,
            group,
        }
    }

    pub fn heap(&self) -> &GpuBuffer {
        self.heap.buffer()
    }

    pub fn heap_bytes(&self) -> u64 {
        self.heap.bytes()
    }

    pub fn tensors(&self) -> &GpuBuffer {
        self.heap.buffer()
    }

    pub fn tensor_bytes(&self) -> u64 {
        self.encoding.tensor_bytes()
    }

    pub fn arena_bytes(&self) -> u64 {
        self.encoding.arena_bytes()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.encoding.resident_bytes()
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    pub fn device_bytes(&self) -> u64 {
        self.heap.bytes()
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

    pub fn updates_weights(&self) -> bool {
        self.encoding.updates_weights()
    }

    pub fn span(&self, value: Value) -> Span {
        let span = self.encoding.span(value);
        Span {
            store: span.store,
            offset: span.offset,
            elements: span.elements,
        }
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        self.heap.release(self.tensors);
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub store: Store,
    pub offset: u64,
    pub elements: u32,
}
