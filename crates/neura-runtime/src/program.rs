use crate::heap::Allocation;
use crate::pool::Recycled;
use crate::tape::DeviceTape;
use neura_abi::{
    BoundsRecord, MatmulTile, Placement, PlacementRecord, Precision, Profile, REFUSAL_BYTES,
    WORD_BYTES,
};
use neura_gpu::{BindGroup, BindGroupEntry, BufferUsages, GpuBuffer, GpuContext, Submission};
use neura_program::{Region, Span, Value};
use neura_shader::{BOUNDS, HEAP, PLACEMENT, REFUSAL, SEGMENTS, STEPS, TASKS, VALUES};
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

#[derive(Clone)]
pub struct Weights<'r> {
    store: Allocation,
    region: Region,
    precision: Precision,
    brand: PhantomData<&'r ()>,
}

impl<'r> Weights<'r> {
    pub(crate) fn new(store: Allocation, region: Region, precision: Precision) -> Self {
        Self {
            store,
            region,
            precision,
            brand: PhantomData,
        }
    }

    pub fn buffer(&self) -> &GpuBuffer {
        self.store.buffer()
    }

    pub fn offset(&self) -> u64 {
        self.store.offset()
    }

    pub fn bytes(&self) -> u64 {
        self.store.bytes()
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

    pub(crate) fn allocation(&self) -> &Allocation {
        &self.store
    }

    pub(crate) fn lives_on(&self, heap: &Arc<crate::heap::Heap>) -> bool {
        self.store.lives_on(heap)
    }
}

pub struct Program<'r> {
    brand: PhantomData<&'r ()>,
    pub(crate) tape: Arc<DeviceTape>,
    pub(crate) refusal: Recycled,
    pub(crate) group: BindGroup,
    pub(crate) tensors: Allocation,
    pub(crate) weights: Weights<'r>,
    placement: Recycled,
}

impl<'r> Program<'r> {
    pub(crate) fn of(
        context: &GpuContext,
        tape: Arc<DeviceTape>,
        tensors: Allocation,
        weights: Weights<'r>,
    ) -> Self {
        let pool = tape.pool();
        let refusal = Recycled::claim(
            pool,
            "neura refusal",
            REFUSAL_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let placement = Recycled::claim(
            pool,
            "neura placement",
            size_of::<PlacementRecord>() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let queue = context.queue();
        let mut clearing = Submission::new(context.device(), "neura tensors");
        clearing.clear_buffer(
            tensors.buffer().buffer(),
            tensors.offset(),
            Some(tensors.bytes()),
        );
        clearing.submit(queue);
        placement.buffer().write(
            queue,
            bytemuck::bytes_of(&PlacementRecord {
                tensors: u32::try_from(tensors.word()).unwrap_or_else(|_| {
                    panic!("the tensors of a program start beyond the device address space")
                }),
                weights: u32::try_from(weights.offset() / WORD_BYTES).unwrap_or_else(|_| {
                    panic!("the weights of a program start beyond the device address space")
                }),
            }),
        );
        let group = tape.kernel.bind_group(&[
            BindGroupEntry {
                binding: TASKS,
                resource: tape.tasks.buffer().resource(0, tape.tasks.buffer().size()),
            },
            BindGroupEntry {
                binding: VALUES,
                resource: tape
                    .values
                    .buffer()
                    .resource(0, tape.values.buffer().size()),
            },
            BindGroupEntry {
                binding: HEAP,
                resource: tensors.buffer().resource(0, tensors.buffer().size()),
            },
            BindGroupEntry {
                binding: REFUSAL,
                resource: refusal.buffer().resource(0, refusal.buffer().size()),
            },
            BindGroupEntry {
                binding: BOUNDS,
                resource: tape
                    .bounds
                    .buffer()
                    .resource(0, size_of::<BoundsRecord>() as u64),
            },
            BindGroupEntry {
                binding: STEPS,
                resource: tape.steps.buffer().resource(0, tape.steps.buffer().size()),
            },
            BindGroupEntry {
                binding: PLACEMENT,
                resource: placement.buffer().resource(0, placement.buffer().size()),
            },
            BindGroupEntry {
                binding: SEGMENTS,
                resource: tape
                    .segments
                    .buffer()
                    .resource(0, tape.segments.buffer().size()),
            },
        ]);
        Self {
            brand: PhantomData,
            tape,
            refusal,
            group,
            tensors,
            weights,
            placement,
        }
    }

    fn at(&self) -> Placement {
        Placement::new(self.tensors.word(), self.weights.offset() / WORD_BYTES)
    }

    pub(crate) fn lives_on(&self, heap: &Arc<crate::heap::Heap>) -> bool {
        self.tensors.lives_on(heap)
    }

    pub fn heap(&self) -> &GpuBuffer {
        self.tensors.buffer()
    }

    pub fn heap_bytes(&self) -> u64 {
        self.tensors.heap().bytes()
    }

    pub fn tensor_bytes(&self) -> u64 {
        self.tape.encoding.tensor_bytes()
    }

    pub fn arena_bytes(&self) -> u64 {
        self.tape.encoding.arena_bytes()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.tape.encoding.resident_bytes()
    }

    pub fn weights(&self) -> &Weights<'r> {
        &self.weights
    }

    pub fn device_bytes(&self) -> u64 {
        self.tensors.heap().bytes()
            + self.tape.tasks.buffer().size()
            + self.tape.values.buffer().size()
            + self.tape.bounds.buffer().size()
            + self.tape.steps.buffer().size()
            + self.tape.segments.buffer().size()
            + self.refusal.buffer().size()
            + self.placement.buffer().size()
    }

    pub fn profile(&self) -> Profile {
        self.tape.encoding.profile()
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        self.tape.encoding.tiles()
    }

    pub fn matmul_geometries(&self) -> Vec<(MatmulTile, u32)> {
        self.tape.encoding.matmul_geometries()
    }

    pub fn task_count(&self) -> u32 {
        self.tape.encoding.task_count()
    }

    pub fn step_count(&self) -> u32 {
        self.tape.encoding.step_count()
    }

    pub fn dispatch_count(&self) -> u32 {
        self.tape.encoding.dispatch_count()
    }

    pub fn value_count(&self) -> u32 {
        self.tape.encoding.value_count()
    }

    pub fn work(&self) -> u64 {
        self.tape.encoding.work()
    }

    pub fn readable(&self, value: Value) -> bool {
        self.tape.encoding.readable(value)
    }

    pub fn updates_weights(&self) -> bool {
        self.tape.encoding.updates_weights()
    }

    pub fn span(&self, value: Value) -> Span {
        self.tape.encoding.span(value, self.at())
    }
}
