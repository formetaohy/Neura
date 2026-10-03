use crate::cache::Resident;
use crate::heap::Allocation;
use crate::pool::Recycled;
use neura_abi::{Placement, PlacementFields, PlacementRecord, REFUSAL_BYTES, WORD_BYTES, progress};
use neura_gpu::{BindGroup, Binding, BufferUsages, GpuBuffer, GpuContext, Submission};
use neura_graph::{GraphStamp, Revision, Value};
use neura_kernel::{HEAP, PLACEMENT, PROGRESS, REFUSAL, SEGMENTS, STEPS, TASKS, VALUES};
use neura_plan::{Region, Span};
use neura_profile::{MatmulTile, Profile};
use std::marker::PhantomData;
use std::mem::size_of;
use std::sync::Arc;

#[derive(Clone)]
pub struct Weights<'r> {
    store: Allocation,
    weights: Region,
    state: Region,
    brand: PhantomData<&'r ()>,
}

impl<'r> Weights<'r> {
    pub(crate) fn new(store: Allocation, weights: Region, state: Region) -> Self {
        Self {
            store,
            weights,
            state,
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
        self.weights.tensors()
    }

    pub(crate) fn region(&self) -> &Region {
        &self.weights
    }

    pub(crate) fn state(&self) -> &Region {
        &self.state
    }

    pub(crate) fn words(&self) -> u64 {
        self.state.words().max(self.weights.words())
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
    pub(crate) resident: Arc<Resident>,
    pub(crate) refusal: Recycled,
    pub(crate) group: BindGroup,
    pub(crate) tensors: Allocation,
    pub(crate) weights: Weights<'r>,
    pub(crate) progress: Recycled,
    pub(crate) header: Vec<u8>,
    pub(crate) workgroups: u32,
    placement: Recycled,
    revision: Revision,
}

impl<'r> Program<'r> {
    pub(crate) fn of(
        context: &GpuContext,
        resident: Arc<Resident>,
        tensors: Allocation,
        weights: Weights<'r>,
        revision: Revision,
    ) -> Self {
        let pool = resident.pool();
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
        let waves = resident.plan.wave_count();
        let segments = resident.plan.segments().len() as u32;
        let header = progress::header(segments, waves);
        let progress_buffer = Recycled::claim(
            pool,
            "neura progress",
            progress::bytes(waves),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        progress_buffer.buffer().write(
            queue,
            bytemuck::cast_slice(&progress::words(segments, resident.plan.wave_tasks())),
        );
        let mut clearing = Submission::new(context.device(), "neura tensors");
        clearing.clear(tensors.buffer(), tensors.offset(), tensors.bytes());
        clearing.submit(queue);
        for quantum in resident.plan.quanta() {
            tensors.buffer().write_at(
                queue,
                tensors.offset() + quantum.offset,
                &quantum.scale.to_ne_bytes(),
            );
        }
        placement.buffer().write(
            queue,
            bytemuck::bytes_of(&PlacementRecord::of(PlacementFields {
                tensors: u32::try_from(tensors.word()).unwrap_or_else(|_| {
                    panic!("the tensors of a program start beyond the device address space")
                }),
                weights: u32::try_from(weights.offset() / WORD_BYTES).unwrap_or_else(|_| {
                    panic!("the weights of a program start beyond the device address space")
                }),
            })),
        );
        let group = resident.kernel.bind_group(&[
            Binding {
                index: TASKS,
                buffer: resident
                    .tasks
                    .buffer()
                    .binding(0, resident.tasks.buffer().size()),
            },
            Binding {
                index: VALUES,
                buffer: resident
                    .values
                    .buffer()
                    .binding(0, resident.values.buffer().size()),
            },
            Binding {
                index: HEAP,
                buffer: tensors.buffer().binding(0, tensors.buffer().size()),
            },
            Binding {
                index: REFUSAL,
                buffer: refusal.buffer().binding(0, refusal.buffer().size()),
            },
            Binding {
                index: PROGRESS,
                buffer: progress_buffer
                    .buffer()
                    .binding(0, progress_buffer.buffer().size()),
            },
            Binding {
                index: STEPS,
                buffer: resident
                    .steps
                    .buffer()
                    .binding(0, resident.steps.buffer().size()),
            },
            Binding {
                index: PLACEMENT,
                buffer: placement.buffer().binding(0, placement.buffer().size()),
            },
            Binding {
                index: SEGMENTS,
                buffer: resident
                    .segments
                    .buffer()
                    .binding(0, resident.segments.buffer().size()),
            },
        ]);
        let workgroups = segments.min(resident.plan.profile().workgroups()).max(1);
        Self {
            brand: PhantomData,
            resident,
            refusal,
            group,
            tensors,
            weights,
            progress: progress_buffer,
            header,
            workgroups,
            placement,
            revision,
        }
    }

    pub fn stamp(&self) -> GraphStamp {
        self.revision.stamp()
    }

    pub fn is_current(&self) -> bool {
        self.revision.is_current()
    }

    pub(crate) fn assert_current(&self) {
        assert!(
            self.is_current(),
            "a program serves the revision of the graph it was compiled from, and this graph has moved on since; compile again from the graph as it now stands",
        );
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
        self.resident.plan.tensor_bytes()
    }

    pub fn arena_bytes(&self) -> u64 {
        self.resident.plan.arena_bytes()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.resident.plan.resident_bytes()
    }

    pub fn weights(&self) -> &Weights<'r> {
        &self.weights
    }

    pub fn device_bytes(&self) -> u64 {
        self.tensors.heap().bytes()
            + self.resident.tasks.buffer().size()
            + self.resident.values.buffer().size()
            + self.resident.steps.buffer().size()
            + self.resident.segments.buffer().size()
            + self.refusal.buffer().size()
            + self.progress.buffer().size()
            + self.placement.buffer().size()
    }

    pub fn profile(&self) -> Profile {
        self.resident.plan.profile()
    }

    pub fn is_compiled(&self) -> bool {
        self.resident.kernel.is_compiled()
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        self.resident.plan.tiles()
    }

    pub fn matmul_geometries(&self) -> Vec<(MatmulTile, u32)> {
        self.resident.plan.matmul_geometries()
    }

    pub fn task_count(&self) -> u32 {
        self.resident.plan.task_count()
    }

    pub fn step_count(&self) -> u32 {
        self.resident.plan.step_count()
    }

    pub fn wave_count(&self) -> u32 {
        self.resident.plan.wave_count()
    }

    pub fn workgroups(&self) -> u32 {
        self.workgroups
    }

    pub fn value_count(&self) -> u32 {
        self.resident.plan.value_count()
    }

    pub fn work(&self) -> u64 {
        self.resident.plan.work()
    }

    pub fn readable(&self, value: Value) -> bool {
        self.resident.plan.readable(value)
    }

    pub fn updates_weights(&self) -> bool {
        self.resident.plan.updates_weights()
    }

    pub fn span(&self, value: Value) -> Span {
        self.resident.plan.span(value, self.at())
    }
}
