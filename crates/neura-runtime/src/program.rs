use crate::heap::Allocation;
use crate::pool::Recycled;
use crate::tape::DeviceTape;
use neura_abi::{Placement, PlacementFields, PlacementRecord, REFUSAL_BYTES, WORD_BYTES, progress};
use neura_gpu::{BindGroup, Binding, BufferUsages, GpuBuffer, GpuContext, Submission};
use neura_graph::{GraphStamp, Revision, Value};
use neura_megakernel::{HEAP, PLACEMENT, PROGRESS, REFUSAL, SEGMENTS, STEPS, TASKS, VALUES};
use neura_profile::{MatmulTile, Profile};
use neura_tape::{Region, Span};
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
    pub(crate) tape: Arc<DeviceTape>,
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
        tape: Arc<DeviceTape>,
        tensors: Allocation,
        weights: Weights<'r>,
        revision: Revision,
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
        let waves = tape.image.wave_count();
        let segments = tape.image.segments().len() as u32;
        let header = progress::header(segments, waves);
        let progress_buffer = Recycled::claim(
            pool,
            "neura progress",
            progress::bytes(waves),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        progress_buffer.buffer().write(
            queue,
            bytemuck::cast_slice(&progress::words(segments, tape.image.wave_tasks())),
        );
        let mut clearing = Submission::new(context.device(), "neura tensors");
        clearing.clear(tensors.buffer(), tensors.offset(), tensors.bytes());
        clearing.submit(queue);
        for quantum in tape.image.quanta() {
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
        let group = tape.kernel.bind_group(&[
            Binding {
                index: TASKS,
                buffer: tape.tasks.buffer().binding(0, tape.tasks.buffer().size()),
            },
            Binding {
                index: VALUES,
                buffer: tape.values.buffer().binding(0, tape.values.buffer().size()),
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
                buffer: tape.steps.buffer().binding(0, tape.steps.buffer().size()),
            },
            Binding {
                index: PLACEMENT,
                buffer: placement.buffer().binding(0, placement.buffer().size()),
            },
            Binding {
                index: SEGMENTS,
                buffer: tape
                    .segments
                    .buffer()
                    .binding(0, tape.segments.buffer().size()),
            },
        ]);
        let workgroups = segments.min(tape.image.profile().workgroups()).max(1);
        Self {
            brand: PhantomData,
            tape,
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
        self.tape.image.tensor_bytes()
    }

    pub fn arena_bytes(&self) -> u64 {
        self.tape.image.arena_bytes()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.tape.image.resident_bytes()
    }

    pub fn weights(&self) -> &Weights<'r> {
        &self.weights
    }

    pub fn device_bytes(&self) -> u64 {
        self.tensors.heap().bytes()
            + self.tape.tasks.buffer().size()
            + self.tape.values.buffer().size()
            + self.tape.steps.buffer().size()
            + self.tape.segments.buffer().size()
            + self.refusal.buffer().size()
            + self.progress.buffer().size()
            + self.placement.buffer().size()
    }

    pub fn profile(&self) -> Profile {
        self.tape.image.profile()
    }

    pub fn is_compiled(&self) -> bool {
        self.tape.kernel.is_compiled()
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        self.tape.image.tiles()
    }

    pub fn matmul_geometries(&self) -> Vec<(MatmulTile, u32)> {
        self.tape.image.matmul_geometries()
    }

    pub fn task_count(&self) -> u32 {
        self.tape.image.task_count()
    }

    pub fn step_count(&self) -> u32 {
        self.tape.image.step_count()
    }

    pub fn wave_count(&self) -> u32 {
        self.tape.image.wave_count()
    }

    pub fn workgroups(&self) -> u32 {
        self.workgroups
    }

    pub fn value_count(&self) -> u32 {
        self.tape.image.value_count()
    }

    pub fn work(&self) -> u64 {
        self.tape.image.work()
    }

    pub fn readable(&self, value: Value) -> bool {
        self.tape.image.readable(value)
    }

    pub fn updates_weights(&self) -> bool {
        self.tape.image.updates_weights()
    }

    pub fn span(&self, value: Value) -> Span {
        self.tape.image.span(value, self.at())
    }
}
