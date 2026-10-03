use crate::cache::Resident;
use crate::heap::Allocation;
use crate::pool::Recycled;
use neura_abi::{Placement, PlacementFields, PlacementRecord, REFUSAL_BYTES, WORD_BYTES, progress};
use neura_gpu::Queue;
use neura_gpu::{BindGroup, Binding, BufferUsages, GpuBuffer, GpuContext, Submission};
use neura_graph::{GraphStamp, Revision, Value};
use neura_kernel::{HEAP, PLACEMENT, PROGRESS, REFUSAL, SEGMENTS, STEPS, TASKS, VALUES};
use neura_plan::{Plan, Region, Span};
use neura_profile::{MatmulTile, Profile};
use std::cell::{Ref, RefCell};
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
    tasks: Recycled,
    values: Recycled,
    bound: RefCell<Bound>,
    plan: Arc<Plan>,
}

struct Bound {
    extents: Option<Vec<u32>>,
    written: bool,
}

impl Bound {
    fn of(dynamic: bool) -> Self {
        let fixed = !dynamic;
        Self {
            extents: (!dynamic).then(Vec::new),
            written: fixed,
        }
    }

    pub(crate) fn extents(&self) -> &[u32] {
        self.extents.as_deref().unwrap_or_else(|| {
            panic!(
                "a program of free extents runs the binding a run names, and no run has named one yet",
            )
        })
    }

    pub(crate) fn bind(&mut self, extents: Vec<u32>, dynamic: bool) {
        self.written = !dynamic;
        self.extents = Some(extents);
    }
}

impl<'r> Program<'r> {
    pub(crate) fn of(
        context: &GpuContext,
        resident: Arc<Resident>,
        plan: Arc<Plan>,
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
        let waves = plan.wave_count();
        let segments = plan.segments().len() as u32;
        let header = progress::header(segments, waves);
        let progress_buffer = Recycled::claim(
            pool,
            "neura progress",
            progress::bytes(waves),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        progress_buffer.buffer().write(
            queue,
            bytemuck::cast_slice(&progress::words(segments, plan.wave_tasks())),
        );
        let mut clearing = Submission::new(context.device(), "neura tensors");
        clearing.clear(tensors.buffer(), tensors.offset(), tensors.bytes());
        clearing.submit(queue);
        for quantum in plan.quanta() {
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
        let tasks = Recycled::claim(
            pool,
            "neura tasks",
            plan.tasks().len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let values = Recycled::claim(
            pool,
            "neura values",
            plan.values().len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        tasks.buffer().write(queue, plan.tasks());
        values.buffer().write(queue, plan.values());
        let group = resident.kernel.bind_group(&[
            Binding {
                index: TASKS,
                buffer: tasks.buffer().binding(0, tasks.buffer().size()),
            },
            Binding {
                index: VALUES,
                buffer: values.buffer().binding(0, values.buffer().size()),
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
        let dynamic = plan.dynamic();
        let workgroups = segments.min(plan.profile().workgroups()).max(1);
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
            tasks,
            values,
            bound: RefCell::new(Bound::of(dynamic)),
            plan,
        }
    }

    pub(crate) fn bind(&self, extents: &[u32]) {
        let bounds = self.plan.slot_bounds();
        assert_eq!(
            extents.len(),
            bounds.len(),
            "a program of {} free extents runs a binding of {} lengths",
            bounds.len(),
            extents.len(),
        );
        for (slot, (extent, bound)) in extents.iter().zip(bounds).enumerate() {
            assert!(
                extent <= bound,
                "free extent {slot} of {extent} outruns the bound of {bound} the graph declares",
            );
        }
        self.bound
            .borrow_mut()
            .bind(extents.to_vec(), self.dynamic());
    }

    pub(crate) fn extents(&self) -> Ref<'_, [u32]> {
        Ref::map(self.bound.borrow(), |bound| bound.extents())
    }

    pub(crate) fn write_records(&self, queue: &Queue, extents: &[u32]) {
        let encoding = self.plan.encode(extents);
        self.values.buffer().write(queue, &encoding.values);
        self.tasks.buffer().write(queue, &encoding.tasks);
    }

    pub(crate) fn records_pending(&self) -> bool {
        let bound = self.bound.borrow();
        bound.extents.is_some() && !bound.written
    }

    pub(crate) fn records_written(&self) {
        self.bound.borrow_mut().written = true;
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
        self.plan.tensor_bytes()
    }

    pub fn arena_bytes(&self) -> u64 {
        self.plan.arena_bytes()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.plan.resident_bytes()
    }

    pub fn weights(&self) -> &Weights<'r> {
        &self.weights
    }

    pub fn device_bytes(&self) -> u64 {
        self.tensors.heap().bytes()
            + self.tasks.buffer().size()
            + self.values.buffer().size()
            + self.resident.steps.buffer().size()
            + self.resident.segments.buffer().size()
            + self.refusal.buffer().size()
            + self.progress.buffer().size()
            + self.placement.buffer().size()
    }

    pub fn profile(&self) -> Profile {
        self.plan.profile()
    }

    pub fn dynamic(&self) -> bool {
        self.plan.dynamic()
    }

    pub fn is_compiled(&self) -> bool {
        self.resident.kernel.is_compiled()
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        self.plan.tiles()
    }

    pub fn matmul_geometries(&self) -> Vec<(MatmulTile, u32)> {
        self.plan.matmul_geometries()
    }

    pub fn task_count(&self) -> u32 {
        self.plan.task_count()
    }

    pub fn step_count(&self) -> u32 {
        self.plan.step_count()
    }

    pub fn wave_count(&self) -> u32 {
        self.plan.wave_count()
    }

    pub fn workgroups(&self) -> u32 {
        self.workgroups
    }

    pub fn value_count(&self) -> u32 {
        self.plan.value_count()
    }

    pub fn work(&self) -> u64 {
        self.plan.work()
    }

    pub fn readable(&self, value: Value) -> bool {
        self.plan.readable(value)
    }

    pub fn updates_weights(&self) -> bool {
        self.plan.updates_weights()
    }

    pub fn span(&self, value: Value) -> Span {
        let extents = self.extents();
        self.plan.span_at(value, self.at(), extents.as_ref())
    }
}
