use crate::cache::Resident;
use crate::heap::Allocation;
use crate::pool::Recycled;
use crate::store::WeightStore;
use neura_abi::{Placement, PlacementFields, PlacementRecord, REFUSAL_BYTES, WORD_BYTES, progress};
use neura_gpu::Queue;
use neura_gpu::{BindGroup, Binding, BufferUsages, GpuBuffer, GpuContext, Submission};
use neura_graph::{GraphStamp, Revision, Value};
use neura_kernel::{HEAP, TASKS, VALUES};
use neura_plan::{Plan, Region, Span};
use neura_profile::{MatmulTile, Profile};
use std::mem::size_of;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

#[derive(Clone)]
pub struct Weights {
    store: Arc<WeightStore>,
    weights: Region,
    state: Region,
}

impl Weights {
    pub(crate) fn new(store: Arc<WeightStore>, weights: Region, state: Region) -> Self {
        Self {
            store,
            weights,
            state,
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

    pub fn pages(&self) -> u32 {
        self.store.pages()
    }

    pub fn resident_pages(&self) -> u32 {
        self.store.resident_pages()
    }

    pub fn resident_bytes(&self) -> u64 {
        self.store.resident_bytes()
    }

    pub(crate) fn region(&self) -> &Region {
        &self.weights
    }

    pub(crate) fn state(&self) -> &Region {
        &self.state
    }

    pub(crate) fn words(&self) -> u64 {
        self.store.words()
    }

    pub(crate) fn store(&self) -> &Arc<WeightStore> {
        &self.store
    }

    pub(crate) fn paged(&self) -> bool {
        self.store.paged()
    }

    pub(crate) fn lives_on(&self, heap: &Arc<crate::heap::Heap>) -> bool {
        self.store.lives_on(heap)
    }
}

pub struct Program {
    pub(crate) resident: Arc<Resident>,
    extents: Option<Recycled>,
    cached: Mutex<Option<Vec<u32>>>,
    pub(crate) refusal: Recycled,
    pub(crate) group: BindGroup,
    pub(crate) tensors: Allocation,
    pub(crate) weights: Weights,
    pub(crate) progress: Recycled,
    pub(crate) header: Vec<u8>,
    pub(crate) workgroups: u32,
    placement: Recycled,
    revision: Revision,
    tasks: Recycled,
    values: Recycled,
    bound: Mutex<Bound>,
    groups: Vec<WeightGroup>,
    plan: Arc<Plan>,
}

pub(crate) struct WeightGroup {
    first_segment: u32,
    segments: u32,
    first_wave: u32,
    first_task: u32,
    last_task: u32,
    pages: Vec<u32>,
    writes: Vec<u32>,
}

impl WeightGroup {
    fn of(plan: &Plan, first_task: u32, last_task: u32, pages: Vec<u32>, writes: Vec<u32>) -> Self {
        let segments = plan.segments();
        let first = segments
            .iter()
            .position(|segment| segment.first + segment.count > first_task)
            .unwrap_or_else(|| panic!("task {first_task} of a plan stands in no segment"));
        let last = segments
            .iter()
            .rposition(|segment| segment.first < last_task)
            .unwrap_or_else(|| panic!("task {} of a plan stands in no segment", last_task - 1));
        Self {
            first_segment: first as u32,
            segments: (last - first + 1) as u32,
            first_wave: segments[first].wave,
            first_task,
            last_task,
            pages,
            writes,
        }
    }

    pub(crate) fn first_segment(&self) -> u32 {
        self.first_segment
    }

    pub(crate) fn segments(&self) -> u32 {
        self.segments
    }

    pub(crate) fn first_wave(&self) -> u32 {
        self.first_wave
    }

    pub(crate) fn first_task(&self) -> u32 {
        self.first_task
    }

    pub(crate) fn last_task(&self) -> u32 {
        self.last_task
    }

    pub(crate) fn pages(&self) -> &[u32] {
        &self.pages
    }

    pub(crate) fn writes(&self) -> &[u32] {
        &self.writes
    }
}

fn weight_groups(plan: &Plan, slots: u32) -> Vec<WeightGroup> {
    let tasks = plan.weight_pages();
    let mut groups = Vec::new();
    let mut pages = Vec::<u32>::new();
    let mut writes = Vec::<u32>::new();
    let mut start = 0u32;
    for (position, task) in tasks.iter().enumerate() {
        let position = position as u32;
        let mut merged = pages.clone();
        merged.extend(task.pages());
        merged.sort_unstable();
        merged.dedup();
        if merged.len() > slots as usize {
            if pages.is_empty() {
                panic!(
                    "task {position} of this plan walks {} weight pages, and the resident weights of {slots} pages cannot hold them; raise the resident weight budget, or bind a graph whose tasks walk no more pages than it holds",
                    merged.len(),
                );
            }
            groups.push(WeightGroup::of(
                plan,
                start,
                position,
                std::mem::take(&mut pages),
                std::mem::take(&mut writes),
            ));
            start = position;
            merged = task.pages().to_vec();
            assert!(
                merged.len() <= slots as usize,
                "task {position} of this plan walks {} weight pages, and the resident weights of {slots} pages cannot hold them; raise the resident weight budget, or bind a graph whose tasks walk no more pages than it holds",
                merged.len(),
            );
        }
        pages = merged;
        writes.extend(task.writes());
        writes.sort_unstable();
        writes.dedup();
    }
    groups.push(WeightGroup::of(
        plan,
        start,
        tasks.len() as u32,
        pages,
        writes,
    ));
    groups
}

struct Bound {
    extents: Option<Vec<u32>>,
    written: bool,
}

impl Bound {
    fn of(plan: &Plan) -> Self {
        let ready = !plan.dynamic() || plan.host_slots().is_empty();
        Self {
            extents: ready.then(|| plan.host_extents()),
            written: ready,
        }
    }
}

impl Program {
    fn binding(&self) -> MutexGuard<'_, Bound> {
        self.bound.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn extents_cache(&self) -> MutexGuard<'_, Option<Vec<u32>>> {
        self.cached.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl Program {
    pub(crate) fn of(
        context: &GpuContext,
        resident: Arc<Resident>,
        plan: Arc<Plan>,
        tensors: Allocation,
        weights: Weights,
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
        let mut clearing = Submission::new(context.device(), "neura tensors and refusal");
        clearing.clear(tensors.buffer(), tensors.offset(), tensors.bytes());
        clearing.clear(refusal.buffer(), 0, refusal.buffer().size());
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
        let extents = plan.carries_authored().then(|| {
            Recycled::claim(
                pool,
                "neura extents",
                (plan.slot_bounds().len() as u64 * WORD_BYTES).max(WORD_BYTES),
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            )
        });
        if let Some(extents) = &extents {
            extents
                .buffer()
                .write(queue, bytemuck::cast_slice(&plan.host_extents()));
        }
        let workgroups = segments.min(plan.profile().workgroups()).max(1);
        let heap = tensors.heap();
        let banks = heap.banks();
        let bank_bytes = heap.bank_bytes();
        let paged = weights.paged();
        let groups = if paged {
            weight_groups(&plan, weights.resident_pages())
        } else {
            Vec::new()
        };
        let mut bindings = vec![
            Binding {
                index: TASKS,
                buffer: tasks.buffer().binding(0, tasks.buffer().size()),
            },
            Binding {
                index: VALUES,
                buffer: values.buffer().binding(0, values.buffer().size()),
            },
        ];
        for bank in 0..banks.count() {
            let offset = u64::from(bank) * bank_bytes;
            let size = (heap.bytes() - offset).min(bank_bytes);
            bindings.push(Binding {
                index: HEAP + bank,
                buffer: tensors.buffer().binding(offset, size),
            });
        }
        if let Some(table) = weights.store().table() {
            bindings.push(Binding {
                index: neura_kernel::pages(banks),
                buffer: table.binding(0, table.size()),
            });
        }
        bindings.extend([
            Binding {
                index: neura_kernel::refusal(banks, paged),
                buffer: refusal.buffer().binding(0, refusal.buffer().size()),
            },
            Binding {
                index: neura_kernel::progress(banks, paged),
                buffer: progress_buffer
                    .buffer()
                    .binding(0, progress_buffer.buffer().size()),
            },
            Binding {
                index: neura_kernel::steps(banks, paged),
                buffer: resident
                    .steps
                    .buffer()
                    .binding(0, resident.steps.buffer().size()),
            },
            Binding {
                index: neura_kernel::placement(banks, paged),
                buffer: placement.buffer().binding(0, placement.buffer().size()),
            },
            Binding {
                index: neura_kernel::segments(banks, paged),
                buffer: resident
                    .segments
                    .buffer()
                    .binding(0, resident.segments.buffer().size()),
            },
        ]);
        if let (Some(extents), Some(measures), Some(patches), Some(patch_list)) = (
            extents.as_ref(),
            resident.measures.as_ref(),
            resident.patches.as_ref(),
            resident.patch_list.as_ref(),
        ) {
            bindings.extend([
                Binding {
                    index: neura_kernel::extents(banks, paged),
                    buffer: extents.buffer().binding(0, extents.buffer().size()),
                },
                Binding {
                    index: neura_kernel::measures(banks, paged),
                    buffer: measures.buffer().binding(0, measures.buffer().size()),
                },
                Binding {
                    index: neura_kernel::patches(banks, paged),
                    buffer: patches.buffer().binding(0, patches.buffer().size()),
                },
                Binding {
                    index: neura_kernel::patch_list(banks, paged),
                    buffer: patch_list.buffer().binding(0, patch_list.buffer().size()),
                },
            ]);
        }
        let group = resident.kernel.bind_group(&bindings);
        Self {
            resident,
            extents,
            cached: Mutex::new(None),
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
            bound: Mutex::new(Bound::of(&plan)),
            groups,
            plan,
        }
    }

    pub(crate) fn bind(&self, extents: &[u32]) {
        let host = self.plan.host_slots();
        assert_eq!(
            extents.len(),
            host.len(),
            "a program of {} free extents the host binds runs a binding of {} lengths",
            host.len(),
            extents.len(),
        );
        let mut bound = self.binding();
        let mut values = bound
            .extents
            .clone()
            .unwrap_or_else(|| self.plan.host_extents());
        for (extent, (slot, bound)) in extents.iter().zip(&host) {
            assert!(
                extent <= bound,
                "free extent {slot} of {extent} outruns the bound of {bound} the graph declares",
            );
            values[*slot as usize] = *extent;
        }
        bound.extents = Some(values);
        bound.written = false;
        drop(bound);
        self.extents_cache().take();
    }

    pub fn carries_authored(&self) -> bool {
        self.plan.carries_authored()
    }

    pub(crate) fn slot_count(&self) -> usize {
        self.plan.slot_bounds().len()
    }

    pub(crate) fn extents_buffer(&self) -> &GpuBuffer {
        self.extents
            .as_ref()
            .expect("a program of a device authored extent holds the lengths it walks")
            .buffer()
    }

    pub(crate) fn cached_extents(&self) -> Option<Vec<u32>> {
        self.extents_cache().clone()
    }

    pub(crate) fn cache_extents(&self, extents: Vec<u32>) {
        *self.extents_cache() = Some(extents);
    }

    pub(crate) fn write_extents(&self, queue: &Queue) {
        if self.extents.is_none() {
            return;
        }
        let extents = self.host_extents();
        self.extents_buffer()
            .write(queue, bytemuck::cast_slice(&extents));
        self.extents_cache().take();
    }

    pub(crate) fn host_extents(&self) -> Vec<u32> {
        let bound = self.binding();
        bound.extents.clone().unwrap_or_else(|| {
            panic!(
                "a program of free extents runs the binding a run names, and no run has named one yet",
            )
        })
    }

    pub(crate) fn span_with(&self, value: Value, extents: &[u32]) -> Span {
        self.plan.span_at(value, self.at(), extents)
    }

    pub(crate) fn weight_groups(&self) -> &[WeightGroup] {
        &self.groups
    }

    pub(crate) fn store(&self) -> &Arc<WeightStore> {
        self.weights.store()
    }

    pub(crate) fn write_records(&self, queue: &Queue, extents: &[u32]) {
        let encoding = self.plan.encode(extents);
        self.values.buffer().write(queue, &encoding.values);
        self.tasks.buffer().write(queue, &encoding.tasks);
    }

    pub(crate) fn records_pending(&self) -> bool {
        let bound = self.binding();
        bound.extents.is_some() && !bound.written
    }

    pub(crate) fn records_written(&self) {
        self.binding().written = true;
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

    pub fn weights(&self) -> &Weights {
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
        let host = self.host_extents();
        self.span_with(value, &host)
    }
}
