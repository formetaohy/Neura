use crate::cache::Resident;
use crate::heap::Allocation;
use crate::pool::Recycled;
use crate::store::WeightStore;
use neura_abi::{
    Placement, PlacementFields, PlacementRecord, REFUSAL_BYTES, SegmentRecord, WORD_BYTES, progress,
};
use neura_gpu::Queue;
use neura_gpu::{
    BindGroup, Binding, BufferUsages, GpuBuffer, GpuContext, READBACK_TIMEOUT, Submission,
    SubmissionIndex,
};
use neura_graph::{GraphStamp, Revision, Value};
use neura_kernel::{HEAP, TASKS, VALUES};
use neura_plan::{Encoding, Plan, Region, Span, TableRows, WeightPages};
use neura_profile::{MatmulTile, Profile};
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

const REMEMBERED_WINDOWS: usize = 8;

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

    pub fn readback_pages(&self) -> u64 {
        self.store.readback_pages()
    }

    pub fn readback_transfers(&self) -> u64 {
        self.store.readback_transfers()
    }

    pub fn host_bytes(&self) -> u64 {
        self.store.host_bytes()
    }

    pub fn spill_file(&self) -> Option<std::path::PathBuf> {
        self.store.spill_file()
    }

    pub fn spill_read_bytes(&self) -> u64 {
        self.store.spill_read_bytes()
    }

    pub fn spill_write_bytes(&self) -> u64 {
        self.store.spill_write_bytes()
    }

    pub(crate) fn region(&self) -> &Region {
        &self.weights
    }

    pub(crate) fn state(&self) -> &Region {
        &self.state
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
    pub(crate) weights: Weights,
    pub(crate) progress: Recycled,
    placement: Recycled,
    segments: Recycled,
    patches: Option<Recycled>,
    patch_list: Option<Recycled>,
    revision: Revision,
    tasks: Recycled,
    values: Recycled,
    tensors: Mutex<Tensors>,
    bound: Mutex<Bound>,
    windows: Mutex<Vec<Windows>>,
    planned_windows: AtomicUsize,
    encoding: Mutex<Option<Arc<Encoding>>>,
    last: Mutex<Option<SubmissionIndex>>,
    plan: Arc<Plan>,
}

struct Tensors {
    active: Option<Allocation>,
    retired: Vec<Allocation>,
}

impl Tensors {
    fn held(&self) -> &Allocation {
        self.active
            .as_ref()
            .expect("a program always walks the tensors it holds")
    }

    fn take(&mut self) -> Allocation {
        self.active
            .take()
            .expect("a program always walks the tensors it holds")
    }

    fn sweep(&mut self, queue: &Queue, last: Option<SubmissionIndex>) {
        match last {
            Some(index) if !queue.complete(index) => {}
            _ => self.retired.clear(),
        }
    }
}

struct Windows {
    extents: Vec<u32>,
    rows: Vec<(u32, Vec<u32>)>,
    groups: Arc<Vec<WeightGroup>>,
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
    fn of(
        encoding: &Encoding,
        first_task: u32,
        last_task: u32,
        pages: Vec<u32>,
        writes: Vec<u32>,
    ) -> Self {
        let segments = encoding.segments();
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

fn raise_or_declare(plan: &Plan) -> &'static str {
    match plan.gather_tables().is_empty() {
        true => "raise the resident weight budget",
        false => {
            "raise the resident weight budget, or name the rows a table walk reads with Runtime::declare_rows"
        }
    }
}

fn weight_groups(
    plan: &Plan,
    encoding: &Encoding,
    slots: u32,
    tasks: &[WeightPages],
) -> Vec<WeightGroup> {
    let mut groups = Vec::new();
    let mut pages = Vec::<u32>::new();
    let mut writes = Vec::<u32>::new();
    let mut start = 0u32;
    for (position, task) in tasks.iter().enumerate() {
        let position = position as u32;
        let added = task
            .pages()
            .iter()
            .filter(|page| pages.binary_search(page).is_err())
            .count();
        if pages.len() + added > slots as usize {
            if pages.is_empty() {
                panic!(
                    "task {position} of this plan walks {added} weight pages, and the resident weights of {slots} pages cannot hold them; {}, or bind a graph whose tasks walk no more pages than it holds",
                    raise_or_declare(plan),
                );
            }
            groups.push(WeightGroup::of(
                encoding,
                start,
                position,
                std::mem::take(&mut pages),
                std::mem::take(&mut writes),
            ));
            start = position;
        }
        for page in task.pages() {
            match pages.binary_search(page) {
                Ok(_) => {}
                Err(at) => pages.insert(at, *page),
            }
        }
        assert!(
            pages.len() <= slots as usize,
            "task {position} of this plan walks {} weight pages, and the resident weights of {slots} pages cannot hold them; {}, or bind a graph whose tasks walk no more pages than it holds",
            pages.len(),
            raise_or_declare(plan),
        );
        for write in task.writes() {
            match writes.binary_search(write) {
                Ok(_) => {}
                Err(at) => writes.insert(at, *write),
            }
        }
    }
    groups.push(WeightGroup::of(
        encoding,
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
    rows: Vec<(u32, Vec<u32>)>,
}

impl Bound {
    fn of(plan: &Plan) -> Self {
        let ready = !plan.dynamic() || plan.host_slots().is_empty();
        Self {
            extents: ready.then(|| plan.host_extents()),
            written: ready,
            rows: Vec::new(),
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
        let progress_buffer = Recycled::claim(
            pool,
            "neura progress",
            progress::bytes(plan.task_count()),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let segments = Recycled::claim(
            pool,
            "neura segments",
            (plan.task_count() as u64 * size_of::<SegmentRecord>() as u64).max(4),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let patches = plan.carries_authored().then(|| {
            Recycled::claim(
                pool,
                "neura patches",
                (plan.patches().len() as u64 * size_of::<neura_abi::PatchRecord>() as u64).max(4),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )
        });
        let patch_list = plan.carries_authored().then(|| {
            Recycled::claim(
                pool,
                "neura patch list",
                (plan.patch_list().len() as u64 * WORD_BYTES).max(4),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            )
        });
        let tensors = Tensors {
            active: Some(tensors),
            retired: Vec::new(),
        };
        let arena = tensors.held().clone();
        let mut clearing = Submission::new(context.device(), "neura tensors and refusal");
        if arena.bytes() > 0 {
            clearing.clear(arena.buffer(), arena.offset(), arena.bytes());
        }
        clearing.clear(refusal.buffer(), 0, refusal.buffer().size());
        clearing.submit(queue);
        if !plan.awaits_a_binding() {
            for quantum in plan.quanta() {
                arena.buffer().write_at(
                    queue,
                    arena.offset() + quantum.offset,
                    &quantum.scale.to_ne_bytes(),
                );
            }
            placement.buffer().write(
                queue,
                bytemuck::bytes_of(&PlacementRecord::of(PlacementFields {
                    tensors: u32::try_from(arena.word()).unwrap_or_else(|_| {
                        panic!("the tensors of a program start beyond the device address space")
                    }),
                    weights: u32::try_from(weights.offset() / WORD_BYTES).unwrap_or_else(|_| {
                        panic!("the weights of a program start beyond the device address space")
                    }),
                })),
            );
        }
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
        let bound_encoding = plan.bound_encoding();
        tasks.buffer().write(queue, bound_encoding.tasks());
        values.buffer().write(queue, bound_encoding.values());
        segments
            .buffer()
            .write(queue, bytemuck::cast_slice(bound_encoding.segments()));
        progress_buffer.buffer().write(
            queue,
            bytemuck::cast_slice(&progress::words(
                bound_encoding.segments().len() as u32,
                bound_encoding.wave_tasks(),
            )),
        );
        if let (Some(patches), Some(patch_list)) = (patches.as_ref(), patch_list.as_ref()) {
            patches
                .buffer()
                .write(queue, bytemuck::cast_slice(bound_encoding.patches()));
            patch_list
                .buffer()
                .write(queue, bytemuck::cast_slice(bound_encoding.patch_list()));
        }
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
        let heap = arena.heap();
        let banks = heap.banks();
        let bank_bytes = heap.bank_bytes();
        let paged = weights.paged();
        let bound = plan.host_extents();
        let preview = match paged && !plan.dynamic() && plan.gather_tables().is_empty() {
            true => {
                let groups = weight_groups(
                    &plan,
                    &bound_encoding,
                    weights.resident_pages(),
                    &plan.weight_pages(&bound_encoding, &[]),
                );
                Some(Windows {
                    extents: bound,
                    rows: Vec::new(),
                    groups: Arc::new(groups),
                })
            }
            false => None,
        };
        let previewed = usize::from(preview.is_some());
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
                buffer: arena.buffer().binding(offset, size),
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
                buffer: segments.buffer().binding(0, segments.buffer().size()),
            },
        ]);
        if let (Some(extents), Some(measures), Some(patches), Some(patch_list)) = (
            extents.as_ref(),
            resident.measures.as_ref(),
            patches.as_ref(),
            patch_list.as_ref(),
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
        let bound = Bound::of(&plan);
        let encoding = (!plan.awaits_a_binding()).then(|| plan.bound_encoding());
        Self {
            resident,
            extents,
            cached: Mutex::new(None),
            refusal,
            group,
            tensors: Mutex::new(tensors),
            weights,
            progress: progress_buffer,
            placement,
            segments,
            patches,
            patch_list,
            revision,
            tasks,
            values,
            bound: Mutex::new(bound),
            windows: Mutex::new(preview.into_iter().collect()),
            planned_windows: AtomicUsize::new(previewed),
            encoding: Mutex::new(encoding),
            last: Mutex::new(None),
            plan,
        }
    }

    pub(crate) fn bind(&self, extents: &[u32]) -> Vec<u32> {
        let host = self.plan.host_slots();
        assert_eq!(
            extents.len(),
            host.len(),
            "a program of {} free extents the host binds runs a binding of {} lengths",
            host.len(),
            extents.len(),
        );
        let bound = self.binding();
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
        values
    }

    pub(crate) fn declare_rows(&self, table: Value<'_>, rows: &[u32]) {
        assert!(
            self.plan.gather_tables().contains(&table.id()),
            "value {} is no table this program gathers rows of, and a host declares the rows of a table a task of the plan walks",
            table.id(),
        );
        assert!(
            !rows.is_empty(),
            "a host declares the rows a table walk reads, and the table of value {} declares none",
            table.id(),
        );
        let held = self.plan.table_rows(table.id());
        let mut declared = rows.to_vec();
        declared.sort_unstable();
        declared.dedup();
        assert!(
            declared[declared.len() - 1] < held,
            "a host declares row {} of the {held} rows value {} holds",
            declared[declared.len() - 1],
            table.id(),
        );
        let mut bound = self.binding();
        match bound.rows.iter_mut().find(|(kept, _)| *kept == table.id()) {
            Some((_, kept)) => *kept = declared,
            None => bound.rows.push((table.id(), declared)),
        }
        drop(bound);
    }

    fn declared_rows(&self) -> Vec<(u32, Vec<u32>)> {
        self.binding().rows.clone()
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
        let encoding = self.encoding();
        self.plan.span_at(&encoding, value, self.at(), extents)
    }

    pub(crate) fn encoding(&self) -> Arc<Encoding> {
        let lengths = self.host_extents();
        if let Some(encoding) = self
            .encoding
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .filter(|held| held.lengths() == lengths)
        {
            return encoding.clone();
        }
        let encoding = self.plan.encode(&lengths);
        *self.encoding.lock().unwrap_or_else(PoisonError::into_inner) = Some(encoding.clone());
        encoding
    }

    pub(crate) fn materialize(&self, queue: &Queue, lengths: &[u32]) {
        {
            let encoding = self.encoding.lock().unwrap_or_else(PoisonError::into_inner);
            if encoding
                .as_ref()
                .is_some_and(|held| held.lengths() == lengths)
            {
                return;
            }
        }
        let encoding = self.plan.encode(lengths);
        self.reserve(queue, encoding.tensor_bytes());
        self.publish(queue, &encoding);
        *self.encoding.lock().unwrap_or_else(PoisonError::into_inner) = Some(encoding);
        let mut bound = self.binding();
        bound.extents = Some(lengths.to_vec());
        bound.written = false;
        bound.rows.clear();
        drop(bound);
        self.extents_cache().take();
    }

    fn reserve(&self, queue: &Queue, tensor_bytes: u64) {
        let mut tensors = self.tensors.lock().unwrap_or_else(PoisonError::into_inner);
        let words = tensor_bytes / WORD_BYTES;
        if tensors.held().words() >= words {
            return;
        }
        let last = *self.last.lock().unwrap_or_else(PoisonError::into_inner);
        tensors.sweep(queue, last);
        let heap = tensors.held().share();
        let doubled = words.max(tensors.held().words() * 2);
        let wanted = if heap.holds(doubled) { doubled } else { words };
        let allocation = if heap.holds(wanted) {
            heap.allocate(wanted)
        } else {
            let in_flight = last.filter(|index| !queue.complete(*index));
            if let Some(index) = in_flight {
                queue.wait(index, READBACK_TIMEOUT);
            }
            tensors.sweep(queue, None);
            drop(tensors.take());
            heap.allocate(wanted)
        };
        let mut clearing = Submission::new(queue.device(), "neura tensors");
        clearing.clear(allocation.buffer(), allocation.offset(), allocation.bytes());
        clearing.submit(queue);
        let old = tensors.active.replace(allocation);
        if let Some(old) = old {
            tensors.retired.push(old);
        }
    }

    fn publish(&self, queue: &Queue, encoding: &Encoding) {
        let tensors = self.tensors.lock().unwrap_or_else(PoisonError::into_inner);
        for quantum in encoding.quanta() {
            tensors.held().buffer().write_at(
                queue,
                tensors.held().offset() + quantum.offset,
                &quantum.scale.to_ne_bytes(),
            );
        }
        self.placement.buffer().write(
            queue,
            bytemuck::bytes_of(&PlacementRecord::of(PlacementFields {
                tensors: u32::try_from(tensors.held().word()).unwrap_or_else(|_| {
                    panic!("the tensors of a program start beyond the device address space")
                }),
                weights: u32::try_from(self.weights.offset() / WORD_BYTES).unwrap_or_else(|_| {
                    panic!("the weights of a program start beyond the device address space")
                }),
            })),
        );
    }

    pub(crate) fn used(&self, submission: SubmissionIndex) {
        *self.last.lock().unwrap_or_else(PoisonError::into_inner) = Some(submission);
    }

    pub(crate) fn windows(&self) -> Arc<Vec<WeightGroup>> {
        let extents = self.host_extents();
        let declared = self.declared_rows();
        let mut windows = self.windows.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(position) = windows
            .iter()
            .position(|kept| kept.extents == extents && kept.rows == declared)
        {
            let kept = windows.remove(position);
            let groups = kept.groups.clone();
            windows.push(kept);
            return groups;
        }
        let rows = declared
            .iter()
            .map(|(table, rows)| TableRows::new(*table, rows))
            .collect::<Vec<TableRows<'_>>>();
        let encoding = self.encoding();
        let groups = Arc::new(match self.weights.paged() {
            true => {
                self.planned_windows.fetch_add(1, Ordering::Relaxed);
                weight_groups(
                    &self.plan,
                    &encoding,
                    self.weights.resident_pages(),
                    &self.plan.weight_pages(&encoding, &rows),
                )
            }
            false => Vec::new(),
        });
        while windows.len() >= REMEMBERED_WINDOWS {
            windows.remove(0);
        }
        windows.push(Windows {
            extents,
            rows: declared,
            groups: groups.clone(),
        });
        groups
    }

    pub fn planned_windows(&self) -> usize {
        self.planned_windows.load(Ordering::Relaxed)
    }

    pub(crate) fn store(&self) -> &Arc<WeightStore> {
        self.weights.store()
    }

    pub(crate) fn write_records(&self, queue: &Queue) {
        let encoding = self.encoding();
        self.values.buffer().write(queue, encoding.values());
        self.tasks.buffer().write(queue, encoding.tasks());
        self.segments
            .buffer()
            .write(queue, bytemuck::cast_slice(encoding.segments()));
        self.progress.buffer().write(
            queue,
            bytemuck::cast_slice(&progress::words(
                encoding.segments().len() as u32,
                encoding.wave_tasks(),
            )),
        );
        if let (Some(patches), Some(patch_list)) = (self.patches.as_ref(), self.patch_list.as_ref())
        {
            patches
                .buffer()
                .write(queue, bytemuck::cast_slice(encoding.patches()));
            patch_list
                .buffer()
                .write(queue, bytemuck::cast_slice(encoding.patch_list()));
        }
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
        let tensors = self.tensors.lock().unwrap_or_else(PoisonError::into_inner);
        Placement::new(tensors.held().word(), self.weights.offset() / WORD_BYTES)
    }

    pub(crate) fn lives_on(&self, heap: &Arc<crate::heap::Heap>) -> bool {
        self.tensors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .held()
            .lives_on(heap)
    }

    pub fn heap(&self) -> GpuBuffer {
        self.tensors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .held()
            .buffer()
            .clone()
    }

    pub fn heap_bytes(&self) -> u64 {
        self.tensors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .held()
            .heap()
            .bytes()
    }

    pub fn tensor_bytes(&self) -> u64 {
        match self
            .encoding
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            Some(encoding) => encoding.tensor_bytes(),
            None => self.plan.tensor_bytes(),
        }
    }

    pub fn arena_bytes(&self) -> u64 {
        match self
            .encoding
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
        {
            Some(encoding) => encoding.arena_bytes(),
            None => self.plan.arena_bytes(),
        }
    }

    pub fn resident_bytes(&self) -> u64 {
        self.plan.resident_bytes()
    }

    pub fn weights(&self) -> &Weights {
        &self.weights
    }

    pub fn device_bytes(&self) -> u64 {
        self.tensors
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .held()
            .heap()
            .bytes()
            + self.tasks.buffer().size()
            + self.values.buffer().size()
            + self.segments.buffer().size()
            + self.resident.steps.buffer().size()
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

    pub fn remembered_encodings(&self) -> usize {
        self.plan.remembered_encodings()
    }

    pub fn remembered_bytes(&self) -> u64 {
        self.plan.remembered_bytes()
    }

    pub fn derived_encodings(&self) -> u64 {
        self.plan.derived_encodings()
    }

    pub fn task_count(&self) -> u32 {
        self.stored_encoding().task_count()
    }

    pub fn step_count(&self) -> u32 {
        self.plan.step_count()
    }

    pub fn wave_count(&self) -> u32 {
        self.stored_encoding().wave_count()
    }

    pub fn weight_windows(&self) -> u32 {
        self.windows().len() as u32
    }

    pub fn workgroups(&self) -> u32 {
        let encoding = self.stored_encoding();
        (encoding.segments().len() as u32)
            .min(self.plan.profile().workgroups())
            .max(1)
    }

    pub(crate) fn header(&self) -> Vec<u8> {
        self.stored_encoding().header()
    }

    fn stored_encoding(&self) -> Arc<Encoding> {
        self.encoding
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
            .unwrap_or_else(|| self.plan.bound_encoding())
    }

    pub fn value_count(&self) -> u32 {
        self.plan.value_count()
    }

    pub fn work(&self) -> u64 {
        self.stored_encoding().work()
    }

    pub fn readable(&self, value: Value) -> bool {
        self.encoding().readable(value.id())
    }

    pub fn updates_weights(&self) -> bool {
        self.plan.updates_weights()
    }

    pub fn span(&self, value: Value) -> Span {
        let host = self.host_extents();
        self.span_with(value, &host)
    }
}
