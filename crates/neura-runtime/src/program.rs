use crate::cache::{self, Resident};
use crate::heap::Allocation;
use crate::pool::Recycled;
use crate::store::WeightStore;
use neura_abi::{Placement, PlacementFields, PlacementRecord, SegmentRecord, WORD_BYTES, control};
use neura_gpu::Queue;
use neura_gpu::{
    BindGroup, Binding, BufferUsages, GpuBuffer, GpuContext, READBACK_TIMEOUT, Submission,
    SubmissionIndex,
};
use neura_graph::{GraphStamp, Revision, Value};
use neura_kernel::{HEAP, TASKS, Tables, VALUES};
use neura_plan::{Arena, Encoding, Plan, Span, TableRows};
use neura_profile::{MatmulTile, Profile};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

mod window;

pub(crate) use window::WeightSchedule;
use window::weight_schedule;

const REMEMBERED_WINDOWS: usize = 8;
static INSTANCES: AtomicU64 = AtomicU64::new(1);

#[derive(Clone)]
pub struct Weights {
    store: Arc<WeightStore>,
    weights: Arena,
    state: Arena,
}

impl Weights {
    pub(crate) fn new(store: Arc<WeightStore>, weights: Arena, state: Arena) -> Self {
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

    pub fn pinned_pages(&self) -> u32 {
        self.store.pinned_pages()
    }

    pub fn readback_pages(&self) -> u64 {
        self.store.readback_pages()
    }

    pub fn readback_transfers(&self) -> u64 {
        self.store.readback_transfers()
    }

    pub fn read_peak_bytes(&self) -> u64 {
        self.store.read_peak_bytes()
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

    pub(crate) fn region(&self) -> &Arena {
        &self.weights
    }

    pub(crate) fn state(&self) -> &Arena {
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
    instance: u64,
    tables: Tables,
    cached: Mutex<Option<Vec<u32>>>,
    pub(crate) state: Recycled,
    group: Mutex<Option<BindGroup>>,
    pub(crate) weights: Weights,
    table_buffer: Recycled,
    segments: Recycled,
    successors: Mutex<Successors>,
    generation: Mutex<u32>,
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
    schedule: Arc<WeightSchedule>,
}

pub(crate) struct Dispatch<'a> {
    pub(crate) generation: u32,
    pub(crate) clear: bool,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) counts: Option<&'a [u32]>,
    pub(crate) roots: &'a [u32],
    pub(crate) waves: &'a [u32],
    pub(crate) wave_table: u32,
    pub(crate) wave: u32,
}

struct Successors {
    active: Recycled,
    retired: Vec<Recycled>,
}

impl Successors {
    fn new(active: Recycled) -> Self {
        Self {
            active,
            retired: Vec::new(),
        }
    }

    fn buffer(&self) -> &GpuBuffer {
        self.active.buffer()
    }

    fn write(&mut self, queue: &Queue, bytes: &[u8]) {
        self.active.buffer().write(queue, bytes);
    }

    fn write_at(&mut self, queue: &Queue, offset: u64, bytes: &[u8]) {
        self.active.buffer().write_at(queue, offset, bytes);
    }

    fn grow(&mut self, queue: &Queue, bytes: &[u8], fresh: Recycled) {
        self.retired
            .push(std::mem::replace(&mut self.active, fresh));
        self.write(queue, bytes);
    }
}

fn counters(encoding: &Encoding, first: usize, count: usize) -> Vec<u32> {
    (first..first + count)
        .map(|segment| encoding.predecessors(segment, first))
        .collect()
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
        let tables = cache::tables(&plan);
        let table_buffer = Recycled::claim(
            pool,
            "neura tables",
            tables.bytes().max(WORD_BYTES),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let queue = context.queue();
        let state_buffer = Recycled::claim(
            pool,
            "neura state",
            control::bytes(plan.task_count()),
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let segments = Recycled::claim(
            pool,
            "neura segments",
            (plan.task_count() as u64 * size_of::<SegmentRecord>() as u64).max(4),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let bound_encoding = plan.bound_encoding();
        let successors = Recycled::claim(
            pool,
            "neura successors",
            ((bound_encoding.successors().len() + bound_encoding.wave_table().len()) as u64
                * WORD_BYTES)
                .next_power_of_two()
                .max(4),
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let tensors = Tensors {
            active: Some(tensors),
            retired: Vec::new(),
        };
        let arena = tensors.held().clone();
        let mut clearing = Submission::new(context.device(), "neura tensors");
        if arena.bytes() > 0 {
            clearing.clear(arena.buffer(), arena.offset(), arena.bytes());
        }
        clearing.submit(queue);
        if !plan.awaits_a_binding() {
            for quantum in plan.quanta() {
                arena.buffer().write_at(
                    queue,
                    arena.offset() + quantum.offset,
                    &quantum.scale.to_ne_bytes(),
                );
            }
            table_buffer.buffer().write_at(
                queue,
                tables.placement() as u64 * WORD_BYTES,
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
        successors
            .buffer()
            .write(queue, bytemuck::cast_slice(bound_encoding.successors()));
        successors.buffer().write_at(
            queue,
            bound_encoding.successors().len() as u64 * WORD_BYTES,
            bytemuck::cast_slice(bound_encoding.wave_table()),
        );
        state_buffer.buffer().write_at(
            queue,
            u64::from(control::COUNTERS) * WORD_BYTES,
            bytemuck::cast_slice(&counters(
                &bound_encoding,
                0,
                bound_encoding.segments().len(),
            )),
        );
        state_buffer.buffer().write_at(
            queue,
            u64::from(control::wave(plan.task_count(), 0)) * WORD_BYTES,
            bytemuck::cast_slice(bound_encoding.wave_segments()),
        );
        if !plan.measures().is_empty() {
            table_buffer.buffer().write_at(
                queue,
                tables.measures_first() as u64 * WORD_BYTES,
                bytemuck::cast_slice(plan.measures()),
            );
        }
        table_buffer.buffer().write_at(
            queue,
            tables.patches_first() as u64 * WORD_BYTES,
            bytemuck::cast_slice(bound_encoding.patches()),
        );
        table_buffer.buffer().write_at(
            queue,
            tables.list_first() as u64 * WORD_BYTES,
            bytemuck::cast_slice(bound_encoding.patch_list()),
        );
        if plan.carries_authored() {
            table_buffer.buffer().write_at(
                queue,
                tables.extents() as u64 * WORD_BYTES,
                bytemuck::cast_slice(&plan.host_extents()),
            );
        }
        let paged = weights.paged();
        let bound = plan.host_extents();
        let preview = match paged && !plan.dynamic() && plan.gather_tables().is_empty() {
            true => {
                let schedule = weight_schedule(
                    &plan,
                    &bound_encoding,
                    weights.resident_pages(),
                    weights.pages(),
                    &plan.weight_pages(&bound_encoding, &[]),
                );
                Some(Windows {
                    extents: bound,
                    rows: Vec::new(),
                    schedule: Arc::new(schedule),
                })
            }
            false => None,
        };
        let previewed = usize::from(preview.is_some());
        let bound = Bound::of(&plan);
        let encoding = (!plan.awaits_a_binding()).then(|| plan.bound_encoding());
        let program = Self {
            resident,
            instance: INSTANCES.fetch_add(1, Ordering::Relaxed),
            tables,
            cached: Mutex::new(None),
            group: Mutex::new(None),
            tensors: Mutex::new(tensors),
            weights,
            state: state_buffer,
            table_buffer,
            segments,
            successors: Mutex::new(Successors::new(successors)),
            generation: Mutex::new(0),
            revision,
            tasks,
            values,
            bound: Mutex::new(bound),
            windows: Mutex::new(preview.into_iter().collect()),
            planned_windows: AtomicUsize::new(previewed),
            encoding: Mutex::new(encoding),
            last: Mutex::new(None),
            plan,
        };
        program.rebuild();
        program
    }

    pub(crate) fn group(&self) -> BindGroup {
        self.group
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .as_ref()
            .expect("a program binds its buffers when it is born")
            .clone()
    }

    pub(crate) fn begin(&self) -> (u32, bool) {
        let mut generation = self
            .generation
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *generation = generation.wrapping_add(1);
        let wrapped = *generation > 0xff;
        if wrapped {
            *generation = 1;
        }
        (*generation, wrapped)
    }

    pub(crate) fn write_dispatch(&self, queue: &Queue, dispatch: Dispatch<'_>) {
        let capacity = self.plan.task_count();
        let queue_base = control::queue(capacity, 0);
        assert!(
            dispatch.first as u64 + u64::from(dispatch.count) <= u64::from(capacity),
            "a dispatch of {} segments at {} outruns the {capacity} a plan holds",
            dispatch.count,
            dispatch.first,
        );
        assert!(
            dispatch.roots.len() <= capacity as usize,
            "a dispatch queues {} roots beyond the {capacity} a plan holds",
            dispatch.roots.len(),
        );
        if dispatch.clear {
            let mut submission = Submission::new(queue.device(), "neura queue");
            submission.clear(
                self.state.buffer(),
                u64::from(queue_base) * WORD_BYTES,
                u64::from(capacity) * WORD_BYTES,
            );
            submission.submit(queue);
        }
        self.state.buffer().write_at(
            queue,
            control::offset_bytes(),
            bytemuck::cast_slice(&control::header(
                dispatch.generation,
                dispatch.roots.len() as u32,
                dispatch.count,
                dispatch.first,
                queue_base,
                control::wave(capacity, 0),
                dispatch.wave_table,
            )),
        );
        if let Some(counts) = dispatch.counts {
            assert_eq!(
                counts.len(),
                dispatch.count as usize,
                "a dispatch of {} segments carries {} predecessor counts",
                dispatch.count,
                counts.len(),
            );
            self.state.buffer().write_at(
                queue,
                u64::from(control::COUNTERS + dispatch.first) * WORD_BYTES,
                bytemuck::cast_slice(counts),
            );
        }
        self.state.buffer().write_at(
            queue,
            u64::from(control::wave(capacity, 0)) * WORD_BYTES,
            bytemuck::cast_slice(dispatch.waves),
        );
        let entries = dispatch
            .roots
            .iter()
            .map(|root| control::entry(dispatch.generation, *root))
            .collect::<Vec<u32>>();
        if dispatch.wave != 0 {
            self.state.buffer().write_at(
                queue,
                u64::from(control::PREFIX) * WORD_BYTES,
                bytemuck::cast_slice(&[dispatch.wave]),
            );
        }
        self.state.buffer().write_at(
            queue,
            u64::from(queue_base) * WORD_BYTES,
            bytemuck::cast_slice(&entries),
        );
    }

    fn write_successors(&self, queue: &Queue, encoding: &Encoding) {
        let bytes = ((encoding.successors().len() + encoding.wave_table().len()) as u64
            * WORD_BYTES)
            .max(4);
        let mut successors = self
            .successors
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if successors.buffer().size() < bytes {
            let fresh = Recycled::claim(
                self.resident.pool(),
                "neura successors",
                bytes.next_power_of_two(),
                BufferUsages::STORAGE | BufferUsages::COPY_DST,
            );
            successors.grow(queue, bytemuck::cast_slice(encoding.successors()), fresh);
            successors.write_at(
                queue,
                encoding.successors().len() as u64 * WORD_BYTES,
                bytemuck::cast_slice(encoding.wave_table()),
            );
            drop(successors);
            self.rebuild();
            return;
        }
        successors.write(queue, bytemuck::cast_slice(encoding.successors()));
        successors.write_at(
            queue,
            encoding.successors().len() as u64 * WORD_BYTES,
            bytemuck::cast_slice(encoding.wave_table()),
        );
    }

    fn rebuild(&self) {
        let group = self.bind_group();
        *self.group.lock().unwrap_or_else(PoisonError::into_inner) = Some(group);
    }

    fn bind_group(&self) -> BindGroup {
        let successors = self
            .successors
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let arena = self.tensors.lock().unwrap_or_else(PoisonError::into_inner);
        let held = arena.held();
        let heap = held.heap();
        let banks = heap.banks();
        let bank_bytes = heap.bank_bytes();
        let paged = self.weights.paged();
        let mut bindings = vec![
            Binding {
                index: TASKS,
                buffer: self.tasks.buffer().binding(0, self.tasks.buffer().size()),
            },
            Binding {
                index: VALUES,
                buffer: self.values.buffer().binding(0, self.values.buffer().size()),
            },
        ];
        for bank in 0..banks.count() {
            let offset = u64::from(bank) * bank_bytes;
            let size = (heap.bytes() - offset).min(bank_bytes);
            bindings.push(Binding {
                index: HEAP + bank,
                buffer: held.buffer().binding(offset, size),
            });
        }
        if let Some(table) = self.weights.store().table() {
            bindings.push(Binding {
                index: neura_kernel::pages(banks),
                buffer: table.binding(0, table.size()),
            });
        }
        bindings.extend([
            Binding {
                index: neura_kernel::state(banks, paged),
                buffer: self.state.buffer().binding(0, self.state.buffer().size()),
            },
            Binding {
                index: neura_kernel::steps(banks, paged),
                buffer: self
                    .resident
                    .steps
                    .buffer()
                    .binding(0, self.resident.steps.buffer().size()),
            },
            Binding {
                index: neura_kernel::segments(banks, paged),
                buffer: self
                    .segments
                    .buffer()
                    .binding(0, self.segments.buffer().size()),
            },
            Binding {
                index: neura_kernel::successors(banks, paged),
                buffer: successors.buffer().binding(0, successors.buffer().size()),
            },
            Binding {
                index: neura_kernel::tables(banks, paged),
                buffer: self
                    .table_buffer
                    .buffer()
                    .binding(0, self.table_buffer.buffer().size()),
            },
        ]);
        self.resident.kernel.bind_group(&bindings)
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

    pub(crate) fn declare_rows(&self, table: Value, rows: &[u32]) {
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

    pub(crate) fn tables_buffer(&self) -> &GpuBuffer {
        self.table_buffer.buffer()
    }

    pub(crate) fn extents_offset(&self) -> u64 {
        self.tables.extents() as u64 * WORD_BYTES
    }

    pub(crate) fn cached_extents(&self) -> Option<Vec<u32>> {
        self.extents_cache().clone()
    }

    pub(crate) fn cache_extents(&self, extents: Vec<u32>) {
        *self.extents_cache() = Some(extents);
    }

    pub(crate) fn write_extents(&self, queue: &Queue) {
        if !self.carries_authored() {
            return;
        }
        let extents = self.host_extents();
        self.tables_buffer()
            .write_at(queue, self.extents_offset(), bytemuck::cast_slice(&extents));
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

    pub(crate) fn plan(&self) -> Arc<Plan> {
        self.plan.clone()
    }

    pub(crate) fn instance(&self) -> u64 {
        self.instance
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
        self.tables_buffer().write_at(
            queue,
            self.tables.placement() as u64 * WORD_BYTES,
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

    pub(crate) fn windows(&self) -> Arc<WeightSchedule> {
        let extents = self.host_extents();
        let declared = self.declared_rows();
        let mut windows = self.windows.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(position) = windows
            .iter()
            .position(|kept| kept.extents == extents && kept.rows == declared)
        {
            let kept = windows.remove(position);
            let schedule = kept.schedule.clone();
            windows.push(kept);
            return schedule;
        }
        let rows = declared
            .iter()
            .map(|(table, rows)| TableRows::new(*table, rows))
            .collect::<Vec<TableRows<'_>>>();
        let encoding = self.encoding();
        let schedule = Arc::new(match self.weights.paged() {
            true => {
                self.planned_windows.fetch_add(1, Ordering::Relaxed);
                weight_schedule(
                    &self.plan,
                    &encoding,
                    self.weights.resident_pages(),
                    self.weights.pages(),
                    &self.plan.weight_pages(&encoding, &rows),
                )
            }
            false => WeightSchedule::empty(),
        });
        while windows.len() >= REMEMBERED_WINDOWS {
            windows.remove(0);
        }
        windows.push(Windows {
            extents,
            rows: declared,
            schedule: schedule.clone(),
        });
        schedule
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
        self.write_successors(queue, &encoding);
        self.state.buffer().write_at(
            queue,
            u64::from(control::COUNTERS) * WORD_BYTES,
            bytemuck::cast_slice(&counters(&encoding, 0, encoding.segments().len())),
        );
        self.state.buffer().write_at(
            queue,
            u64::from(control::wave(self.plan.task_count(), 0)) * WORD_BYTES,
            bytemuck::cast_slice(encoding.wave_segments()),
        );
        if !encoding.patches().is_empty() {
            self.tables_buffer().write_at(
                queue,
                self.tables.patches_first() as u64 * WORD_BYTES,
                bytemuck::cast_slice(encoding.patches()),
            );
        }
        if !encoding.patch_list().is_empty() {
            self.tables_buffer().write_at(
                queue,
                self.tables.list_first() as u64 * WORD_BYTES,
                bytemuck::cast_slice(encoding.patch_list()),
            );
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
            + self
                .successors
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .buffer()
                .size()
            + self.resident.steps.buffer().size()
            + self.state.buffer().size()
            + self.tables_buffer().size()
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
        self.windows().groups().len() as u32
    }

    pub fn pinned_pages(&self) -> u32 {
        self.weights.pinned_pages()
    }

    pub(crate) fn streams_weights(&self) -> bool {
        self.weights.paged()
    }

    pub fn workgroups(&self) -> u32 {
        let encoding = self.stored_encoding();
        (encoding.segments().len() as u32)
            .min(self.plan.profile().workgroups())
            .max(1)
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

    pub fn wave_work(&self, first: u32, count: u32) -> u64 {
        let encoding = self.stored_encoding();
        let at = encoding.wave_work_at();
        let start = first as usize;
        let end = start + count as usize;
        assert!(
            end < at.len(),
            "waves {first} to {} of a program that walks {} waves",
            first + count,
            at.len() - 1,
        );
        at[end] - at[start]
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
