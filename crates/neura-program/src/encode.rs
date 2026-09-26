use crate::access::{self, Access, Reads};
use crate::fuse;
use crate::layout::{Layout, Region, store_of};
use crate::lower;
use crate::lower::Task;
use crate::schedule::{self, Dispatch};
use neura_abi::{
    BoundsFields, BoundsRecord, Element, Kind, Placement, SegmentRecord, StepRecord, Store,
    TaskFields, TaskRecord, ValueFields, ValueRecord, WORD_BYTES,
};
use neura_graph::{Graph, GraphSnapshot, Residency, Value, ValueInfo};
use neura_profile::{AttentionTile, MatmulTile, Profile};
use std::mem::size_of;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub store: Store,
    pub offset: u64,
    pub elements: u32,
    pub element: Element,
}

struct Block {
    offset: u64,
    bytes: u64,
}

struct Blocks {
    free: Vec<Block>,
    end: u64,
}

impl Blocks {
    fn with_base(base: u64) -> Self {
        Self {
            free: Vec::new(),
            end: base,
        }
    }

    fn bytes(&self) -> u64 {
        self.end
    }

    fn reserve(&mut self, bytes: u64, alignment: u64) -> u64 {
        for index in 0..self.free.len() {
            let block = &self.free[index];
            let offset = block.offset.next_multiple_of(alignment);
            let end = block.offset + block.bytes;
            if offset + bytes > end {
                continue;
            }
            if offset + bytes == end {
                self.free.remove(index);
            } else {
                self.free[index].offset = offset + bytes;
                self.free[index].bytes = end - offset - bytes;
            }
            return offset;
        }
        let offset = self.end.next_multiple_of(alignment);
        self.end = offset + bytes;
        offset
    }

    fn release(&mut self, offset: u64, bytes: u64) {
        self.free.push(Block { offset, bytes });
        self.free.sort_by_key(|block| block.offset);
        let mut merged: Vec<Block> = Vec::with_capacity(self.free.len());
        for block in self.free.drain(..) {
            match merged.last_mut() {
                Some(last) if last.offset + last.bytes >= block.offset => {
                    last.bytes = last.bytes.max(block.offset + block.bytes - last.offset);
                }
                _ => merged.push(block),
            }
        }
        self.free = merged;
    }
}

#[derive(Clone, Copy)]
struct Live {
    first: usize,
    last: usize,
    reads: u32,
    aliased_at: Option<usize>,
}

#[derive(Clone, Copy)]
struct Active {
    live: Live,
    offset: u64,
    bytes: u64,
}

#[derive(Clone, Copy)]
struct Placed {
    store: Store,
    address: u64,
    elements: u32,
    element: Element,
}

pub struct Encoding {
    profile: Profile,
    kinds: Vec<Kind>,
    elements: Vec<Element>,
    tasks: Vec<u8>,
    values: Vec<u8>,
    bounds: Vec<u8>,
    steps: Vec<u8>,
    segments: Vec<SegmentRecord>,
    dispatches: Vec<Dispatch>,
    spans: Vec<Option<Placed>>,
    readable: Vec<bool>,
    geometries: Vec<u32>,
    attention: Vec<AttentionTile>,
    arena_bytes: u64,
    tensor_bytes: u64,
    layout: Layout,
    updates_weights: bool,
    work: u64,
}

impl Encoding {
    pub fn of(graph: &Graph<'_>, alignment: u64, profile: Profile) -> Self {
        Self::plan(&graph.snapshot(), profile, alignment)
    }

    fn plan(state: &GraphSnapshot, profile: Profile, alignment: u64) -> Self {
        assert!(
            alignment.is_power_of_two() && alignment >= 4,
            "an arena alignment of {alignment} bytes is not usable",
        );
        let plan = lower::lower(state.values(), &fuse::fuse(state), profile);
        let values = &plan.values;
        let tasks = &plan.tasks;
        let tiles = &plan;
        let matmul_tiles = profile.tiles();

        let kinds = carried_kinds(tasks);
        let elements = carried_elements(values);
        let layout = Layout::of_values(values, alignment);
        assert_writes_match_their_element(values, tasks);
        assert_writers_precede_readers(values, tasks);
        assert_units_keep_their_order(tasks);
        let schedule = schedule::Schedule::of(values, tasks);
        let order = schedule.order();
        let ends = schedule.ends();
        let live = storage_liveness(values, tasks, order);
        let reserved = layout.tensors().bytes();
        let (offsets, tensor_bytes) = allocate(values, &live, ends, alignment, reserved);
        let arena_bytes = tensor_bytes - reserved;
        assert!(
            tensor_bytes.is_multiple_of(WORD_BYTES),
            "a plan of {tensor_bytes} bytes leaves the word grid the device indexes",
        );

        let mut records = Vec::new();
        for (id, info) in values.iter().enumerate() {
            let address = layout.address(values, &offsets, id as u32);
            let record = ValueRecord::of(ValueFields {
                base: u32::try_from(address).unwrap_or_else(|_| {
                    panic!("value {id} lies at {address}, beyond the device address space")
                }),
                store: layout.store(values, id as u32).code(),
                element: layout.element(values, id as u32).code(),
                dims: info.shape.dims(),
                strides: info.strides,
            });
            records.extend_from_slice(bytemuck::bytes_of(&record));
        }

        let mut tape = Vec::with_capacity(tasks.len() * size_of::<TaskRecord>());
        let mut steps = Vec::new();
        let mut updates_weights = false;
        let mut geometries = vec![0u32; matmul_tiles.len()];
        let mut work = 0;
        for index in order {
            let task = &tasks[*index as usize];
            let geometry = match task.kind {
                Kind::Attention
                | Kind::AttentionQueryGrad
                | Kind::AttentionKeyGrad
                | Kind::AttentionValueGrad => {
                    assert!(
                        (task.geometry as usize) < tiles.attention.len(),
                        "an attention names geometry {} beyond the {} tiles its plan carries",
                        task.geometry,
                        tiles.attention.len(),
                    );
                    task.geometry
                }
                Kind::Matmul => {
                    assert!(
                        (task.geometry as usize) < matmul_tiles.len(),
                        "a product names geometry {} beyond the {} tiles its profile carries",
                        task.geometry,
                        matmul_tiles.len(),
                    );
                    geometries[task.geometry as usize] += 1;
                    task.geometry
                }
                Kind::Argmax
                | Kind::Categorical
                | Kind::SumAxis
                | Kind::Conv2dWeightGrad
                | Kind::Pack => task.geometry,
                Kind::Binary
                | Kind::Unary
                | Kind::Partial
                | Kind::Fill
                | Kind::Broadcast
                | Kind::SumChunk
                | Kind::Softmax
                | Kind::SoftmaxGrad
                | Kind::LogSoftmax
                | Kind::LogSoftmaxGrad
                | Kind::OneHot
                | Kind::Gather
                | Kind::Scatter
                | Kind::Conv2d
                | Kind::Conv2dInputGrad
                | Kind::PoolMax2d
                | Kind::PoolMax2dInputGrad
                | Kind::PoolMean2d
                | Kind::PoolMean2dInputGrad
                | Kind::MatmulFold => 0,
            };
            assert!(
                task.kind != Kind::SumChunk || task.chain.is_empty(),
                "a reduction task writes one slot per task and carries no chain",
            );
            assert!(
                task.prelude.is_empty() || task.kind.takes_prelude(),
                "a {} task opens with a prelude no device body of it reads",
                task.kind.name(),
            );
            assert!(
                task.kind != Kind::Matmul || task.splits == 1 || task.chain.is_empty(),
                "a product split across the depth hands its chain to the fold",
            );
            let prelude = (steps.len() / size_of::<StepRecord>()) as u32;
            for step in &task.prelude {
                steps.extend_from_slice(bytemuck::bytes_of(step));
            }
            let chain = (steps.len() / size_of::<StepRecord>()) as u32;
            for step in &task.chain {
                steps.extend_from_slice(bytemuck::bytes_of(step));
            }
            let record = TaskRecord::of(TaskFields {
                kind: task.kind.code(),
                op: task.op,
                geometry,
                first: task.first,
                count: task.count,
                slot: task.slot,
                splits: task.splits,
                out: task.out,
                extra: task.extra,
                a: task.inputs[0],
                b: task.inputs[1],
                c: task.inputs[2],
                d: task.inputs[3],
                e: task.inputs[4],
                f: task.inputs[5],
                param: task.param,
                prelude,
                prelude_steps: task.prelude.len() as u32,
                chain,
                steps: task.chain.len() as u32,
                reach_rows: task.window.reach_rows(),
                reach_columns: task.window.reach_columns(),
                stride_rows: task.window.stride_rows(),
                stride_columns: task.window.stride_columns(),
                pad_rows: task.window.pad_rows(),
                pad_columns: task.window.pad_columns(),
            });
            if task.in_place
                && task
                    .writes()
                    .any(|out| store_of(info_of(values, out).residency) == Store::Weights)
            {
                updates_weights = true;
            }
            work += task.work;
            tape.extend_from_slice(bytemuck::bytes_of(&record));
        }

        let mut bounds = Vec::new();
        for dispatch in schedule.dispatches() {
            let record = BoundsRecord::of(BoundsFields {
                first_segment: dispatch.first_segment,
            });
            bounds.extend_from_slice(bytemuck::bytes_of(&record));
            bounds.resize(bounds.len().next_multiple_of(alignment as usize), 0);
        }
        let segments = schedule.segments().to_vec();

        let mut readable = vec![false; values.len()];
        let mut last_writer = std::collections::HashMap::<u64, u32>::new();
        for index in order {
            let task = &tasks[*index as usize];
            for out in task.writes() {
                let storage = values[out as usize].storage as usize;
                if !arena_resident(values, storage) {
                    continue;
                }
                last_writer.insert(offsets[storage], out);
            }
        }
        for (id, info) in values.iter().enumerate() {
            if info.storage as usize != id {
                continue;
            }
            readable[id] = match info.residency {
                Residency::Parameter | Residency::Resident => true,
                _ => match last_writer.get(&offsets[id]) {
                    Some(writer) => *writer == id as u32,
                    None => held(values, id),
                },
            };
        }
        let mut spans = vec![None; values.len()];
        for (id, info) in values.iter().enumerate() {
            if info.storage as usize != id {
                continue;
            }
            spans[id] = Some(Placed {
                store: layout.store(values, id as u32),
                address: layout.address(values, &offsets, id as u32),
                elements: info.shape.elements(),
                element: layout.element(values, id as u32),
            });
        }

        Self {
            profile,
            kinds,
            elements,
            tasks: tape,
            values: records,
            bounds,
            steps,
            segments,
            dispatches: schedule.dispatches().to_vec(),
            spans,
            readable,
            geometries,
            attention: tiles.attention.clone(),
            arena_bytes,
            tensor_bytes,
            layout,
            updates_weights,
            work,
        }
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        self.profile.tiles()
    }

    pub fn kinds(&self) -> &[Kind] {
        &self.kinds
    }

    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    pub fn attention(&self) -> &[AttentionTile] {
        &self.attention
    }

    pub fn matmul_geometries(&self) -> Vec<(MatmulTile, u32)> {
        self.tiles()
            .iter()
            .copied()
            .zip(self.geometries.iter().copied())
            .filter(|(_, count)| *count > 0)
            .collect()
    }

    pub fn walked_tiles(&self) -> impl Iterator<Item = (u32, MatmulTile)> + '_ {
        self.geometries
            .iter()
            .enumerate()
            .filter(|(_, count)| **count > 0)
            .map(|(index, _)| (index as u32, self.tiles()[index]))
    }

    pub fn tasks(&self) -> &[u8] {
        &self.tasks
    }

    pub fn values(&self) -> &[u8] {
        &self.values
    }

    pub fn bounds(&self) -> &[u8] {
        &self.bounds
    }

    pub fn steps(&self) -> &[u8] {
        &self.steps
    }

    pub fn segments(&self) -> &[SegmentRecord] {
        &self.segments
    }

    pub fn dispatches(&self) -> &[Dispatch] {
        &self.dispatches
    }

    pub fn span(&self, value: Value<'_>, placement: Placement) -> Span {
        let placed = self
            .spans
            .get(value.id() as usize)
            .copied()
            .flatten()
            .unwrap_or_else(|| {
                panic!(
                    "{} elements of a view hold no storage of their own",
                    value.shape().elements(),
                )
            });
        let offset = match placed.store {
            Store::Weights => (placement.weights() + placed.address) * WORD_BYTES,
            Store::Tensors => (placement.tensors() + placed.address) * WORD_BYTES,
        };
        Span {
            store: placed.store,
            offset,
            elements: placed.elements,
            element: placed.element,
        }
    }

    pub fn readable(&self, value: Value<'_>) -> bool {
        self.readable
            .get(value.id() as usize)
            .copied()
            .unwrap_or(false)
    }

    pub fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    pub fn weights(&self) -> &Region {
        self.layout.weights()
    }

    pub fn tensors(&self) -> &Region {
        self.layout.tensors()
    }

    pub fn tensor_bytes(&self) -> u64 {
        self.tensor_bytes
    }

    pub fn resident_bytes(&self) -> u64 {
        self.layout.tensors().bytes()
    }

    pub fn updates_weights(&self) -> bool {
        self.updates_weights
    }

    pub fn task_count(&self) -> u32 {
        (self.tasks.len() / size_of::<TaskRecord>()) as u32
    }

    pub fn step_count(&self) -> u32 {
        (self.steps.len() / size_of::<StepRecord>()) as u32
    }

    pub fn value_count(&self) -> u32 {
        (self.values.len() / size_of::<ValueRecord>()) as u32
    }

    pub fn dispatch_count(&self) -> u32 {
        self.dispatches.len() as u32
    }

    pub fn work(&self) -> u64 {
        self.work
    }
}

fn carried_kinds(tasks: &[Task]) -> Vec<Kind> {
    Kind::ALL
        .iter()
        .copied()
        .filter(|kind| {
            tasks.iter().any(|task| task.kind == *kind)
                || (*kind == Kind::MatmulFold && tasks.iter().any(|task| task.kind == Kind::Matmul))
        })
        .collect()
}

fn carried_elements(values: &[ValueInfo]) -> Vec<Element> {
    Element::ALL
        .iter()
        .copied()
        .filter(|element| values.iter().any(|info| info.element == *element))
        .collect()
}

fn assert_writes_match_their_element(values: &[ValueInfo], tasks: &[Task]) {
    for task in tasks {
        for out in task.writes() {
            let out = &values[out as usize];
            if task.kind == Kind::Pack {
                let source = &values[task.inputs[0] as usize];
                assert!(
                    out.element.narrow() && source.element == Element::Single,
                    "a pack of {} numbers into a {} tensor reads the {} source {} writes element by element",
                    source.shape.elements(),
                    out.element.name(),
                    source.element.name(),
                    task.kind.name(),
                );
                continue;
            }
            assert!(
                !out.element.narrow(),
                "a {} task writes the {} tensor {} element by element, and a narrow tensor is written a word at a time by a pack",
                task.kind.name(),
                out.element.name(),
                task.out,
            );
        }
    }
}

fn assert_writers_precede_readers(values: &[ValueInfo], tasks: &[Task]) {
    let mut last_writer = vec![None::<usize>; values.len()];
    for (position, task) in tasks.iter().enumerate() {
        let access = Access::of(values, task);
        for storage in access.reads() {
            if access.in_place() && access.writes().contains(storage) {
                continue;
            }
            match last_writer[*storage as usize] {
                Some(writer) => assert!(
                    writer < position,
                    "task {position} reads a tensor that its own tape only writes later",
                ),
                None => assert!(
                    held(values, *storage as usize),
                    "task {position} reads a tensor no task of the tape writes before it",
                ),
            }
        }
        for storage in access.writes() {
            last_writer[*storage as usize] = Some(position);
        }
    }
}

fn assert_units_keep_their_order(tasks: &[Task]) {
    for pair in tasks.windows(2) {
        assert!(
            pair[0].unit <= pair[1].unit,
            "a task of the fold that became unit {} stands before a task of unit {}",
            pair[1].unit,
            pair[0].unit,
        );
    }
}

fn storage_liveness(values: &[ValueInfo], tasks: &[Task], order: &[u32]) -> Vec<Option<Live>> {
    let mut live = vec![None::<Live>; values.len()];
    let mut readers = Vec::new();
    for (position, index) in order.iter().enumerate() {
        let task = &tasks[*index as usize];
        let writes = Access::of(values, task).writes().to_vec();
        let aliases = writes
            .first()
            .is_some_and(|write| reads_every_element_in_place(values, task, *write));
        for write in &writes {
            touch(&mut live, *write, position, false, None);
        }
        readers.clear();
        for value in task.reads() {
            if readers.contains(&value) {
                continue;
            }
            readers.push(value);
            let storage = access::storage(values, value);
            let in_place = aliases && !writes.contains(&storage);
            touch(
                &mut live,
                storage,
                position,
                true,
                in_place.then_some(position),
            );
        }
    }
    live
}

fn reads_every_element_in_place(values: &[ValueInfo], task: &Task, write: u32) -> bool {
    if !matches!(
        task.kind,
        Kind::Binary | Kind::Unary | Kind::Partial | Kind::Fill | Kind::Broadcast
    ) {
        return false;
    }
    let out = &values[write as usize];
    task.reads().all(|value| {
        let value = &values[value as usize];
        value.shape == out.shape && value.strides == out.strides
    })
}

fn touch(
    live: &mut [Option<Live>],
    storage: u32,
    position: usize,
    read: bool,
    aliased_at: Option<usize>,
) {
    match &mut live[storage as usize] {
        Some(entry) => {
            entry.first = entry.first.min(position);
            entry.last = entry.last.max(position);
            entry.reads += u32::from(read);
            entry.aliased_at = entry.aliased_at.max(aliased_at);
        }
        slot @ None => {
            *slot = Some(Live {
                first: position,
                last: position,
                reads: u32::from(read),
                aliased_at,
            });
        }
    }
}

fn storage_bytes(values: &[ValueInfo], storage: usize) -> u64 {
    u64::from(values[storage].shape.elements()) * WORD_BYTES
}

fn info_of(values: &[ValueInfo], value: u32) -> &ValueInfo {
    &values[values[value as usize].storage as usize]
}

fn arena_resident(values: &[ValueInfo], storage: usize) -> bool {
    matches!(
        values[storage].residency,
        Residency::Input | Residency::Derived
    )
}

fn held(values: &[ValueInfo], storage: usize) -> bool {
    matches!(
        values[storage].residency,
        Residency::Input | Residency::Parameter | Residency::Resident
    )
}

fn allocate(
    values: &[ValueInfo],
    live: &[Option<Live>],
    waves: &[u32],
    alignment: u64,
    reserved: u64,
) -> (Vec<u64>, u64) {
    let wave_of = |position: usize| waves.partition_point(|end| *end <= position as u32) as u32;
    let mut arena = Blocks::with_base(reserved);
    let mut offsets = vec![0u64; live.len()];
    let owners = (0..values.len()).filter(|id| values[*id].storage as usize == *id);
    for id in owners.clone() {
        if !arena_resident(values, id) {
            continue;
        }
        if held(values, id) || values[id].retained {
            offsets[id] = arena.reserve(storage_bytes(values, id), alignment);
        }
    }
    let mut pending = live
        .iter()
        .enumerate()
        .filter(|(id, _)| !held(values, *id) && !values[*id].retained)
        .filter_map(|(id, live)| live.map(|live| (id, live)))
        .collect::<Vec<_>>();
    pending.sort_by_key(|(_, live)| (live.first, live.last));
    let mut active = Vec::<Active>::new();
    for (storage, live) in pending {
        let wave = wave_of(live.first);
        let taken = active
            .iter()
            .position(|held| held.live.reads == 1 && held.live.aliased_at == Some(live.first));
        if let Some(position) = taken {
            let held = active.swap_remove(position);
            let bytes = storage_bytes(values, storage);
            assert_eq!(
                held.bytes, bytes,
                "a value read in place by one task hands that task a storage of another size",
            );
            offsets[storage] = held.offset;
            active.push(Active {
                live,
                offset: held.offset,
                bytes,
            });
            continue;
        }
        active.retain(|held| {
            if wave_of(held.live.last) < wave {
                arena.release(held.offset, held.bytes);
                false
            } else {
                true
            }
        });
        let bytes = storage_bytes(values, storage);
        let offset = arena.reserve(bytes, alignment);
        offsets[storage] = offset;
        active.push(Active {
            live,
            offset,
            bytes,
        });
    }
    (offsets, arena.bytes())
}
