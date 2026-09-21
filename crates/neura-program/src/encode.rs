use crate::fuse;
use crate::graph::{GraphState, Residency, ValueInfo};
use crate::layout::{Layout, Region, store_of};
use crate::lower;
use crate::lower::Task;
use neura_abi::{
    BoundsRecord, Kind, MatmulTile, Placement, Precision, Profile, StepRecord, Store, TaskRecord,
    ValueRecord, WORD_BYTES,
};
use std::cmp::Reverse;
use std::mem::size_of;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Span {
    pub store: Store,
    pub offset: u64,
    pub elements: u32,
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
}

pub struct Encoding {
    profile: Profile,
    tasks: Vec<u8>,
    values: Vec<u8>,
    bounds: Vec<u8>,
    steps: Vec<u8>,
    waves: Vec<u32>,
    spans: Vec<Option<Placed>>,
    readable: Vec<bool>,
    tiles: Vec<MatmulTile>,
    geometries: Vec<u32>,
    arena_bytes: u64,
    tensor_bytes: u64,
    layout: Layout,
    updates_weights: bool,
    work: u64,
}

impl Encoding {
    pub(crate) fn plan(
        state: &GraphState,
        profile: Profile,
        alignment: u64,
        precision: Precision,
    ) -> Self {
        assert!(
            alignment.is_power_of_two() && alignment >= 4,
            "an arena alignment of {alignment} bytes is not usable",
        );
        let plan = lower::lower(&state.values, &fuse::fuse(state), profile);
        let values = &plan.values;
        let tasks = &plan.tasks;
        let layout = Layout::of(values, precision, alignment);
        let depths = wave_depths(values, tasks);
        let mut order = (0..tasks.len()).collect::<Vec<_>>();
        order.sort_by_key(|index| (depths[*index], Reverse(tasks[*index].work)));

        let waves = wave_ends(&depths, &order);
        assert!(
            waves.len() as u32 <= neura_abi::MAX_WAVES,
            "a tape of {} waves outruns the {} cursor slots one dispatch hands the device",
            waves.len(),
            neura_abi::MAX_WAVES,
        );
        let live = storage_liveness(values, tasks, &order);
        let reserved = layout.tensors().bytes();
        let (offsets, tensor_bytes) = allocate(values, &live, &waves, alignment, reserved);
        let arena_bytes = tensor_bytes - reserved;
        assert!(
            tensor_bytes.is_multiple_of(WORD_BYTES),
            "a plan of {tensor_bytes} bytes leaves the word grid the device indexes",
        );

        let mut records = Vec::new();
        for (id, info) in values.iter().enumerate() {
            let address = layout.address(values, &offsets, id as u32);
            let mut record: ValueRecord = bytemuck::Zeroable::zeroed();
            record.base = u32::try_from(address).unwrap_or_else(|_| {
                panic!("value {id} lies at {address}, beyond the device address space")
            });
            record.store = layout.store(values, id as u32).code();
            record.dims = info.shape.dims();
            record.strides = info.strides;
            records.extend_from_slice(bytemuck::bytes_of(&record));
        }

        let used = used_tiles(profile, tasks, &order);
        let mut tape = Vec::with_capacity(tasks.len() * size_of::<TaskRecord>());
        let mut steps = Vec::new();
        let mut updates_weights = false;
        let mut geometries = vec![0u32; used.len()];
        let mut work = 0;
        for index in &order {
            let task = &tasks[*index];
            let geometry = match task.kind {
                Kind::Matmul => {
                    let geometry = used
                        .iter()
                        .position(|tile| *tile == profile.ladder()[task.geometry as usize])
                        .expect("every tile a tape names is carried by its program");
                    geometries[geometry] += 1;
                    geometry as u32
                }
                Kind::Argmax | Kind::Categorical | Kind::SumAxis | Kind::Conv2dWeightGrad => {
                    task.geometry
                }
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
                | Kind::Conv2d
                | Kind::Conv2dInputGrad => 0,
            };
            assert!(
                task.kind != Kind::SumChunk || task.chain.is_empty(),
                "a reduction task writes one slot per task and carries no chain",
            );
            let mut record: TaskRecord = bytemuck::Zeroable::zeroed();
            record.kind = task.kind.code();
            record.op = task.op;
            record.geometry = geometry;
            record.first = task.first;
            record.count = task.count;
            record.slot = task.slot;
            record.out = task.out;
            record.a = task.inputs[0];
            record.b = task.inputs[1];
            record.c = task.inputs[2];
            record.param = task.param;
            record.chain = (steps.len() / size_of::<StepRecord>()) as u32;
            record.steps = task.chain.len() as u32;
            record.stride_rows = task.window.stride_rows();
            record.stride_columns = task.window.stride_columns();
            record.pad_rows = task.window.pad_rows();
            record.pad_columns = task.window.pad_columns();
            for step in &task.chain {
                steps.extend_from_slice(bytemuck::bytes_of(step));
            }
            if task.in_place && store_of(info_of(values, task.out).residency) == Store::Weights {
                updates_weights = true;
            }
            work += task.work;
            tape.extend_from_slice(bytemuck::bytes_of(&record));
        }

        let mut bounds = Vec::new();
        let mut first = 0;
        for (wave, end) in waves.iter().enumerate() {
            let mut record: BoundsRecord = bytemuck::Zeroable::zeroed();
            record.first_task = first;
            record.task_count = end - first;
            record.wave = wave as u32;
            bounds.extend_from_slice(bytemuck::bytes_of(&record));
            bounds.resize(bounds.len().next_multiple_of(alignment as usize), 0);
            first = *end;
        }

        let mut readable = vec![false; values.len()];
        let mut last_writer = std::collections::HashMap::<u64, u32>::new();
        for index in &order {
            let task = &tasks[*index];
            let storage = values[task.out as usize].storage as usize;
            if !arena_resident(values, storage) {
                continue;
            }
            last_writer.insert(offsets[storage], task.out);
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
            });
        }

        Self {
            profile,
            tasks: tape,
            values: records,
            bounds,
            steps,
            waves,
            spans,
            readable,
            tiles: used,
            geometries,
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
        &self.tiles
    }

    pub fn matmul_geometries(&self) -> Vec<(MatmulTile, u32)> {
        self.tiles
            .iter()
            .copied()
            .zip(self.geometries.iter().copied())
            .collect()
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

    pub fn waves(&self) -> &[u32] {
        &self.waves
    }

    pub fn span(&self, value: crate::graph::Value, placement: Placement) -> Span {
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
            Store::Weights => self.layout.weight_bytes(placement, placed.address),
            Store::Tensors => (placement.tensors() + placed.address) * WORD_BYTES,
        };
        Span {
            store: placed.store,
            offset,
            elements: placed.elements,
        }
    }

    pub fn readable(&self, value: crate::graph::Value) -> bool {
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

    pub fn wave_count(&self) -> u32 {
        self.waves.len() as u32
    }

    pub fn work(&self) -> u64 {
        self.work
    }
}

fn used_tiles(profile: Profile, tasks: &[Task], order: &[usize]) -> Vec<MatmulTile> {
    profile
        .ladder()
        .iter()
        .filter(|tile| {
            order.iter().any(|index| {
                let task = &tasks[*index];
                task.kind == Kind::Matmul && profile.ladder()[task.geometry as usize] == **tile
            })
        })
        .copied()
        .collect()
}

fn wave_depths(values: &[ValueInfo], tasks: &[Task]) -> Vec<u32> {
    assert_writers_precede_readers(values, tasks);
    let mut available = vec![0u32; values.len()];
    let mut deepest_read = vec![0u32; values.len()];
    let mut depths = vec![0u32; tasks.len()];
    for (index, task) in tasks.iter().enumerate() {
        let reads = task.reads().collect::<Vec<_>>();
        let mut depth = reads
            .iter()
            .map(|value| available[values[*value as usize].storage as usize])
            .max()
            .unwrap_or(0);
        if task.in_place {
            depth = depth.max(deepest_read[values[task.out as usize].storage as usize] + 1);
        }
        depths[index] = depth;
        for value in reads {
            if task.in_place && value == task.out {
                continue;
            }
            let storage = values[value as usize].storage as usize;
            deepest_read[storage] = deepest_read[storage].max(depth);
        }
        available[values[task.out as usize].storage as usize] = depth + 1;
    }
    depths
}

fn assert_writers_precede_readers(values: &[ValueInfo], tasks: &[Task]) {
    let mut last_writer = vec![None::<usize>; values.len()];
    for (position, task) in tasks.iter().enumerate() {
        let out = values[task.out as usize].storage as usize;
        for value in task.reads() {
            let storage = values[value as usize].storage as usize;
            if task.in_place && storage == out {
                continue;
            }
            match last_writer[storage] {
                Some(writer) => assert!(
                    tasks[writer].time < task.time,
                    "task {position} reads a tensor that its own tape only writes later",
                ),
                None => assert!(
                    held(values, storage),
                    "task {position} reads a tensor no task of the tape writes before it",
                ),
            }
        }
        last_writer[out] = Some(position);
    }
}

fn wave_ends(depths: &[u32], order: &[usize]) -> Vec<u32> {
    let mut ends = Vec::new();
    let mut cursor = 0usize;
    while cursor < order.len() {
        let depth = depths[order[cursor]];
        while cursor < order.len() && depths[order[cursor]] == depth {
            cursor += 1;
        }
        ends.push(cursor as u32);
    }
    ends
}

fn storage_liveness(values: &[ValueInfo], tasks: &[Task], order: &[usize]) -> Vec<Option<Live>> {
    let mut live = vec![None::<Live>; values.len()];
    let mut readers = Vec::new();
    for (position, index) in order.iter().enumerate() {
        let task = &tasks[*index];
        let aliases = reads_every_element_in_place(values, task);
        touch(values, &mut live, task.out, position, false, None);
        readers.clear();
        for input in task.reads() {
            if readers.contains(&input) {
                continue;
            }
            readers.push(input);
            let in_place = aliases && input != task.out;
            touch(
                values,
                &mut live,
                input,
                position,
                true,
                in_place.then_some(position),
            );
        }
    }
    live
}

fn reads_every_element_in_place(values: &[ValueInfo], task: &Task) -> bool {
    if !task.kind.pointwise() {
        return false;
    }
    let out = &values[task.out as usize];
    task.reads().all(|value| {
        let value = &values[value as usize];
        value.shape == out.shape && value.strides == out.strides
    })
}

fn touch(
    values: &[ValueInfo],
    live: &mut [Option<Live>],
    value: u32,
    position: usize,
    read: bool,
    aliased_at: Option<usize>,
) {
    let storage = values[value as usize].storage as usize;
    match &mut live[storage] {
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
