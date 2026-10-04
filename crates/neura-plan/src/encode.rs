use crate::access::{self, Access, Reads};
use crate::authored;
use crate::fuse;
use crate::layout::{Layout, Region, store_of};
use crate::lower;
use crate::lower::Task;
use crate::product::Product;
use crate::schedule;
use crate::span::{Extents, Split};
use neura_abi::{
    Element, Geometry, Kind, NO_SLOT, NO_VALUE, Placement, SegmentRecord, StepRecord, Store,
    TaskFields, TaskRecord, ValueFields, ValueRecord, WORD_BYTES,
};
use neura_graph::{Graph, GraphSnapshot, Residency, Value, ValueInfo};
use neura_profile::{AttentionTile, MatmulTile, Profile};
use std::mem::size_of;

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Quantum {
    pub offset: u64,
    pub scale: f32,
}

pub struct Encoding {
    pub values: Vec<u8>,
    pub tasks: Vec<u8>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Span {
    pub store: Store,
    pub offset: u64,
    pub elements: u32,
    pub element: Element,
    pub scale: f32,
    table: u64,
}

impl Span {
    pub fn payload_bytes(&self) -> u64 {
        self.element.payload_words(u64::from(self.elements)) * WORD_BYTES
    }

    pub fn table_bytes(&self) -> u64 {
        self.element.quanta(u64::from(self.elements)) * WORD_BYTES
    }

    pub fn table_offset(&self) -> u64 {
        self.table * WORD_BYTES
    }

    pub fn image_bytes(&self) -> u64 {
        self.payload_bytes() + self.table_bytes()
    }
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
    scale: f32,
    table: u64,
}

pub struct Plan {
    profile: Profile,
    kinds: Vec<Kind>,
    elements: Vec<Element>,
    tasks: Vec<u8>,
    values: Vec<u8>,
    steps: Vec<u8>,
    segments: Vec<SegmentRecord>,
    wave_tasks: Vec<u32>,
    spans: Vec<Option<Placed>>,
    readable: Vec<bool>,
    geometries: Vec<u32>,
    products: Vec<Product>,
    attention: Vec<AttentionTile>,
    arena_bytes: u64,
    tensor_bytes: u64,
    quanta: Vec<Quantum>,
    layout: Layout,
    updates_weights: bool,
    work: u64,
    extents: Extents,
    splits: Vec<Split>,
    order: Vec<u32>,
    slot_bounds: Vec<u32>,
    authored: authored::Authored,
}

impl Plan {
    pub fn of(graph: &Graph<'_>, alignment: u64, profile: Profile) -> Self {
        Self::chosen(graph, alignment, profile, &[])
    }

    pub fn chosen(
        graph: &Graph<'_>,
        alignment: u64,
        profile: Profile,
        chosen: &[(Product, MatmulTile)],
    ) -> Self {
        Self::compile(&graph.snapshot(), profile, alignment, chosen)
    }

    fn compile(
        state: &GraphSnapshot,
        profile: Profile,
        alignment: u64,
        chosen: &[(Product, MatmulTile)],
    ) -> Self {
        assert!(
            alignment.is_power_of_two() && alignment >= 4,
            "an arena alignment of {alignment} bytes is not usable",
        );
        let lowered = lower::lower(state.values(), &fuse::fuse(state), profile, chosen);
        let lower::Plan {
            values,
            mut tasks,
            tiles: menu,
            products,
            attention,
            measures,
        } = lowered;
        let values = &values;
        let matmul_tiles = profile.tiles();

        let kinds = carried_kinds(&tasks);
        let elements = carried_elements(values);
        let layout = Layout::of_values(values, alignment);
        assert_authored_extents_cut_one_walk(values, state.authored());
        assert_writes_match_their_element(values, &tasks);
        assert_quantized_scales_reconstruct(values);
        assert_writers_precede_readers(values, &tasks);
        assert_units_keep_their_order(&tasks);
        assert_prefix_tables_close_their_walk(values, &tasks);
        let mut authored =
            authored::analyse(state.authored(), values, &mut tasks, &measures, &menu);
        let schedule = schedule::Schedule::of(values, &menu, &tasks, profile.workgroups());
        authored::plan_patches(
            &mut authored,
            values,
            &mut tasks,
            schedule.order(),
            &measures,
            &menu,
        );
        let order = schedule.order();
        let waves = schedule.waves();
        let live = storage_liveness(values, &tasks, order);
        let reserved = layout.tensors().bytes();
        let (offsets, tensor_bytes) = allocate(values, &live, waves, alignment, reserved);
        let arena_bytes = tensor_bytes - reserved;
        assert!(
            tensor_bytes.is_multiple_of(WORD_BYTES),
            "a plan of {tensor_bytes} bytes leaves the word grid the device indexes",
        );

        let mut records = Vec::new();
        for (id, info) in values.iter().enumerate() {
            let address = layout.address(values, &offsets, id as u32);
            let mut free = [NO_SLOT; 4];
            for axis in 0..neura_abi::MAX_RANK {
                if let Some(slot) = info.shape.free(axis) {
                    free[axis as usize] = slot;
                }
            }
            let source = match info.strides_source {
                Some(source) => source.map(u32::from),
                None => [NO_SLOT; 4],
            };
            let record = ValueRecord::of(ValueFields {
                base: u32::try_from(address).unwrap_or_else(|_| {
                    panic!("value {id} lies at {address}, beyond the device address space")
                }),
                store: layout.store(values, id as u32).code(),
                element: layout.element(values, id as u32).code(),
                table: record_table(values, id as u32),
                storage: info.storage,
                bounds: info.shape.dims(),
                free,
                source,
                dims: info.shape.dims(),
                strides: info.strides,
            });
            records.extend_from_slice(bytemuck::bytes_of(&record));
        }

        let mut task_bytes = Vec::with_capacity(tasks.len() * size_of::<TaskRecord>());
        let mut steps = Vec::new();
        let mut updates_weights = false;
        let mut geometries = vec![0u32; matmul_tiles.len()];
        let mut work = 0;
        for (position, index) in order.iter().enumerate() {
            let task = &tasks[*index as usize];
            let geometry = match task.kind.geometry() {
                Geometry::Attention => {
                    assert!(
                        (task.geometry as usize) < attention.len(),
                        "an attention names geometry {} beyond the {} tiles its plan carries",
                        task.geometry,
                        attention.len(),
                    );
                    task.geometry
                }
                Geometry::Product => {
                    assert!(
                        (task.geometry as usize) < matmul_tiles.len(),
                        "a product names geometry {} beyond the {} tiles its profile carries",
                        task.geometry,
                        matmul_tiles.len(),
                    );
                    geometries[task.geometry as usize] += 1;
                    task.geometry
                }
                Geometry::Strategy => task.geometry,
                Geometry::Access => {
                    assert!(
                        task.geometry == neura_abi::strategy::FRAME
                            || task.geometry == neura_abi::strategy::INDEX,
                        "a {} task walks its reads by the frame or by the plan index, not by geometry {}",
                        task.kind.name(),
                        task.geometry,
                    );
                    task.geometry
                }
                Geometry::None => {
                    assert_eq!(
                        task.geometry,
                        0,
                        "a {} task carries geometry {} where its vocabulary declares none",
                        task.kind.name(),
                        task.geometry,
                    );
                    0
                }
            };
            assert!(
                task.chain.is_empty() || task.kind.takes_chain(),
                "a {} task carries a chain no device body of it reads",
                task.kind.name(),
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
            assert!(
                task.origin == NO_VALUE || task.kind.reads_origin(),
                "a {} task carries a cursor no device body of it reads",
                task.kind.name(),
            );
            let prelude = (steps.len() / size_of::<StepRecord>()) as u32;
            for step in &task.prelude {
                steps.extend_from_slice(bytemuck::bytes_of(step));
            }
            let chain = (steps.len() / size_of::<StepRecord>()) as u32;
            for step in &task.chain {
                steps.extend_from_slice(bytemuck::bytes_of(step));
            }
            let (split_kind, split_measure, index, group, planes) = match task.split {
                Split::Range { .. } => (neura_abi::split::RANGE, NO_VALUE, 0, 0, 0),
                Split::Uniform {
                    measure,
                    index,
                    group,
                } => (neura_abi::split::UNIFORM, measure, index, group, 0),
                Split::Plane {
                    measure,
                    planes,
                    index,
                    group,
                    ..
                } => (neura_abi::split::PLANE, measure, index, group, planes),
                Split::Segment {
                    measure,
                    index,
                    group,
                    ..
                } => {
                    assert_eq!(
                        (task.first, task.count),
                        (index, 1),
                        "a segment task walks the one tile its plan index names",
                    );
                    (neura_abi::split::SEGMENT, measure, index, group, 0)
                }
            };
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
                origin: task.origin,
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
                axis: task.axis,
                offset: task.offset,
                wave: waves[position],
                split: split_kind,
                measure: split_measure,
                index,
                group,
                planes,
                plane: task.plane,
                patch: task.patch,
                segment: task.segments,
                keys: task.keys,
            });
            if task.in_place
                && task
                    .writes()
                    .any(|out| store_of(info_of(values, out).residency) == Store::Weights)
            {
                updates_weights = true;
            }
            work += task.work;
            task_bytes.extend_from_slice(bytemuck::bytes_of(&record));
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
            if !addressed_as_its_storage(values, id) {
                continue;
            }
            let storage = info.storage as usize;
            readable[id] = match values[storage].residency {
                Residency::Parameter | Residency::State | Residency::Resident => true,
                _ => match last_writer.get(&offsets[storage]) {
                    Some(writer) => *writer == storage as u32,
                    None => held(values, storage),
                },
            };
        }
        let mut spans = vec![None; values.len()];
        for (id, info) in values.iter().enumerate() {
            if !addressed_as_its_storage(values, id) {
                continue;
            }
            spans[id] = Some(Placed {
                store: layout.store(values, id as u32),
                address: layout.address(values, &offsets, id as u32),
                elements: info.shape.elements(),
                element: layout.element(values, id as u32),
                scale: layout.scale(values, id as u32),
                table: table_of(values, id as u32),
            });
        }

        let splits = tasks.iter().map(|task| task.split).collect::<Vec<_>>();
        let order = order.to_vec();
        let extents = Extents::of(values, &menu, &measures);
        let slot_bounds = {
            let mut bounds = Vec::new();
            for info in values {
                for axis in 0..neura_abi::MAX_RANK {
                    if let Some(slot) = info.shape.free(axis) {
                        let bound = info.shape.dims()[axis as usize];
                        if bounds.len() <= slot as usize {
                            bounds.resize(slot as usize + 1, 0);
                        }
                        bounds[slot as usize] = bound;
                    }
                }
            }
            bounds
        };
        Self {
            profile,
            kinds,
            elements,
            tasks: task_bytes,
            values: records,
            steps,
            segments,
            wave_tasks: schedule.wave_tasks(),
            spans,
            readable,
            geometries,
            products: products.clone(),
            attention: attention.clone(),
            arena_bytes,
            tensor_bytes,
            quanta: quanta(values, &offsets),
            layout,
            updates_weights,
            work,
            extents,
            splits,
            order,
            slot_bounds,
            authored,
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

    pub fn products(&self) -> &[Product] {
        &self.products
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

    pub fn dynamic(&self) -> bool {
        !self.slot_bounds.is_empty()
    }

    pub fn slot_bounds(&self) -> &[u32] {
        &self.slot_bounds
    }

    pub fn carries_authored(&self) -> bool {
        self.authored.carries()
    }

    pub fn authored_slots(&self) -> &[u32] {
        self.authored.slots()
    }

    pub fn host_slots(&self) -> Vec<(u32, u32)> {
        self.slot_bounds
            .iter()
            .enumerate()
            .filter(|(slot, _)| {
                self.authored
                    .slots()
                    .get(*slot)
                    .copied()
                    .unwrap_or(NO_VALUE)
                    == NO_VALUE
            })
            .map(|(slot, bound)| (slot as u32, *bound))
            .collect()
    }

    pub fn host_extents(&self) -> Vec<u32> {
        self.slot_bounds.clone()
    }

    pub fn measures(&self) -> &[neura_abi::MeasureRecord] {
        self.authored.measures()
    }

    pub fn patches(&self) -> &[neura_abi::PatchRecord] {
        self.authored.patches()
    }

    pub fn patch_list(&self) -> &[u32] {
        self.authored.patch_list()
    }

    pub fn authored_values(&self, value: u32) -> &[u32] {
        self.authored.values_of(value)
    }

    pub fn encode(&self, extents: &[u32]) -> Encoding {
        assert!(
            self.dynamic(),
            "a plan of one shape carries the records of every run it serves",
        );
        let mut values = self.values.clone();
        for id in 0..self.spans.len() {
            let dims = self.extents.dims(id as u32, extents);
            let strides = self.extents.strides(id as u32, extents);
            let at = id * size_of::<ValueRecord>();
            let mut record: ValueRecord =
                bytemuck::pod_read_unaligned(&values[at..at + size_of::<ValueRecord>()]);
            record.dims = dims;
            record.strides = strides;
            values[at..at + size_of::<ValueRecord>()].copy_from_slice(bytemuck::bytes_of(&record));
        }
        let mut tasks = self.tasks.clone();
        for (position, index) in self.order.iter().enumerate() {
            let (first, count) = self.extents.span(self.splits[*index as usize], extents);
            let at = position * size_of::<TaskRecord>();
            let mut record: TaskRecord =
                bytemuck::pod_read_unaligned(&tasks[at..at + size_of::<TaskRecord>()]);
            record.first = first;
            record.count = count;
            tasks[at..at + size_of::<TaskRecord>()].copy_from_slice(bytemuck::bytes_of(&record));
        }
        Encoding { values, tasks }
    }

    pub fn span_at(&self, value: Value<'_>, placement: Placement, extents: &[u32]) -> Span {
        let mut span = self.span(value, placement);
        span.elements = self.extents.dims(value.id(), extents).iter().product();
        assert!(
            span.table_offset() >= span.payload_bytes(),
            "a binding of {} numbers walks {} bytes of payload past the quantum table its bound places at {}",
            span.elements,
            span.payload_bytes(),
            span.table_offset(),
        );
        span
    }

    pub fn values(&self) -> &[u8] {
        &self.values
    }

    pub fn steps(&self) -> &[u8] {
        &self.steps
    }

    pub fn segments(&self) -> &[SegmentRecord] {
        &self.segments
    }

    pub fn wave_tasks(&self) -> &[u32] {
        &self.wave_tasks
    }

    pub fn wave_count(&self) -> u32 {
        self.wave_tasks.len() as u32
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
            scale: placed.scale,
            table: placed.table,
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

    pub fn quanta(&self) -> &[Quantum] {
        &self.quanta
    }

    pub fn weights(&self) -> &Region {
        self.layout.weights()
    }

    pub fn state(&self) -> &Region {
        self.layout.state()
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

fn assert_authored_extents_cut_one_walk(values: &[ValueInfo], authored: &[u32]) {
    for (id, info) in values.iter().enumerate() {
        let dims = info.shape.dims();
        for axis in 0..neura_abi::MAX_RANK {
            let Some(slot) = info.shape.free(axis) else {
                continue;
            };
            if authored.get(slot as usize).copied().unwrap_or(NO_VALUE) == NO_VALUE {
                continue;
            }
            let beside = dims[..axis as usize].iter().product::<u32>();
            assert!(
                beside == 1,
                "value {id} walks axis {axis} of {dims:?} through the extent the device authors at slot {slot}, and the {beside} planes beside it hold every stride the cut moves; a count cuts one plane, and every plane of a batch walks the offsets a ragged axis closes",
            );
        }
    }
}

fn assert_writes_match_their_element(values: &[ValueInfo], tasks: &[Task]) {
    for task in tasks {
        for out in task.writes() {
            let out = &values[out as usize];
            assert!(
                !out.element.per_block(),
                "a {} task writes the block quantized tensor {}; a {} tensor reconstructs through the quantum its storage holds of every {} blocks, and only its host holds those",
                task.kind.name(),
                task.out,
                out.element.name(),
                out.element.block(),
            );
            if task.kind == Kind::Convert {
                assert!(
                    out.element.narrow(),
                    "a convert writes the {} tensor {} word by word, and a word holds one element",
                    out.element.name(),
                    task.out,
                );
                for source in task
                    .inputs
                    .iter()
                    .take(1)
                    .chain(task.prelude.iter().map(|step| &step.operand))
                    .chain(task.chain.iter().map(|step| &step.operand))
                    .filter(|source| **source != NO_VALUE)
                    .copied()
                {
                    assert!(
                        values[source as usize].shape.fits_within(out.shape),
                        "a convert reads {:?} through the {:?} it writes",
                        values[source as usize].shape.dims(),
                        out.shape.dims(),
                    );
                }
                continue;
            }
            assert!(
                !out.element.narrow(),
                "a {} task writes the {} tensor {} element by element, and a narrow tensor is written word by word by a convert",
                task.kind.name(),
                out.element.name(),
                task.out,
            );
        }
    }
}

fn assert_quantized_scales_reconstruct(values: &[ValueInfo]) {
    for (id, info) in values.iter().enumerate() {
        if !info.element.per_tensor() {
            continue;
        }
        assert!(
            info.scale.is_finite() && info.scale > 0.0,
            "value {id} quantum of scale {} reconstructs nothing",
            info.scale,
        );
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
                    "task {position} reads a tensor that its own plan only writes later",
                ),
                None => assert!(
                    held(values, *storage as usize),
                    "task {position} reads a tensor no task of the plan writes before it",
                ),
            }
        }
        for storage in access.writes() {
            last_writer[*storage as usize] = Some(position);
        }
    }
}

fn assert_prefix_tables_close_their_walk(values: &[ValueInfo], tasks: &[Task]) {
    for task in tasks {
        if task.kind != Kind::PrefixClose {
            continue;
        }
        let table = values[task.inputs[0] as usize].shape.elements();
        let walked = values[task.inputs[1] as usize].shape.elements();
        assert!(
            table == walked || table == walked + 1,
            "a prefix holds {table} offsets of the {walked} numbers it sums; a table of one entry past that walk carries the total that closes it, and a table of one entry per number carries the offsets alone",
        );
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
        value.shape == out.shape && value.strides == out.strides && value.element == out.element
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
    let info = &values[storage];
    info.element.storage_words(u64::from(info.shape.elements())) * WORD_BYTES
}

fn table_of(values: &[ValueInfo], value: u32) -> u64 {
    let owner = &values[values[value as usize].storage as usize];
    owner
        .element
        .payload_words(u64::from(owner.shape.elements()))
}

fn record_table(values: &[ValueInfo], value: u32) -> u32 {
    if !values[value as usize].element.quantized() {
        return NO_VALUE;
    }
    u32::try_from(table_of(values, value)).unwrap_or_else(|_| {
        panic!("the quantum table of value {value} lies beyond the device address space")
    })
}

fn quanta(values: &[ValueInfo], offsets: &[u64]) -> Vec<Quantum> {
    values
        .iter()
        .enumerate()
        .filter(|(id, info)| {
            info.storage as usize == *id && arena_resident(values, *id) && info.element.quantized()
        })
        .map(|(id, info)| Quantum {
            offset: offsets[id] + table_of(values, id as u32) * WORD_BYTES,
            scale: info.scale,
        })
        .collect()
}

fn info_of(values: &[ValueInfo], value: u32) -> &ValueInfo {
    &values[values[value as usize].storage as usize]
}

fn addressed_as_its_storage(values: &[ValueInfo], id: usize) -> bool {
    let info = &values[id];
    let storage = &values[info.storage as usize];
    info.shape.elements() == storage.shape.elements() && info.strides == info.shape.strides()
}

fn owns_its_quanta(values: &[ValueInfo], storage: usize) -> bool {
    values[storage].element.quantized()
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
        Residency::Input | Residency::Parameter | Residency::State | Residency::Resident
    )
}

fn allocate(
    values: &[ValueInfo],
    live: &[Option<Live>],
    waves: &[u32],
    alignment: u64,
    reserved: u64,
) -> (Vec<u64>, u64) {
    let wave_of = |position: usize| waves[position];
    let mut arena = Blocks::with_base(reserved);
    let mut offsets = vec![0u64; live.len()];
    let owners = (0..values.len()).filter(|id| values[*id].storage as usize == *id);
    for id in owners.clone() {
        if !arena_resident(values, id) {
            continue;
        }
        if held(values, id) || values[id].retained || owns_its_quanta(values, id) {
            offsets[id] = arena.reserve(storage_bytes(values, id), alignment);
        }
    }
    let mut pending = live
        .iter()
        .enumerate()
        .filter(|(id, _)| !held(values, *id) && !values[*id].retained)
        .filter(|(id, _)| !owns_its_quanta(values, *id))
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
