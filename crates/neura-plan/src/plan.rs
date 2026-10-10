use crate::access::{Access, Reads};
use crate::authored;
use crate::fuse;
use crate::layout::{Arena, Layout, store_of};
use crate::lower;
use crate::lower::Task;
use crate::pages::{self, WeightPages};
use crate::product::Product;
use crate::record::{self, Recorded};
use crate::region::{self, Resolved, TableRows, Touches, Walk};
use crate::remembered::Remembered;
use crate::schedule;
use crate::span::{self, Extents, Split};
use neura_abi::{
    Element, Geometry, Kind, MAX_RANK, NO_SLOT, NO_VALUE, Placement, SegmentRecord, StepRecord,
    Store, TaskFields, TaskRecord, ValueFields, ValueRecord, WORD_BYTES,
};
use neura_graph::{Graph, GraphSnapshot, Residency, Shape, Value, ValueInfo};
use neura_profile::{AttentionTile, MatmulTile, Profile};
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Quantum {
    pub offset: u64,
    pub scale: f32,
}

pub struct Encoding {
    values: Vec<u8>,
    tasks: Vec<u8>,
    spans: Vec<Option<Placed>>,
    readable: Vec<bool>,
    quanta: Vec<Quantum>,
    lengths: Vec<u32>,
    arena_bytes: u64,
    tensor_bytes: u64,
    order: Vec<u32>,
    waves: Vec<u32>,
    segments: Vec<SegmentRecord>,
    wave_tasks: Vec<u32>,
    work: u64,
    patches: Vec<neura_abi::PatchRecord>,
    patch_list: Vec<u32>,
    held: Vec<bool>,
}

impl Encoding {
    pub fn bytes(&self) -> u64 {
        self.values.len() as u64
            + self.tasks.len() as u64
            + self.spans.len() as u64 * size_of::<Option<Placed>>() as u64
            + self.readable.len() as u64
            + self.quanta.len() as u64 * size_of::<Quantum>() as u64
            + self.lengths.len() as u64 * size_of::<u32>() as u64
            + self.order.len() as u64 * size_of::<u32>() as u64
            + self.waves.len() as u64 * size_of::<u32>() as u64
            + self.segments.len() as u64 * size_of::<SegmentRecord>() as u64
            + self.wave_tasks.len() as u64 * size_of::<u32>() as u64
            + self.patches.len() as u64 * size_of::<neura_abi::PatchRecord>() as u64
            + self.patch_list.len() as u64 * size_of::<u32>() as u64
            + self.held.len() as u64
    }

    fn empty() -> Self {
        Self {
            values: Vec::new(),
            tasks: Vec::new(),
            spans: Vec::new(),
            readable: Vec::new(),
            quanta: Vec::new(),
            lengths: Vec::new(),
            arena_bytes: 0,
            tensor_bytes: 0,
            order: Vec::new(),
            waves: Vec::new(),
            segments: Vec::new(),
            wave_tasks: Vec::new(),
            work: 0,
            patches: Vec::new(),
            patch_list: Vec::new(),
            held: Vec::new(),
        }
    }

    pub fn values(&self) -> &[u8] {
        &self.values
    }

    pub fn tasks(&self) -> &[u8] {
        &self.tasks
    }

    pub fn quanta(&self) -> &[Quantum] {
        &self.quanta
    }

    pub fn lengths(&self) -> &[u32] {
        &self.lengths
    }

    pub fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }

    pub fn tensor_bytes(&self) -> u64 {
        self.tensor_bytes
    }

    pub fn readable(&self, value: u32) -> bool {
        self.readable.get(value as usize).copied().unwrap_or(false)
    }

    pub fn order(&self) -> &[u32] {
        &self.order
    }

    pub fn waves(&self) -> &[u32] {
        &self.waves
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

    pub fn task_count(&self) -> u32 {
        self.order.len() as u32
    }

    pub fn work(&self) -> u64 {
        self.work
    }

    pub fn patches(&self) -> &[neura_abi::PatchRecord] {
        &self.patches
    }

    pub fn patch_list(&self) -> &[u32] {
        &self.patch_list
    }

    pub fn header(&self) -> Vec<u8> {
        neura_abi::control::header(self.segments.len() as u32, self.wave_count())
    }

    pub fn holds(&self, storage: u32) -> bool {
        self.held.get(storage as usize).copied().unwrap_or(false)
    }

    pub(crate) fn placed(&self, value: u32) -> Option<Placed> {
        self.spans.get(value as usize).copied().flatten()
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Span {
    pub store: Store,
    pub offset: u64,
    pub word: u64,
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
pub(crate) struct Placed {
    store: Store,
    address: u64,
    element: Element,
    scale: f32,
    table: u64,
}

pub struct Plan {
    profile: Profile,
    kinds: Vec<Kind>,
    elements: Vec<Element>,
    tasks: Vec<u8>,
    steps: Vec<u8>,
    geometries: Vec<u32>,
    products: Vec<Product>,
    attention: Vec<AttentionTile>,
    layout: Layout,
    updates_weights: bool,
    work: u64,
    extents: Extents,
    splits: Vec<Split>,
    works: Vec<u64>,
    slot_bounds: Vec<u32>,
    depends: Vec<u32>,
    depends_at: Vec<u32>,
    authored: authored::Authored,
    patches: authored::Patches,
    declared_waves: Vec<u32>,
    gather_tables: Vec<u32>,
    shapes: Arc<Vec<ValueInfo>>,
    alignment: u64,
    bound: Arc<Encoding>,
    remembered: Mutex<Remembered>,
    derived: AtomicU64,
}

impl Plan {
    pub fn of(graph: &Graph<'_>, alignment: u64, profile: Profile, encoding_bytes: u64) -> Self {
        Self::chosen(graph, alignment, profile, &[], encoding_bytes, 0)
    }

    pub fn chosen(
        graph: &Graph<'_>,
        alignment: u64,
        profile: Profile,
        chosen: &[(Product, MatmulTile)],
        encoding_bytes: u64,
        weight_slots: u32,
    ) -> Self {
        Self::compile(
            &graph.snapshot(),
            profile,
            alignment,
            chosen,
            encoding_bytes,
            weight_slots,
        )
    }

    fn compile(
        state: &GraphSnapshot,
        profile: Profile,
        alignment: u64,
        chosen: &[(Product, MatmulTile)],
        encoding_bytes: u64,
        weight_slots: u32,
    ) -> Self {
        assert!(
            alignment.is_power_of_two() && alignment >= 4,
            "an arena alignment of {alignment} bytes is not usable",
        );
        let fused = fuse::fuse(state);
        let authored_slots = state.authored().to_vec();
        let lowered = lower::lower(state.values(), &fused, profile, chosen, weight_slots);
        let lower::Plan {
            values,
            mut tasks,
            tiles: menu,
            products,
            attention,
            measures,
        } = lowered;
        let shapes = Arc::new(values);
        let values: &[ValueInfo] = &shapes;
        let matmul_tiles = profile.tiles();

        let kinds = carried_kinds(&tasks);
        let elements = carried_elements(values);
        let layout = Layout::of_values(values, alignment);
        assert_authored_extents_cut_one_walk(values, &authored_slots);
        assert_writes_match_their_element(values, &tasks);
        assert_quantized_scales_reconstruct(values);
        assert_writers_precede_readers(values, &tasks);
        assert_units_keep_their_order(&tasks);
        assert_ragged_chunks_cover_their_plane(&tasks);
        assert_a_segmented_attention_walks_the_planes_its_query_holds(values, &tasks);
        assert_a_per_plane_sum_walks_the_planes_its_offsets_close(values, &tasks);
        assert_prefix_tables_close_their_walk(values, &tasks);
        assert_a_task_needs_exact_lengths_the_plan_froze(values, &tasks);
        let authored = authored::analyse(&authored_slots, values, &mut tasks, &measures, &menu);
        let patches = authored::patches(&authored, values, &tasks);
        assert_a_packed_rope_turns_the_rows_of_one_plane_at_a_time(
            values,
            &tasks,
            &ragged_axes(&patches),
        );
        let authored_walks: Vec<bool> = tasks.iter().map(|task| !task.depends.is_empty()).collect();
        assert_authored_walks_read_the_counts_that_rule_them(
            values.len(),
            &patches,
            &tasks,
            &authored_walks,
        );
        let mut task_bytes = Vec::with_capacity(tasks.len() * size_of::<TaskRecord>());
        let mut steps = Vec::new();
        let mut updates_weights = false;
        let mut geometries = vec![0u32; matmul_tiles.len()];
        let mut work = 0;
        let mut splits = Vec::with_capacity(tasks.len());
        let mut works = Vec::with_capacity(tasks.len());
        for task in &tasks {
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
                Split::Ragged {
                    planes,
                    plane,
                    index,
                    group,
                } => {
                    assert_eq!(
                        (task.first, task.count, task.plane),
                        (0, 0, plane),
                        "a ragged walk of chunk {index} of {group} of plane {plane} takes the rows the patch that closes the axis writes, and carries the plane it walks",
                    );
                    (neura_abi::split::RAGGED, NO_VALUE, index, group, planes)
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
                literal: task.literal,
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
                in_place: u32::from(task.in_place),
                wave: 0,
                split: split_kind,
                measure: split_measure,
                index,
                group,
                planes,
                plane: task.plane,
                patch: NO_VALUE,
                segment: task.segments,
                keys: task.keys,
                reach: task.reach,
                queries: task.queries,
                tokens: task.tokens,
                grid: task.grid,
                planned_first: task.first,
                planned_count: task.count,
                knob: task.knob,
                keep: task.keep,
            });
            if task.in_place
                && task
                    .writes()
                    .any(|out| store_of(info_of(values, out).residency) == Store::Weights)
            {
                updates_weights = true;
            }
            work += task.work;
            splits.push(task.split);
            works.push(task.work);
            task_bytes.extend_from_slice(bytemuck::bytes_of(&record));
        }

        let gather_tables = {
            let mut tables = tasks
                .iter()
                .filter(|task| task.kind == Kind::Gather)
                .map(|task| task.inputs[0])
                .filter(|table| values[*table as usize].storage == *table)
                .collect::<Vec<u32>>();
            tables.sort_unstable();
            tables.dedup();
            tables
        };
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
        let mut depends = Vec::new();
        let mut depends_at = Vec::with_capacity(tasks.len() + 1);
        for task in &tasks {
            depends_at.push(depends.len() as u32);
            depends.extend_from_slice(&task.depends);
        }
        depends_at.push(depends.len() as u32);
        let mut plan = Self {
            profile,
            kinds,
            elements,
            tasks: task_bytes,
            steps,
            geometries,
            products,
            attention,
            layout,
            updates_weights,
            work,
            extents,
            splits,
            works,
            slot_bounds,
            depends,
            depends_at,
            authored,
            patches,
            declared_waves: Vec::new(),
            gather_tables,
            shapes,
            alignment,
            bound: Arc::new(Encoding::empty()),
            remembered: Mutex::new(Remembered::of(encoding_bytes)),
            derived: AtomicU64::new(0),
        };
        let bound = plan.deliver(&plan.slot_bounds);
        plan.declared_waves = declared_waves(&bound);
        plan.bound = Arc::new(bound);
        for (task, owner) in tasks.iter_mut().zip(&plan.patches.owners) {
            task.patch = *owner;
        }
        record::assert_records_hold_the_tasks_they_carry(
            &tasks,
            plan.bound.order(),
            plan.bound.tasks(),
            &plan.steps,
        );
        plan
    }

    pub fn profile(&self) -> Profile {
        self.profile
    }

    pub fn gather_tables(&self) -> &[u32] {
        &self.gather_tables
    }

    pub fn table_rows(&self, value: u32) -> u32 {
        assert!(
            self.gather_tables.binary_search(&value).is_ok(),
            "value {value} is no table this plan gathers rows of, and a host names the rows of a table a task walks",
        );
        let offset = value as usize * size_of::<ValueRecord>();
        let record: ValueRecord = bytemuck::pod_read_unaligned(
            &self.bound.values()[offset..offset + size_of::<ValueRecord>()],
        );
        record.dims[0] * record.dims[1] * record.dims[2]
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

    pub fn awaits_a_binding(&self) -> bool {
        !self.host_slots().is_empty()
    }

    pub fn measures(&self) -> &[neura_abi::MeasureRecord] {
        self.authored.measures()
    }

    pub fn patches(&self) -> &[neura_abi::PatchRecord] {
        self.bound.patches()
    }

    pub fn patch_list(&self) -> &[u32] {
        self.bound.patch_list()
    }

    pub fn authored_values(&self, value: u32) -> &[u32] {
        self.authored.values_of(value)
    }

    pub fn encode(&self, lengths: &[u32]) -> Arc<Encoding> {
        assert_eq!(
            lengths.len(),
            self.slot_bounds.len(),
            "this plan walks {} free extents and the host bound it to {} lengths",
            self.slot_bounds.len(),
            lengths.len(),
        );
        if lengths == self.slot_bounds.as_slice() {
            return self.bound.clone();
        }
        if let Some(held) = self.store().find(lengths) {
            return held;
        }
        let derived = Arc::new(self.deliver(lengths));
        self.derived.fetch_add(1, Ordering::Relaxed);
        self.store().insert(lengths, derived)
    }

    fn store(&self) -> MutexGuard<'_, Remembered> {
        self.remembered
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }

    pub fn remembered_encodings(&self) -> usize {
        self.store().len()
    }

    pub fn remembered_bytes(&self) -> u64 {
        self.store().bytes()
    }

    pub fn derived_encodings(&self) -> u64 {
        self.derived.load(Ordering::Relaxed)
    }

    fn same_steps(&self, left: &Recorded<'_>, right: &Recorded<'_>) -> bool {
        let step = size_of::<StepRecord>();
        let slice = |at: u32, count: u32| {
            let first = at as usize * step;
            &self.steps[first..first + count as usize * step]
        };
        let (left, right) = (left.record(), right.record());
        slice(left.prelude, left.prelude_steps) == slice(right.prelude, right.prelude_steps)
            && slice(left.chain, left.steps) == slice(right.chain, right.steps)
    }

    fn depends_of(&self, index: usize) -> &[u32] {
        &self.depends[self.depends_at[index] as usize..self.depends_at[index + 1] as usize]
    }

    fn dispatches(
        &self,
        index: usize,
        split: Split,
        span: (u32, u32),
        recorded: &Recorded<'_>,
    ) -> bool {
        if span.1 > 0 || self.patches.owners[index] != NO_VALUE {
            return true;
        }
        if matches!(split, Split::Ragged { .. } | Split::Segment { .. }) {
            return true;
        }
        identity_kind(recorded.kind())
    }

    fn emit<'a>(
        &'a self,
        lengths: &[u32],
        decoded: &'a [Recorded<'a>],
        start: usize,
        end: usize,
        at_bound: bool,
        emitted: &mut Emitted<'a, 'a>,
    ) {
        let split = self.splits[start];
        let dependencies = self.depends_of(start);
        let regrouped = !at_bound
            && self.patches.owners[start] == NO_VALUE
            && dependencies.is_empty()
            && !matches!(split, Split::Ragged { .. } | Split::Segment { .. });
        match split {
            Split::Uniform { measure, group, .. }
                if regrouped && (end - start) as u32 == group.max(1) =>
            {
                let bound = self.extents.count(measure, &self.slot_bounds);
                let live = self.extents.count(measure, lengths);
                let per_task = bound.div_ceil(group).max(1);
                let pieces = live.div_ceil(per_task).max(1);
                let base = self.extents.span(split, &self.slot_bounds).1.max(1);
                let work = self.works[start];
                for index in 0..pieces {
                    let (first, count) = span::uniform(live, index, pieces);
                    if count == 0 && !identity_kind(decoded[start].kind()) {
                        continue;
                    }
                    let mut record = decoded[start].record();
                    record.index = index;
                    record.group = pieces;
                    record.first = first;
                    record.count = count;
                    emitted.walks.push(Walked {
                        recorded: Recorded::with(record, &self.steps),
                        depends: dependencies,
                        span: (first, count),
                        work: scaled(work, count, base),
                    });
                    emitted.sources.push(start as u32);
                    emitted
                        .declared
                        .push(self.declared_waves.get(start).copied().unwrap_or(0));
                }
            }
            Split::Plane {
                measure,
                plane,
                group,
                ..
            } if regrouped && (end - start) as u32 == group.max(1) => {
                let planes = self.extents.planes(measure, lengths);
                if plane >= planes {
                    return;
                }
                let bound = self.extents.count(measure, &self.slot_bounds);
                let live = self.extents.count(measure, lengths);
                let per_task = bound.div_ceil(group).max(1);
                let pieces = live.div_ceil(per_task).max(1);
                let base = self.extents.span(split, &self.slot_bounds).1.max(1);
                let work = self.works[start];
                for index in 0..pieces {
                    let (within, count) = span::uniform(live, index, pieces);
                    if count == 0 && !identity_kind(decoded[start].kind()) {
                        continue;
                    }
                    let first = plane * live + within;
                    let mut record = decoded[start].record();
                    record.index = index;
                    record.group = pieces;
                    record.first = first;
                    record.count = count;
                    emitted.walks.push(Walked {
                        recorded: Recorded::with(record, &self.steps),
                        depends: dependencies,
                        span: (first, count),
                        work: scaled(work, count, base),
                    });
                    emitted.sources.push(start as u32);
                    emitted
                        .declared
                        .push(self.declared_waves.get(start).copied().unwrap_or(0));
                }
            }
            _ => {
                for (offset, recorded) in decoded[start..end].iter().enumerate() {
                    let index = start + offset;
                    let span = self.extents.span(self.splits[index], lengths);
                    if !at_bound && !self.dispatches(index, self.splits[index], span, recorded) {
                        continue;
                    }
                    emitted.walks.push(Walked {
                        recorded: *recorded,
                        depends: self.depends_of(index),
                        span,
                        work: self.works[index],
                    });
                    emitted.sources.push(index as u32);
                    emitted
                        .declared
                        .push(self.declared_waves.get(index).copied().unwrap_or(0));
                }
            }
        }
    }

    pub(crate) fn deliver(&self, lengths: &[u32]) -> Encoding {
        let sized = Sized::of(&self.shapes, &self.extents, lengths, self.authored.slots());
        assert_eq!(
            self.depends_at.len(),
            self.splits.len() + 1,
            "a plan carries one dependency run per task it schedules and one more for the end",
        );
        let at_bound = lengths == self.slot_bounds.as_slice();
        let decoded = Recorded::of(&self.tasks, &self.steps).collect::<Vec<_>>();
        let mut emitted = Emitted {
            walks: Vec::new(),
            sources: Vec::new(),
            declared: Vec::new(),
        };
        let mut start = 0usize;
        while start < self.splits.len() {
            let key = family_key(decoded[start].record());
            let split = split_key(self.splits[start]);
            let mut end = start + 1;
            while end < self.splits.len()
                && split_key(self.splits[end]) == split
                && family_key(decoded[end].record()) == key
                && self.same_steps(&decoded[start], &decoded[end])
                && self.depends_of(end) == self.depends_of(start)
            {
                end += 1;
            }
            self.emit(lengths, &decoded, start, end, at_bound, &mut emitted);
            start = end;
        }
        let schedule = schedule::Schedule::of(
            &sized,
            self.profile.tiles(),
            &emitted.walks,
            self.profile.workgroups(),
            &emitted.declared,
        );
        let order = schedule
            .order()
            .iter()
            .map(|position| emitted.sources[*position as usize])
            .collect::<Vec<u32>>();
        let tables = authored::tables(&self.patches, &order);
        let mut task_bytes = Vec::with_capacity(schedule.order().len() * size_of::<TaskRecord>());
        let mut waves = Vec::with_capacity(schedule.order().len());
        for (position, index) in schedule.order().iter().enumerate() {
            let walked = &emitted.walks[*index as usize];
            let source = emitted.sources[*index as usize] as usize;
            let mut record = walked.recorded.record();
            if !matches!(walked.split(), Split::Range { .. }) {
                record.first = walked.span.0;
                record.count = walked.span.1;
            }
            record.wave = schedule.waves()[position];
            record.patch = self.patches.owners[source];
            record.planned_first = record.first;
            record.planned_count = record.count;
            task_bytes.extend_from_slice(bytemuck::bytes_of(&record));
            waves.push(schedule.waves()[position]);
        }
        let ordered = schedule
            .order()
            .iter()
            .map(|index| emitted.walks[*index as usize])
            .collect::<Vec<_>>();
        let live_values = storage_liveness(&sized, &ordered);
        let reserved = self.layout.tensors().bytes();
        let (offsets, tensor_bytes, placed) =
            allocate(&sized, &live_values, &waves, self.alignment, reserved);
        assert!(
            tensor_bytes.is_multiple_of(WORD_BYTES),
            "a plan of {tensor_bytes} bytes leaves the word grid the device indexes",
        );
        let mut records = Vec::with_capacity(sized.len() * size_of::<ValueRecord>());
        for id in 0..sized.len() {
            let id = id as u32;
            let owner = sized.storage(id);
            assert!(
                sized.element(id) == sized.element(owner) && sized.scale(id) == sized.scale(owner),
                "value {id} holds numbers of another storage",
            );
            let address = self.layout.address(owner, sized.residency(owner), &offsets);
            let mut free = [NO_SLOT; 4];
            for axis in 0..neura_abi::MAX_RANK {
                if let Some(slot) = self.shapes[id as usize].shape.free(axis) {
                    free[axis as usize] = slot;
                }
            }
            let source = match self.shapes[id as usize].strides_source {
                Some(source) => source.map(u32::from),
                None => [NO_SLOT; 4],
            };
            let record = ValueRecord::of(ValueFields {
                base: u32::try_from(address).unwrap_or_else(|_| {
                    panic!("value {id} lies at {address}, beyond the device address space")
                }),
                store: store_of(sized.residency(owner)).code(),
                element: sized.element(id).code(),
                table: record_table(&sized, id),
                storage: owner,
                bounds: self.shapes[id as usize].shape.dims(),
                free,
                source,
                dims: sized.dims(id),
                strides: sized.strides(id),
            });
            records.extend_from_slice(bytemuck::bytes_of(&record));
        }
        let mut readable = vec![false; sized.len()];
        let mut last_writer = std::collections::HashMap::<u64, u32>::new();
        for task in &ordered {
            for out in task.writes() {
                let storage = sized.storage(out);
                if !sized.resident(storage) || sized.bytes(storage) == 0 {
                    continue;
                }
                last_writer.insert(offsets[storage as usize], out);
            }
        }
        for id in 0..sized.len() {
            let id = id as u32;
            if !sized.addressed_as_its_storage(id) {
                continue;
            }
            let storage = sized.storage(id);
            readable[id as usize] = sized.held(storage)
                || sized.elements(storage) == 0
                || last_writer.get(&offsets[storage as usize]) == Some(&storage);
        }
        let mut spans = vec![None; sized.len()];
        for id in 0..sized.len() {
            let id = id as u32;
            if !sized.addressed_as_its_storage(id) {
                continue;
            }
            let owner = sized.storage(id);
            if !placed[owner as usize] && sized.bytes(owner) != 0 {
                continue;
            }
            spans[id as usize] = Some(Placed {
                store: store_of(sized.residency(owner)),
                address: self.layout.address(owner, sized.residency(owner), &offsets),
                element: sized.element(id),
                scale: sized.scale(id),
                table: sized.table(id),
            });
        }
        let work = ordered.iter().map(|task| task.work).sum();
        Encoding {
            values: records,
            tasks: task_bytes,
            spans,
            readable,
            quanta: quanta(&sized, &offsets),
            lengths: lengths.to_vec(),
            arena_bytes: tensor_bytes - reserved,
            tensor_bytes,
            order,
            waves,
            segments: schedule.segments().to_vec(),
            wave_tasks: schedule.wave_tasks(),
            work,
            patches: tables.patches,
            patch_list: tables.list,
            held: placed,
        }
    }

    pub fn bound_encoding(&self) -> Arc<Encoding> {
        self.bound.clone()
    }

    pub fn values(&self) -> &[u8] {
        self.bound.values()
    }

    pub fn steps(&self) -> &[u8] {
        &self.steps
    }

    pub fn segments(&self) -> &[SegmentRecord] {
        self.bound.segments()
    }

    pub fn wave_tasks(&self) -> &[u32] {
        self.bound.wave_tasks()
    }

    pub fn wave_count(&self) -> u32 {
        self.bound.wave_count()
    }

    pub fn tasks(&self) -> &[u8] {
        self.bound.tasks()
    }

    pub fn span(&self, value: Value<'_>, placement: Placement) -> Span {
        self.span_at(&self.bound, value, placement, &self.slot_bounds)
    }

    pub fn span_at(
        &self,
        encoding: &Encoding,
        value: Value<'_>,
        placement: Placement,
        lengths: &[u32],
    ) -> Span {
        let placed = encoding.placed(value.id()).unwrap_or_else(|| {
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
            word: placed.address,
            elements: self.extents.dims(value.id(), lengths).iter().product(),
            element: placed.element,
            scale: placed.scale,
            table: placed.table,
        }
    }

    pub fn readable(&self, value: Value<'_>) -> bool {
        self.bound.readable(value.id())
    }

    pub fn arena_bytes(&self) -> u64 {
        self.bound.arena_bytes()
    }

    pub fn quanta(&self) -> &[Quantum] {
        self.bound.quanta()
    }

    pub fn weights(&self) -> &Arena {
        self.layout.weights()
    }

    pub fn weight_pages(&self, encoding: &Encoding, rows: &[TableRows<'_>]) -> Vec<WeightPages> {
        let values = Resolved::of(
            encoding.values(),
            &self.extents,
            encoding.lengths(),
            self.authored.slots(),
        );
        let tiles = self.profile.tiles();
        let mut touched = Touches::default();
        Recorded::of(encoding.tasks(), &self.steps)
            .map(|task| {
                let walked = region::bounded(&values, &task);
                match walked {
                    true => region::walked(
                        &mut touched,
                        &task,
                        &values,
                        tiles,
                        task.walked(&self.extents, encoding.lengths()),
                        rows,
                    ),
                    false => region::whole(&mut touched, &task, &values),
                }
                pages::weight_pages(&touched, &values, &self.layout)
            })
            .collect()
    }

    pub fn weight_pages_at(&self, extents: &[u32], rows: &[TableRows<'_>]) -> Vec<WeightPages> {
        self.weight_pages(&self.encode(extents), rows)
    }

    pub fn store_words(&self) -> u64 {
        self.layout.words()
    }

    pub fn state(&self) -> &Arena {
        self.layout.state()
    }

    pub fn tensors(&self) -> &Arena {
        self.layout.tensors()
    }

    pub fn tensor_bytes(&self) -> u64 {
        self.bound.tensor_bytes()
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
        (self.bound.values().len() / size_of::<ValueRecord>()) as u32
    }

    pub fn work(&self) -> u64 {
        self.work
    }
}

struct Emitted<'a, 'b> {
    walks: Vec<Walked<'a, 'b>>,
    sources: Vec<u32>,
    declared: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Walked<'a, 'b> {
    recorded: Recorded<'a>,
    depends: &'b [u32],
    span: (u32, u32),
    work: u64,
}

impl Reads for Walked<'_, '_> {
    fn out(&self) -> u32 {
        self.recorded.out()
    }

    fn extra(&self) -> u32 {
        self.recorded.extra()
    }

    fn in_place(&self) -> bool {
        self.recorded.in_place()
    }

    fn reads(&self) -> impl Iterator<Item = u32> + '_ {
        self.recorded.reads().chain(self.depends.iter().copied())
    }
}

impl Walk for Walked<'_, '_> {
    fn kind(&self) -> Kind {
        self.recorded.kind()
    }

    fn geometry(&self) -> u32 {
        self.recorded.geometry()
    }

    fn splits(&self) -> u32 {
        self.recorded.splits()
    }

    fn slot(&self) -> u32 {
        self.recorded.slot()
    }

    fn input(&self, slot: usize) -> u32 {
        self.recorded.input(slot)
    }

    fn prelude(&self) -> impl Iterator<Item = u32> {
        self.recorded.prelude()
    }

    fn chain(&self) -> impl Iterator<Item = u32> {
        self.recorded.chain()
    }

    fn depends(&self) -> impl Iterator<Item = u32> {
        self.depends.iter().copied()
    }

    fn split(&self) -> Split {
        self.recorded.split()
    }
}

impl schedule::Scheduled for Walked<'_, '_> {
    fn span(&self) -> (u32, u32) {
        self.span
    }

    fn work(&self) -> u64 {
        self.work
    }
}

struct Sized<'a> {
    shapes: &'a [ValueInfo],
    extents: &'a Extents,
    authored: &'a [u32],
    dims: Vec<[u32; 4]>,
    strides: Vec<[u32; 4]>,
    elements: Vec<u64>,
}

impl<'a> Sized<'a> {
    fn of(
        shapes: &'a [ValueInfo],
        extents: &'a Extents,
        lengths: &[u32],
        authored: &'a [u32],
    ) -> Self {
        let dims = (0..shapes.len())
            .map(|id| extents.dims(id as u32, lengths))
            .collect::<Vec<_>>();
        let strides = (0..shapes.len())
            .map(|id| extents.strides(id as u32, lengths))
            .collect::<Vec<_>>();
        let elements = dims
            .iter()
            .map(|dims| dims.iter().map(|dim| u64::from(*dim)).product())
            .collect();
        Self {
            shapes,
            extents,
            authored,
            dims,
            strides,
            elements,
        }
    }

    fn len(&self) -> usize {
        self.shapes.len()
    }

    fn storage(&self, value: u32) -> u32 {
        self.shapes[value as usize].storage
    }

    fn residency(&self, storage: u32) -> Residency {
        self.shapes[storage as usize].residency
    }

    fn element(&self, value: u32) -> Element {
        self.shapes[value as usize].element
    }

    fn scale(&self, value: u32) -> f32 {
        self.shapes[value as usize].scale
    }

    fn retained(&self, value: u32) -> bool {
        self.shapes[value as usize].retained
    }

    fn elements(&self, value: u32) -> u64 {
        self.elements[value as usize]
    }

    fn dims(&self, value: u32) -> [u32; 4] {
        self.dims[value as usize]
    }

    fn strides(&self, value: u32) -> [u32; 4] {
        self.strides[value as usize]
    }

    fn bytes(&self, value: u32) -> u64 {
        self.element(value).storage_words(self.elements(value)) * WORD_BYTES
    }

    fn table(&self, value: u32) -> u64 {
        let owner = self.storage(value);
        self.element(owner).payload_words(self.elements(owner))
    }

    fn quantized(&self, value: u32) -> bool {
        self.element(value).quantized()
    }

    fn resident(&self, value: u32) -> bool {
        matches!(self.residency(value), Residency::Input | Residency::Derived)
    }

    fn held(&self, value: u32) -> bool {
        matches!(
            self.residency(value),
            Residency::Input | Residency::Parameter | Residency::State | Residency::Resident
        )
    }

    fn owns_its_quanta(&self, value: u32) -> bool {
        self.element(value).quantized()
    }

    fn addressed_as_its_storage(&self, value: u32) -> bool {
        let owner = self.storage(value);
        self.elements(value) == self.elements(owner)
            && self.strides(value) == Shape::dense_strides(self.dims(value))
    }

    fn matches(&self, left: u32, right: u32) -> bool {
        if self.dims(left) != self.dims(right)
            || self.strides(left) != self.strides(right)
            || self.element(left) != self.element(right)
        {
            return false;
        }
        let (left, right) = (
            self.shapes[left as usize].shape,
            self.shapes[right as usize].shape,
        );
        (0..neura_abi::MAX_RANK).all(|axis| left.free(axis) == right.free(axis))
    }
}

impl region::Values for Sized<'_> {
    fn len(&self) -> usize {
        Sized::len(self)
    }

    fn dims(&self, value: u32) -> [u32; 4] {
        Sized::dims(self, value)
    }

    fn strides(&self, value: u32) -> [u32; 4] {
        Sized::strides(self, value)
    }

    fn bounds(&self, value: u32) -> [u32; 4] {
        self.shapes[value as usize].shape.dims()
    }

    fn element(&self, value: u32) -> Element {
        Sized::element(self, value)
    }

    fn storage(&self, value: u32) -> u32 {
        Sized::storage(self, value)
    }

    fn recomputes(&self, value: u32) -> Option<u32> {
        self.shapes[value as usize].recomputes
    }

    fn exact(&self, value: u32) -> bool {
        self.extents.sealed(value, self.authored)
            && self
                .extents
                .sealed(Sized::storage(self, value), self.authored)
    }

    fn bounded(&self, value: u32) -> bool {
        self.strides(value) == Shape::dense_strides(self.dims(value))
            && self.extents.cut_is_outer(value, self.authored)
            && self.extents.strides_hold(value, self.authored)
    }

    fn strides_hold(&self, value: u32) -> bool {
        self.extents.strides_hold(value, self.authored)
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
        if info.strides != info.shape.strides() {
            continue;
        }
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
                    "task {position} of kind {} reads storage {storage} no task of the plan writes before it",
                    task.kind.name(),
                ),
            }
        }
        for storage in access.writes() {
            last_writer[*storage as usize] = Some(position);
        }
    }
}

fn assert_ragged_chunks_cover_their_plane(tasks: &[Task]) {
    let mut walked = std::collections::BTreeMap::<(u32, u32, u32), (u32, Vec<u32>)>::new();
    for task in tasks {
        let Split::Ragged {
            planes,
            plane,
            index,
            group,
        } = task.split
        else {
            assert_eq!(
                task.grid,
                NO_VALUE,
                "a {} task walks no rows a ragged axis packs, and it names the offsets value {} as the grid of its rows; a grid closes the rows of every plane of one axis",
                task.kind.name(),
                task.grid,
            );
            continue;
        };
        assert!(
            plane < planes && index < group,
            "a ragged task walks chunk {index} of {group} of plane {plane}, where the axis closes {planes} planes",
        );
        assert!(
            task.grid == task.segments || task.grid == task.queries,
            "a {} task walks the rows a ragged axis packs, and value {} closes the rows of no axis it weighs; a grid is the key offsets or the query offsets of the task",
            task.kind.name(),
            task.grid,
        );
        let (grouped, chunks) = walked
            .entry((task.grid, plane, task.out))
            .or_insert((group, Vec::new()));
        assert_eq!(
            *grouped, group,
            "plane {plane} of the ragged axis value {} walks {group} chunks, and the tasks that write value {} walk {grouped}; one output parts a plane one way",
            task.grid, task.out,
        );
        chunks.push(index);
    }
    for ((segment, plane, out), (group, mut chunks)) in walked {
        chunks.sort_unstable();
        assert_eq!(
            chunks,
            (0..group).collect::<Vec<u32>>(),
            "plane {plane} of the ragged axis value {segment} walks {group} chunks, and the tasks that write value {out} walk {chunks:?}; every chunk of a plane is walked exactly once",
        );
    }
}

fn ragged_axes(patches: &authored::Patches) -> Vec<(u32, u32)> {
    let mut axes = patches
        .counted
        .iter()
        .filter(|counted| counted.segment != NO_VALUE)
        .flat_map(|counted| counted.slots.iter().map(|slot| (*slot, counted.segment)))
        .collect::<Vec<(u32, u32)>>();
    axes.sort_unstable();
    axes.dedup();
    axes
}

pub(crate) fn assert_a_packed_rope_turns_the_rows_of_one_plane_at_a_time(
    values: &[ValueInfo],
    tasks: &[Task],
    ragged: &[(u32, u32)],
) {
    for task in tasks {
        if !matches!(task.kind, Kind::Rope | Kind::RopeGrad) {
            continue;
        }
        let shape = values[task.out as usize].shape;
        let walked = (0..MAX_RANK).find_map(|at| {
            let slot = shape.free(at)?;
            let offsets = ragged
                .iter()
                .find(|(kept, _)| *kept == slot)
                .map(|(_, offsets)| *offsets)?;
            Some((at, slot, offsets))
        });
        let packed = task.segments != NO_VALUE;
        if !packed {
            if let Some((at, _, offsets)) = walked {
                panic!(
                    "a {} task turns {:?} without the segments of a ragged axis, and axis {at} walks the extent the ragged axis value {offsets} packs; the device seats a packed row at the position its own plane's offsets name, so a rotation that names no segments turns every row from the plan's own range",
                    task.kind.name(),
                    shape.dims(),
                );
            }
            continue;
        }
        let Some((at, _, offsets)) = walked else {
            panic!(
                "a packed {} task walks the segments of value {} and turns {:?}, which walks the extent of no ragged axis",
                task.kind.name(),
                task.segments,
                shape.dims(),
            );
        };
        assert_eq!(
            at,
            2,
            "a packed {} task turns the rows of axis 2, and value {offsets} packs the rows of axis {at} of {:?}: a packed tensor lays the rows of every plane on axis 2 with axes 0 and 1 holding one plane, so that the row of a plane stands where the offsets of that plane reach",
            task.kind.name(),
            shape.dims(),
        );
        assert_eq!(
            matches!(task.split, Split::Ragged { .. }),
            packed,
            "a {} task turns the rows an extent packs beside the rows one plane holds, and a packed task walks the rows of one plane at a time; a rotation of a packed tensor gathers its rows from a ragged axis and no plan hands it another range",
            task.kind.name(),
        );
        let dims = values[task.out as usize].shape.dims();
        assert!(
            dims[0] == 1 && dims[1] == 1,
            "a packed {} task turns the rows of one plane at a time, and value {} holds {} planes beside the rows the axis value {} packs into axis 2",
            task.kind.name(),
            task.out,
            dims[0] * dims[1],
            task.segments,
        );
        assert_eq!(
            task.grid,
            task.segments,
            "a packed {} task walks the rows the axis value {} closes, and names value {} as the grid of its rows",
            task.kind.name(),
            task.segments,
            task.grid,
        );
        assert_eq!(
            task.queries,
            NO_VALUE,
            "a packed {} task turns the rows a key axis packs, and value {} closes the rows of a query chunk the device places at the end of the key plane its sequence already holds",
            task.kind.name(),
            task.queries,
        );
    }
}

fn assert_a_segmented_attention_walks_the_planes_its_query_holds(
    values: &[ValueInfo],
    tasks: &[Task],
) {
    for task in tasks {
        if task.segments == NO_VALUE
            || !matches!(
                task.kind,
                Kind::Attention
                    | Kind::AttentionQueryGrad
                    | Kind::AttentionKeyGrad
                    | Kind::AttentionValueGrad
            )
        {
            continue;
        }
        let query = values[task.inputs[0] as usize].shape;
        let key = values[task.inputs[1] as usize].shape;
        let packed =
            query.free(2).is_some() && (query.free(2) == key.free(2) || task.queries != NO_VALUE);
        if matches!(task.kind, Kind::Attention | Kind::AttentionQueryGrad) {
            assert_eq!(
                matches!(task.split, Split::Ragged { .. }),
                packed,
                "a segmented {} task weighs the queries of a plane against the keys its offsets close, and the query of {:?} walks free extent {:?} where those offsets pack free extent {:?}",
                task.kind.name(),
                query.dims(),
                query.free(2),
                key.free(2),
            );
        }
        let planes = if packed {
            values[task.grid as usize].shape.elements() - 1
        } else {
            query.dims()[0] * query.dims()[1]
        };
        assert!(
            task.plane < planes,
            "a segmented {} task walks plane {} of the ragged axis its keys pack, and the query it weighs holds {planes} planes",
            task.kind.name(),
            task.plane,
        );
    }
}

fn assert_a_per_plane_sum_walks_the_planes_its_offsets_close(values: &[ValueInfo], tasks: &[Task]) {
    for task in tasks {
        if task.kind != Kind::SegmentSum || task.segments == NO_VALUE {
            continue;
        }
        let planes = values[task.segments as usize].shape.elements() - 1;
        let partials = values[task.out as usize].shape.dims();
        assert_eq!(
            partials[0] * partials[1],
            planes,
            "a per-plane sum hands one number to each of the {planes} planes the offsets value {} closes, and value {} holds {} of them; the lengths of a ragged axis lay their planes out in one order, and the sum carries that order into the first two axes of its result",
            task.segments,
            task.out,
            partials[0] * partials[1],
        );
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

pub(crate) fn assert_a_task_needs_exact_lengths_the_plan_froze(
    values: &[ValueInfo],
    tasks: &[Task],
) {
    for task in tasks {
        match task.kind {
            Kind::Concat => {
                for operand in task.inputs {
                    exact_length(values, task, operand, task.axis);
                }
            }
            Kind::Slice => exact_length(values, task, task.inputs[0], task.axis),
            Kind::Conv2d => {
                for axis in 1..MAX_RANK {
                    exact_length(values, task, task.inputs[0], axis);
                    exact_length(values, task, task.inputs[1], axis);
                }
            }
            Kind::Conv2dInputGrad | Kind::Conv2dTranspose => {
                for axis in 1..MAX_RANK {
                    exact_length(values, task, task.inputs[0], axis);
                    exact_length(values, task, task.out, axis);
                }
                for axis in 2..MAX_RANK {
                    exact_length(values, task, task.inputs[1], axis);
                }
            }
            Kind::Conv2dWeightGrad => {
                for axis in 1..MAX_RANK {
                    exact_length(values, task, task.inputs[2], axis);
                    exact_length(values, task, task.out, axis);
                }
                for axis in 2..MAX_RANK {
                    exact_length(values, task, task.inputs[1], axis);
                }
            }
            Kind::PoolMax2d | Kind::PoolMean2d => {
                for axis in 2..MAX_RANK {
                    exact_length(values, task, task.inputs[0], axis);
                }
            }
            Kind::PoolMax2dInputGrad | Kind::PoolMean2dInputGrad => {
                for axis in 2..MAX_RANK {
                    exact_length(values, task, task.inputs[0], axis);
                    exact_length(values, task, task.inputs[1], axis);
                    exact_length(values, task, task.out, axis);
                }
            }
            Kind::Rope | Kind::RopeGrad => exact_length(values, task, task.inputs[0], 3),
            Kind::Matmul
            | Kind::MatmulFold
            | Kind::Attention
            | Kind::AttentionQueryGrad
            | Kind::AttentionKeyGrad
            | Kind::AttentionValueGrad
            | Kind::Binary
            | Kind::Unary
            | Kind::Partial
            | Kind::PrefixChunk
            | Kind::PrefixScan
            | Kind::PrefixClose
            | Kind::Fill
            | Kind::Noise
            | Kind::Broadcast
            | Kind::Layout
            | Kind::Extend
            | Kind::SumChunk
            | Kind::SumAxis
            | Kind::SegmentSum
            | Kind::Softmax
            | Kind::SoftmaxGrad
            | Kind::LogSoftmax
            | Kind::LogSoftmaxGrad
            | Kind::Argmax
            | Kind::Categorical
            | Kind::Sample
            | Kind::TopK
            | Kind::OneHot
            | Kind::Gather
            | Kind::Scatter
            | Kind::ScatterWrite
            | Kind::Compact
            | Kind::Convert
            | Kind::Rows
            | Kind::MatmulWeightGrad
            | Kind::Select
            | Kind::Length => {}
        }
    }
}

fn exact_length(values: &[ValueInfo], task: &Task, value: u32, axis: u32) {
    if value == NO_VALUE {
        return;
    }
    let info = &values[value as usize];
    assert!(
        info.shape.free(axis).is_none(),
        "a {} task holds the length of axis {axis} as the number the plan froze, and value {value} of {:?} walks a free extent whose length a binding rules",
        task.kind.name(),
        info.shape.dims(),
    );
}

fn assert_authored_walks_read_the_counts_that_rule_them(
    values: usize,
    patches: &authored::Patches,
    tasks: &[Task],
    authored_walks: &[bool],
) {
    if patches.counted.is_empty() {
        return;
    }
    let mut ruled = vec![false; values];
    for counted in &patches.counted {
        for value in &counted.values {
            ruled[*value as usize] = true;
        }
        for index in &counted.tasks {
            assert!(
                authored_walks[*index as usize],
                "task {index} walks a count the device authors, and the host predicts its walk only where no count rules it",
            );
        }
    }
    for (index, task) in tasks.iter().enumerate() {
        if authored_walks[index] {
            continue;
        }
        for value in task.reads().chain(task.writes()) {
            assert!(
                !ruled[value as usize],
                "task {index} reads tensor {value} of a count the device authors, and the host predicts its walk only where no count rules it",
            );
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

fn storage_liveness<T: Reads + Walk>(sized: &Sized<'_>, tasks: &[T]) -> Vec<Option<Live>> {
    let mut live = vec![None::<Live>; sized.len()];
    let mut readers = Vec::new();
    for (position, task) in tasks.iter().enumerate() {
        let writes = Access::over(|value| sized.storage(value), task)
            .writes()
            .to_vec();
        let aliases = writes
            .first()
            .is_some_and(|write| reads_every_element_in_place(sized, task.kind(), task, *write));
        for write in &writes {
            touch(&mut live, *write, position, false, None);
        }
        readers.clear();
        for value in task.reads() {
            if readers.contains(&value) {
                continue;
            }
            readers.push(value);
            let storage = sized.storage(value);
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

fn reads_every_element_in_place<T: Reads>(
    sized: &Sized<'_>,
    kind: Kind,
    task: &T,
    write: u32,
) -> bool {
    if !matches!(
        kind,
        Kind::Binary | Kind::Unary | Kind::Partial | Kind::Fill | Kind::Broadcast
    ) {
        return false;
    }
    task.reads().all(|value| sized.matches(value, write))
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

fn declared_waves(encoding: &Encoding) -> Vec<u32> {
    let mut waves = vec![0u32; encoding.order().len().max(1)];
    for (position, index) in encoding.order().iter().enumerate() {
        waves[*index as usize] = encoding.waves()[position];
    }
    waves
}

fn family_key(mut record: TaskRecord) -> TaskRecord {
    record.first = 0;
    record.count = 0;
    record.planned_first = 0;
    record.planned_count = 0;
    record.prelude = 0;
    record.chain = 0;
    record.index = 0;
    record.group = 0;
    record.wave = 0;
    record.patch = 0;
    record
}

fn split_key(split: Split) -> Split {
    match split {
        Split::Range { .. } => Split::Range { first: 0, count: 0 },
        Split::Uniform { measure, .. } => Split::Uniform {
            measure,
            index: 0,
            group: 0,
        },
        Split::Plane {
            measure,
            planes,
            plane,
            ..
        } => Split::Plane {
            measure,
            planes,
            plane,
            index: 0,
            group: 0,
        },
        Split::Segment { measure, plane, .. } => Split::Segment {
            measure,
            plane,
            index: 0,
            group: 0,
        },
        Split::Ragged { planes, plane, .. } => Split::Ragged {
            planes,
            plane,
            index: 0,
            group: 0,
        },
    }
}

fn identity_kind(kind: Kind) -> bool {
    matches!(
        kind,
        Kind::SumChunk | Kind::PrefixChunk | Kind::PrefixScan | Kind::PrefixClose | Kind::Length
    )
}

fn scaled(work: u64, count: u32, base: u32) -> u64 {
    work.saturating_mul(u64::from(count)) / u64::from(base.max(1))
}

fn record_table(sized: &Sized<'_>, value: u32) -> u32 {
    if !sized.quantized(value) {
        return NO_VALUE;
    }
    u32::try_from(sized.table(value)).unwrap_or_else(|_| {
        panic!("the quantum table of value {value} lies beyond the device address space")
    })
}

fn quanta(sized: &Sized<'_>, offsets: &[u64]) -> Vec<Quantum> {
    (0..sized.len())
        .filter(|id| {
            sized.storage(*id as u32) as usize == *id
                && sized.resident(*id as u32)
                && sized.quantized(*id as u32)
        })
        .map(|id| Quantum {
            offset: offsets[id] + sized.table(id as u32) * WORD_BYTES,
            scale: sized.scale(id as u32),
        })
        .collect()
}

fn info_of(values: &[ValueInfo], value: u32) -> &ValueInfo {
    &values[values[value as usize].storage as usize]
}

fn held(values: &[ValueInfo], storage: usize) -> bool {
    matches!(
        values[storage].residency,
        Residency::Input | Residency::Parameter | Residency::State | Residency::Resident
    )
}

fn allocate(
    sized: &Sized<'_>,
    live: &[Option<Live>],
    waves: &[u32],
    alignment: u64,
    reserved: u64,
) -> (Vec<u64>, u64, Vec<bool>) {
    let wave_of = |position: usize| waves[position];
    let mut arena = Blocks::with_base(reserved);
    let mut offsets = vec![0u64; live.len()];
    let mut placed = vec![false; live.len()];
    let owners = (0..sized.len()).filter(|id| sized.storage(*id as u32) as usize == *id);
    for id in owners.clone() {
        let storage = id as u32;
        if sized.held(storage) || sized.retained(storage) || sized.owns_its_quanta(storage) {
            if sized.resident(storage) {
                offsets[id] = arena.reserve(sized.bytes(storage), alignment);
            }
            placed[id] = true;
        }
    }

    let mut pending = live
        .iter()
        .enumerate()
        .filter(|(id, _)| !sized.held(*id as u32) && !sized.retained(*id as u32))
        .filter(|(id, _)| !sized.owns_its_quanta(*id as u32))
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
            let bytes = sized.bytes(storage as u32);
            assert_eq!(
                held.bytes, bytes,
                "a value read in place by one task hands that task a storage of another size",
            );
            offsets[storage] = held.offset;
            placed[storage] = true;
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
        let bytes = sized.bytes(storage as u32);
        let offset = arena.reserve(bytes, alignment);
        offsets[storage] = offset;
        placed[storage] = true;
        active.push(Active {
            live,
            offset,
            bytes,
        });
    }
    (offsets, arena.bytes(), placed)
}
