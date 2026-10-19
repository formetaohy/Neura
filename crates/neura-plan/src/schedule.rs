use crate::access::Access;
use crate::flow::{Flow, WordFlow, arena_spans};
use crate::hazard::{Hazard, Hazards};
use crate::region::{self, GeometryTiles, Walk};
use neura_abi::{SegmentFields, SegmentRecord, control};
use neura_profile::{ConvolutionInputTile, ConvolutionTile, MatmulTile};

const GATE_DEPTH: usize = 64;
const GATE_EDGES: u64 = 4096;

pub(crate) trait Scheduled: Walk {
    fn span(&self) -> (u32, u32);
}

pub(crate) struct Schedule {
    order: Vec<u32>,
    segments: Vec<SegmentRecord>,
    waves: Vec<u32>,
    dependencies: Vec<Vec<u32>>,
    successors: Vec<u32>,
    gated: Vec<bool>,
    wave_table: Vec<u32>,
    count: u32,
}

struct Packed {
    waves: Vec<u32>,
    segments: Vec<Vec<u32>>,
    dependencies: Vec<Vec<u32>>,
}

impl Schedule {
    pub(crate) fn of<T: Scheduled, V: region::Values>(
        values: &V,
        tiles: &[MatmulTile],
        windows: &[ConvolutionTile],
        input_windows: &[ConvolutionInputTile],
        tasks: &[T],
        declared: &[u32],
    ) -> Self {
        let barriers = barriers(values, tasks);
        let geometry = GeometryTiles {
            matmul: tiles,
            convolution: windows,
            convolution_input: input_windows,
        };
        let depths = depths(values, geometry, tasks, &barriers);
        let packed = Packer::new(values, geometry, tasks, &barriers, &depths, declared).pack();
        let schedule = Self::layout(&packed);
        assert_barriers(tasks, &barriers, &schedule);
        schedule
    }

    fn layout(packed: &Packed) -> Self {
        let mut grouped = (0..packed.segments.len())
            .filter(|segment| !packed.segments[*segment].is_empty())
            .collect::<Vec<_>>();
        grouped.sort_by_key(|segment| {
            (
                packed.waves[packed.segments[*segment][0] as usize],
                *segment,
            )
        });
        let mut order = Vec::new();
        let mut segments = Vec::with_capacity(grouped.len());
        let mut waves = Vec::new();
        let mut count = 0;
        let mut final_of = vec![0u32; packed.segments.len()];
        for (position, segment) in grouped.iter().enumerate() {
            final_of[*segment] = position as u32;
        }
        for segment in grouped {
            let tasks = &packed.segments[segment];
            let wave = packed.waves[tasks[0] as usize];
            assert!(
                tasks
                    .iter()
                    .all(|task| packed.waves[*task as usize] == wave),
                "a workgroup carries tasks of two waves the device gates apart",
            );
            segments.push(SegmentRecord::of(SegmentFields {
                first: order.len() as u32,
                count: tasks.len() as u32,
                wave,
                predecessors: 0,
                successors: 0,
                successors_count: 0,
            }));
            for task in tasks {
                order.push(*task);
                waves.push(wave);
            }
            count = count.max(wave + 1);
        }
        assert!(
            count as usize <= order.len(),
            "a schedule of {} tasks gates {count} waves, and every wave opens on a task of its own",
            order.len(),
        );
        let mut dependencies = vec![Vec::new(); segments.len()];
        let mut successors_of = vec![Vec::new(); segments.len()];
        for (segment, held) in packed.dependencies.iter().enumerate() {
            if packed.segments[segment].is_empty() {
                continue;
            }
            let here = final_of[segment] as usize;
            for dependency in held {
                let before = final_of[*dependency as usize] as usize;
                assert!(
                    before < here,
                    "segment {here} depends on segment {before}, which the device reaches later",
                );
                dependencies[here].push(before as u32);
                successors_of[before].push(here as u32);
            }
            dependencies[here].sort_unstable();
            dependencies[here].dedup();
        }
        let mut successors = Vec::new();
        for (segment, record) in segments.iter_mut().enumerate() {
            record.predecessors = dependencies[segment].len() as u32;
            record.successors = successors.len() as u32;
            record.successors_count = successors_of[segment].len() as u32;
            successors.extend_from_slice(&successors_of[segment]);
        }
        assert!(
            segments.len() <= control::MAX_SEGMENTS as usize,
            "a plan of {} segments outruns the {} a queue entry names",
            segments.len(),
            control::MAX_SEGMENTS,
        );
        let mut schedule = Self {
            order,
            segments,
            waves,
            dependencies,
            successors,
            gated: vec![false; count as usize],
            wave_table: Vec::new(),
            count,
        };
        schedule.compact();
        schedule.gated = vec![false; schedule.count as usize];
        schedule
    }

    fn compact(&mut self) {
        let mut ranks = std::collections::BTreeMap::new();
        for wave in &self.waves {
            let next = ranks.len() as u32;
            ranks.entry(*wave).or_insert(next);
        }
        for wave in &mut self.waves {
            *wave = ranks[wave];
        }
        for segment in &mut self.segments {
            segment.wave = ranks[&segment.wave];
        }
        self.count = ranks.len() as u32;
        assert!(
            self.count as usize <= self.order.len(),
            "a schedule of {} tasks gates {} waves, and every wave opens on a task of its own",
            self.order.len(),
            self.count,
        );
    }

    pub(crate) fn order(&self) -> &[u32] {
        &self.order
    }

    pub(crate) fn segments(&self) -> &[SegmentRecord] {
        &self.segments
    }

    pub(crate) fn waves(&self) -> &[u32] {
        &self.waves
    }

    pub(crate) fn dependencies(&self) -> &[Vec<u32>] {
        &self.dependencies
    }

    pub(crate) fn successors(&self) -> &[u32] {
        &self.successors
    }

    pub(crate) fn add_arena_hazards<T: Scheduled, V: region::Values>(
        &mut self,
        values: &V,
        tiles: GeometryTiles<'_>,
        resident: &[bool],
        offsets: &[u64],
        tasks: &[T],
        skip_gated: bool,
    ) {
        let mut segment_at = vec![0u32; tasks.len()];
        for (segment, record) in self.segments.iter().enumerate() {
            for at in record.first..record.first + record.count {
                segment_at[at as usize] = segment as u32;
            }
        }
        let mut written = vec![false; values.len()];
        for task in tasks {
            for value in task.writes() {
                written[values.storage(value) as usize] = true;
            }
        }
        let mut flow = WordFlow::default();
        let mut touches = region::Touches::default();
        let mut pages = Vec::new();
        let mut collected = Vec::new();
        for (position, task) in tasks.iter().enumerate() {
            let segment = segment_at[position];
            if skip_gated && self.gated[self.segments[segment as usize].wave as usize] {
                continue;
            }
            region::touches(values, tiles, task, task.span(), &mut touches);
            collected.clear();
            for (storage, region) in &touches.reads {
                let storage = *storage;
                if !resident[storage as usize] || !written[storage as usize] {
                    continue;
                }
                arena_spans(
                    offsets[storage as usize],
                    values.element(storage),
                    values.elements(storage),
                    *region,
                    &mut pages,
                );
                flow.read(&pages, segment, &mut collected);
            }
            for (storage, region) in &touches.writes {
                let storage = *storage;
                if !resident[storage as usize] {
                    continue;
                }
                arena_spans(
                    offsets[storage as usize],
                    values.element(storage),
                    values.elements(storage),
                    *region,
                    &mut pages,
                );
                flow.write(&pages, segment, task.in_place(), &mut collected);
            }
            collected.sort_unstable();
            collected.dedup();
            if self.gated[self.segments[segment as usize].wave as usize] {
                collected.clear();
            }
            for dependency in collected.drain(..) {
                if dependency == segment {
                    continue;
                }
                assert!(
                    dependency < segment,
                    "segment {segment} walks storage a later segment {dependency} still owns",
                );
                self.dependencies[segment as usize].push(dependency);
            }
        }
        self.rebuild_successors();
    }

    pub(crate) fn gate_dense_waves(&mut self, tail: bool) {
        let waves = self.count as usize;
        let mut deepest = vec![0usize; waves];
        let mut edges = vec![0u64; waves];
        for (segment, dependencies) in self.dependencies.iter().enumerate() {
            let wave = self.segments[segment].wave as usize;
            deepest[wave] = deepest[wave].max(dependencies.len());
            edges[wave] += dependencies.len() as u64;
        }
        let mut gated = self.gated.clone();
        for wave in 0..waves {
            gated[wave] |= deepest[wave] > GATE_DEPTH || edges[wave] > GATE_EDGES;
        }
        if tail && let Some(first) = gated.iter().position(|gated| *gated) {
            for wave in &mut gated[first..] {
                *wave = true;
            }
        }
        for (segment, dependencies) in self.dependencies.iter_mut().enumerate() {
            if gated[self.segments[segment].wave as usize] {
                dependencies.clear();
            }
        }
        self.gated = gated;
        self.rebuild_successors();
        self.build_wave_table();
    }

    pub(crate) fn build_wave_table(&mut self) {
        let waves = self.count as usize;
        let mut table = vec![0u32; 1 + waves * 4];
        table[0] = waves as u32;
        for (segment, record) in self.segments.iter().enumerate() {
            let wave = record.wave as usize;
            if table[1 + wave * 4 + 1] == 0 {
                table[1 + wave * 4] = segment as u32;
            }
            table[1 + wave * 4 + 1] += 1;
        }
        for wave in 0..waves {
            table[1 + wave * 4 + 2] = u32::from(self.gated[wave]);
        }
        self.wave_table = table;
    }

    pub(crate) fn gated(&self) -> &[bool] {
        &self.gated
    }

    pub(crate) fn wave_table(&self) -> &[u32] {
        &self.wave_table
    }

    fn rebuild_successors(&mut self) {
        let mut successors_of = vec![Vec::new(); self.segments.len()];
        for (segment, dependencies) in self.dependencies.iter_mut().enumerate() {
            dependencies.sort_unstable();
            dependencies.dedup();
            assert!(
                !dependencies.contains(&(segment as u32)),
                "segment {segment} depends on itself",
            );
            for dependency in dependencies.iter() {
                successors_of[*dependency as usize].push(segment as u32);
            }
        }
        let mut successors = Vec::new();
        for (segment, record) in self.segments.iter_mut().enumerate() {
            record.predecessors = self.dependencies[segment].len() as u32;
            record.successors = successors.len() as u32;
            record.successors_count = successors_of[segment].len() as u32;
            successors.extend_from_slice(&successors_of[segment]);
        }
        self.successors = successors;
    }

    pub(crate) fn wave_segments(&self) -> Vec<u32> {
        let mut segments = vec![0u32; self.count as usize];
        for segment in &self.segments {
            segments[segment.wave as usize] += 1;
        }
        segments
    }
}

fn depths<T: Scheduled, V: region::Values>(
    values: &V,
    tiles: GeometryTiles<'_>,
    tasks: &[T],
    barriers: &[Vec<u32>],
) -> Vec<u32> {
    let mut hazards = Hazards::of(values.len());
    let mut depths = vec![0u32; tasks.len()];
    let mut touches = region::Touches::default();
    for (index, task) in tasks.iter().enumerate() {
        region::touches(values, tiles, task, task.span(), &mut touches);
        let evidence = hazards.inspect(values, &touches, task.in_place());
        let mut deepest = evidence.deepest.map(|deepest| deepest + 1);
        for dependency in &barriers[index] {
            let carried = depths[*dependency as usize] + 1;
            deepest = Some(deepest.map_or(carried, |deepest| deepest.max(carried)));
        }
        depths[index] = deepest.unwrap_or(0);
        hazards.record(values, &touches, &Hazard::deep(depths[index]));
    }
    depths
}

fn lonely(depths: &[u32]) -> Vec<bool> {
    let levels = depths
        .iter()
        .copied()
        .max()
        .map_or(0, |depth| depth as usize + 1);
    let mut width = vec![0u32; levels];
    for depth in depths {
        width[*depth as usize] += 1;
    }
    depths
        .iter()
        .map(|depth| width[*depth as usize] == 1)
        .collect()
}

fn barriers<T: Scheduled, V: region::Values>(values: &V, tasks: &[T]) -> Vec<Vec<u32>> {
    let mut barriers = vec![Vec::new(); tasks.len()];
    if !tasks
        .iter()
        .any(|task| values.recomputes(task.out()).is_some())
    {
        return barriers;
    }
    let mut touched = vec![Vec::<u32>::new(); values.len()];
    for (index, task) in tasks.iter().enumerate() {
        let access = Access::over(|value| values.storage(value), task);
        for storage in access.reads().iter().chain(access.writes()) {
            touched[*storage as usize].push(index as u32);
        }
    }
    for (index, task) in tasks.iter().enumerate() {
        let Some(original) = values.recomputes(task.out()) else {
            continue;
        };
        for before in &touched[original as usize] {
            if *before != index as u32 {
                barriers[index].push(*before);
            }
        }
    }
    barriers
}

struct Packer<'a, T, V> {
    values: &'a V,
    tiles: GeometryTiles<'a>,
    tasks: &'a [T],
    barriers: &'a [Vec<u32>],
    declared: &'a [u32],
    lonely: Vec<bool>,
    flow: Flow,
    waves: Vec<u32>,
    segments: Vec<Vec<u32>>,
    segment_of: Vec<u32>,
    segment_wave: Vec<u32>,
    dependencies: Vec<Vec<u32>>,
    collected: Vec<u32>,
    touches: region::Touches,
    prefix: u32,
    stage: u32,
    in_recompute: bool,
}

impl<'a, T: Scheduled, V: region::Values> Packer<'a, T, V> {
    fn new(
        values: &'a V,
        tiles: GeometryTiles<'a>,
        tasks: &'a [T],
        barriers: &'a [Vec<u32>],
        depths: &'a [u32],
        declared: &'a [u32],
    ) -> Self {
        Self {
            values,
            tiles,
            tasks,
            barriers,
            declared,
            lonely: lonely(depths),
            flow: Flow::of(values, tasks.iter().flat_map(|task| task.writes())),
            waves: vec![0u32; tasks.len()],
            segments: Vec::new(),
            segment_of: vec![0u32; tasks.len()],
            segment_wave: Vec::new(),
            dependencies: Vec::new(),
            collected: Vec::new(),
            touches: region::Touches::default(),
            prefix: 0,
            stage: 0,
            in_recompute: false,
        }
    }

    fn pack(mut self) -> Packed {
        for index in 0..self.tasks.len() as u32 {
            self.place(index);
        }
        Packed {
            waves: self.waves,
            segments: self.segments,
            dependencies: self.dependencies,
        }
    }

    fn place(&mut self, index: u32) {
        let task = &self.tasks[index as usize];
        region::touches(
            self.values,
            self.tiles,
            task,
            task.span(),
            &mut self.touches,
        );
        let recomputed = self
            .touches
            .reads
            .iter()
            .chain(&self.touches.writes)
            .any(|(storage, _)| self.values.recomputes(*storage).is_some());
        if recomputed && !self.in_recompute {
            self.stage = self.prefix + 1;
        }
        self.in_recompute = recomputed;
        self.flow.inspect(
            self.values,
            &self.touches,
            task.in_place(),
            &mut self.collected,
        );
        for dependency in &self.barriers[index as usize] {
            self.collected.push(self.segment_of[*dependency as usize]);
        }
        self.collected.sort_unstable();
        self.collected.dedup();
        let mut earliest = None;
        for dependency in &self.collected {
            let wave = self.segment_wave[*dependency as usize];
            earliest = Some(earliest.map_or(wave, |held: u32| held.max(wave)));
        }
        let joined = earliest
            .filter(|earliest| {
                self.lonely[index as usize]
                    && !recomputed
                    && self.collected.len() == 1
                    && self.segment_wave[self.collected[0] as usize] == *earliest
            })
            .map(|earliest| (earliest, self.collected[0]));
        let (wave, segment) = match joined {
            Some((wave, segment)) => (wave, segment),
            None => {
                let wave = earliest.map_or_else(
                    || self.declared.get(index as usize).copied().unwrap_or(0),
                    |earliest| earliest + 1,
                );
                self.segments.push(Vec::new());
                self.segment_wave.push(wave);
                self.dependencies.push(Vec::new());
                (wave, (self.segments.len() - 1) as u32)
            }
        };
        let wave = if recomputed {
            wave.max(self.stage)
        } else {
            wave
        };
        self.collected.retain(|dependency| *dependency != segment);
        for dependency in &self.collected {
            assert!(
                self.segment_wave[*dependency as usize] < wave,
                "task {index} lands in wave {wave} while a dependency of its own reaches wave {} of segment {dependency}",
                self.segment_wave[*dependency as usize],
            );
        }
        self.segment_wave[segment as usize] = self.segment_wave[segment as usize].max(wave);
        self.waves[index as usize] = wave;
        self.segment_of[index as usize] = segment;
        self.segments[segment as usize].push(index);
        self.dependencies[segment as usize].extend_from_slice(&self.collected);
        self.prefix = self.prefix.max(wave);
        self.flow.record(self.values, &self.touches, segment);
    }
}

fn assert_barriers<T: Scheduled>(tasks: &[T], barriers: &[Vec<u32>], schedule: &Schedule) {
    if barriers.iter().all(|dependencies| dependencies.is_empty()) {
        return;
    }
    let mut seat = vec![(0u32, 0u32, 0u32); tasks.len()];
    for (index, record) in schedule.segments.iter().enumerate() {
        for (within, at) in (record.first..record.first + record.count).enumerate() {
            let task = schedule.order[at as usize] as usize;
            seat[task] = (index as u32, within as u32, schedule.waves[at as usize]);
        }
    }
    for (index, dependencies) in barriers.iter().enumerate() {
        for dependency in dependencies {
            let before = *dependency as usize;
            let ordered = seat[before].2 < seat[index].2
                || (seat[before].0 == seat[index].0 && seat[before].1 < seat[index].1);
            assert!(
                ordered,
                "task {index} recomputes a tensor that task {before} touches later in the plan",
            );
        }
    }
}
