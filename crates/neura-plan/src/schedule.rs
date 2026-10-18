use crate::access::Access;
use crate::hazard::{Hazard, Hazards};
use crate::region::{self, GeometryTiles, Walk};
use neura_abi::{SegmentFields, SegmentRecord};
use neura_profile::{ConvolutionInputTile, ConvolutionTile, MatmulTile};

pub(crate) trait Scheduled: Walk {
    fn span(&self) -> (u32, u32);
}

pub(crate) struct Schedule {
    order: Vec<u32>,
    segments: Vec<SegmentRecord>,
    waves: Vec<u32>,
    count: u32,
}

struct Packed {
    waves: Vec<u32>,
    segments: Vec<Vec<u32>>,
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
        let mut schedule = Self {
            order,
            segments,
            waves,
            count,
        };
        schedule.compact();
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
    hazards: Hazards,
    waves: Vec<u32>,
    segments: Vec<Vec<u32>>,
    segment_of: Vec<u32>,
    segment_wave: Vec<u32>,
    entry: Hazard,
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
            hazards: Hazards::of(values.len()),
            waves: vec![0u32; tasks.len()],
            segments: Vec::new(),
            segment_of: vec![0u32; tasks.len()],
            segment_wave: Vec::new(),
            entry: Hazard::default(),
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
        let mut evidence = self
            .hazards
            .inspect(self.values, &self.touches, task.in_place());
        for dependency in &self.barriers[index as usize] {
            evidence.join(&Hazard::at(
                self.waves[*dependency as usize],
                self.segment_of[*dependency as usize],
            ));
        }
        let earliest = evidence.wave;
        let mut dependencies = std::mem::take(&mut evidence.segments);
        dependencies.settle();
        let joined = earliest
            .filter(|earliest| {
                !dependencies.overflowed()
                    && self.lonely[index as usize]
                    && !recomputed
                    && dependencies.len() == 1
                    && self.segment_wave.get(dependencies.get(0) as usize).copied()
                        == Some(*earliest)
            })
            .map(|earliest| (earliest, dependencies.get(0)));
        let (wave, segment) = match joined {
            Some((wave, segment)) => (wave, segment),
            None => {
                let wave = match dependencies.overflowed() {
                    true => earliest.expect("a hazard that overflows names a wave") + 1,
                    false => earliest.map_or_else(
                        || self.declared.get(index as usize).copied().unwrap_or(0),
                        |earliest| earliest + 1,
                    ),
                };
                self.segments.push(Vec::new());
                self.segment_wave.push(wave);
                (wave, (self.segments.len() - 1) as u32)
            }
        };
        let wave = if recomputed {
            wave.max(self.stage)
        } else {
            wave
        };
        if let Some(earliest) = earliest {
            let ordered = wave > earliest
                || (wave == earliest && dependencies.iter().all(|reached| *reached == segment));
            assert!(
                ordered,
                "task {index} lands in wave {wave} while a hazard of its own reaches wave {earliest} of segments {dependencies:?}, and the device gates two waves apart only what a segment orders",
            );
        }
        self.segment_wave[segment as usize] = self.segment_wave[segment as usize].max(wave);
        self.waves[index as usize] = wave;
        self.segment_of[index as usize] = segment;
        self.segments[segment as usize].push(index);
        self.prefix = self.prefix.max(wave);
        self.entry.wave = Some(wave);
        self.entry.segments.clear();
        self.entry.segments.extend_from(&[segment]);
        self.hazards.record(self.values, &self.touches, &self.entry);
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
