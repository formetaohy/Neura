use crate::access::Access;
use crate::hazard::{Hazard, Hazards};
use crate::region::{self, Walk};
use neura_abi::{SegmentFields, SegmentRecord};
use neura_profile::MatmulTile;

const MERGE_SPREAD: u64 = 16;

pub(crate) trait Scheduled: Walk {
    fn span(&self) -> (u32, u32);
    fn work(&self) -> u64;
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
        tasks: &[T],
        workgroups: u32,
        declared: &[u32],
    ) -> Self {
        let barriers = barriers(values, tasks);
        let depths = depths(values, tiles, tasks, &barriers);
        let packed = pack(
            values, tiles, tasks, &barriers, &depths, workgroups, declared,
        );
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
        Self {
            order,
            segments,
            waves,
            count,
        }
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

    pub(crate) fn wave_tasks(&self) -> Vec<u32> {
        let mut tasks = vec![0u32; self.count as usize];
        for wave in &self.waves {
            tasks[*wave as usize] += 1;
        }
        tasks
    }
}

fn depths<T: Scheduled, V: region::Values>(
    values: &V,
    tiles: &[MatmulTile],
    tasks: &[T],
    barriers: &[Vec<u32>],
) -> Vec<u32> {
    let mut hazards = Hazards::of(values.len());
    let mut depths = vec![0u32; tasks.len()];
    for (index, task) in tasks.iter().enumerate() {
        let touches = region::touches(values, tiles, task, task.span());
        let evidence = hazards.inspect(values, &touches, task.in_place());
        let mut deepest = evidence.deepest.map(|deepest| deepest + 1);
        for dependency in &barriers[index] {
            let carried = depths[*dependency as usize] + 1;
            deepest = Some(deepest.map_or(carried, |deepest| deepest.max(carried)));
        }
        depths[index] = deepest.unwrap_or(0);
        hazards.record(values, touches, &Hazard::deep(depths[index]));
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

#[derive(Default)]
struct Ancestry {
    parents: Vec<u32>,
}

impl Ancestry {
    fn push(&mut self) {
        let segment = self.parents.len() as u32;
        self.parents.push(segment);
    }

    fn root(&self, segment: u32) -> u32 {
        let mut root = segment;
        loop {
            let Some(&parent) = self.parents.get(root as usize) else {
                return root;
            };
            if parent == root {
                return root;
            }
            root = parent;
        }
    }

    fn adopt(&mut self, kept: u32, absorbed: u32) {
        let absorbed = self.root(absorbed);
        if absorbed != kept {
            self.parents[absorbed as usize] = kept;
        }
    }
}

enum Placement {
    Fold { wave: u32, segment: u32 },
    Merge { wave: u32, segments: Vec<u32> },
    Open { wave: u32 },
}

#[allow(clippy::too_many_arguments)]
fn pack<T: Scheduled, V: region::Values>(
    values: &V,
    tiles: &[MatmulTile],
    tasks: &[T],
    barriers: &[Vec<u32>],
    depths: &[u32],
    workgroups: u32,
    declared: &[u32],
) -> Packed {
    let lonely = lonely(depths);
    let mut hazards = Hazards::of(values.len());
    let mut waves = vec![0u32; tasks.len()];
    let mut segments = Vec::<Vec<u32>>::new();
    let mut work = Vec::<u64>::new();
    let mut created = Vec::<u32>::new();
    let mut rebuilt = Vec::<bool>::new();
    let mut segment_of = vec![0u32; tasks.len()];
    let mut segment_wave = Vec::<u32>::new();
    let mut ancestry = Ancestry::default();
    let mut prefix = 0u32;
    let mut stage = 0u32;
    let mut in_recompute = false;
    let mut entry = Hazard::default();
    for index in 0..tasks.len() as u32 {
        let task = &tasks[index as usize];
        let touches = region::touches(values, tiles, task, task.span());
        let recomputed = touches
            .reads
            .iter()
            .chain(&touches.writes)
            .any(|(storage, _)| values.recomputes(*storage).is_some());
        if recomputed && !in_recompute {
            stage = prefix + 1;
        }
        in_recompute = recomputed;
        let mut evidence = hazards.inspect(values, &touches, task.in_place());
        for dependency in &barriers[index as usize] {
            evidence.join(&Hazard::at(
                waves[*dependency as usize],
                ancestry.root(segment_of[*dependency as usize]),
            ));
        }
        let earliest = evidence.wave;
        let mut dependencies = std::mem::take(&mut evidence.segments);
        for reached in &mut dependencies {
            *reached = ancestry.root(*reached);
        }
        dependencies.sort_unstable();
        dependencies.dedup();
        let folded = earliest
            .filter(|earliest| {
                lonely[index as usize]
                    && !recomputed
                    && dependencies.len() == 1
                    && segment_wave.get(dependencies[0] as usize).copied() == Some(*earliest)
            })
            .map(|earliest| (earliest, dependencies[0]));
        let placement = match folded {
            Some((wave, segment)) => Placement::Fold { wave, segment },
            None => close(
                task.work(),
                &dependencies,
                earliest,
                &work,
                &created,
                &rebuilt,
                recomputed,
                workgroups,
                declared.get(index as usize).copied().unwrap_or(0),
            ),
        };
        let (wave, segment) = match placement {
            Placement::Fold { wave, segment } => (wave, segment),
            Placement::Merge {
                wave,
                segments: merged,
            } => {
                let kept = merged[0];
                for absorbed in &merged[1..] {
                    let moved = std::mem::take(&mut segments[*absorbed as usize]);
                    work[kept as usize] += work[*absorbed as usize];
                    work[*absorbed as usize] = 0;
                    segments[kept as usize].extend(moved);
                    ancestry.adopt(kept, *absorbed);
                }
                for reached in &mut dependencies {
                    *reached = ancestry.root(*reached);
                }
                (wave, kept)
            }
            Placement::Open { wave } => {
                segments.push(Vec::new());
                work.push(0);
                rebuilt.push(false);
                segment_wave.push(wave);
                ancestry.push();
                (wave, (segments.len() - 1) as u32)
            }
        };
        let wave = if recomputed { wave.max(stage) } else { wave };
        if let Some(earliest) = earliest {
            let ordered = wave > earliest
                || (wave == earliest && dependencies.iter().all(|reached| *reached == segment));
            assert!(
                ordered,
                "task {index} lands in wave {wave} while a hazard of its own reaches wave {earliest} of segments {dependencies:?}, and the device gates two waves apart only what a segment orders",
            );
        }
        segment_wave[segment as usize] = segment_wave[segment as usize].max(wave);
        waves[index as usize] = wave;
        segment_of[index as usize] = segment;
        if segments[segment as usize].is_empty() {
            if created.len() <= wave as usize {
                created.resize(wave as usize + 1, 0);
            }
            created[wave as usize] += 1;
        }
        work[segment as usize] += task.work();
        rebuilt[segment as usize] |= recomputed;
        segments[segment as usize].push(index);
        prefix = prefix.max(wave);
        entry.wave = Some(wave);
        entry.segments.clear();
        entry.segments.push(segment);
        hazards.record(values, touches, &entry);
    }
    Packed { waves, segments }
}

#[allow(clippy::too_many_arguments)]
fn close(
    work_of_task: u64,
    dependencies: &[u32],
    earliest: Option<u32>,
    work: &[u64],
    created: &[u32],
    rebuilt: &[bool],
    recomputed: bool,
    workgroups: u32,
    declared: u32,
) -> Placement {
    let Some(earliest) = earliest else {
        return Placement::Open { wave: declared };
    };
    let heaviest = dependencies
        .iter()
        .map(|segment| work[*segment as usize])
        .max()
        .unwrap_or(0);
    let claimed = dependencies
        .iter()
        .map(|segment| work[*segment as usize])
        .sum::<u64>();
    let segments_after_the_merge = created
        .get(earliest as usize)
        .copied()
        .unwrap_or(0)
        .saturating_sub(dependencies.len().saturating_sub(1) as u32);
    let keeps_every_workgroup_busy = segments_after_the_merge >= workgroups;
    if dependencies.len() > 1
        && heaviest > 0
        && keeps_every_workgroup_busy
        && !recomputed
        && dependencies
            .iter()
            .all(|segment| !rebuilt[*segment as usize])
        && claimed + work_of_task <= MERGE_SPREAD * heaviest
    {
        return Placement::Merge {
            wave: earliest,
            segments: dependencies.to_vec(),
        };
    }
    Placement::Open { wave: earliest + 1 }
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
