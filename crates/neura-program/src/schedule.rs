use crate::access::Access;
use crate::lower::Task;
use crate::region::{self, Region};
use neura_abi::{MAX_DISPATCH_SEGMENTS, SegmentFields, SegmentRecord};
use neura_graph::ValueInfo;
use neura_profile::MatmulTile;

const MERGE_SPREAD: u64 = 16;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dispatch {
    pub first_segment: u32,
    pub segments: u32,
}

pub(crate) struct Schedule {
    order: Vec<u32>,
    ends: Vec<u32>,
    segments: Vec<SegmentRecord>,
    dispatches: Vec<Dispatch>,
}

struct Packed {
    waves: Vec<u32>,
    segments: Vec<Vec<u32>>,
}

impl Schedule {
    pub(crate) fn of(
        values: &[ValueInfo],
        tiles: &[MatmulTile],
        tasks: &[Task],
        workgroups: u32,
    ) -> Self {
        let conflicts = conflicts(values, tiles, tasks);
        let packed = pack(values, tasks, &conflicts, workgroups);
        let schedule = Self::layout(&packed);
        assert_ordered(values, tiles, tasks, &schedule);
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
        let mut segments = Vec::with_capacity(packed.segments.len());
        let mut ends = Vec::new();
        let mut dispatches = Vec::new();
        let mut cursor = 0;
        while cursor < grouped.len() {
            let wave = packed.waves[packed.segments[grouped[cursor]][0] as usize];
            let mut last = cursor;
            while last < grouped.len()
                && packed.waves[packed.segments[grouped[last]][0] as usize] == wave
            {
                last += 1;
            }
            for chunk in grouped[cursor..last].chunks(MAX_DISPATCH_SEGMENTS as usize) {
                let first_segment = segments.len() as u32;
                for segment in chunk {
                    segments.push(SegmentRecord::of(SegmentFields {
                        first: order.len() as u32,
                        count: packed.segments[*segment].len() as u32,
                    }));
                    order.extend(packed.segments[*segment].iter().copied());
                }
                dispatches.push(Dispatch {
                    first_segment,
                    segments: chunk.len() as u32,
                });
                ends.push(order.len() as u32);
            }
            cursor = last;
        }
        Self {
            order,
            ends,
            segments,
            dispatches,
        }
    }

    pub(crate) fn order(&self) -> &[u32] {
        &self.order
    }

    pub(crate) fn ends(&self) -> &[u32] {
        &self.ends
    }

    pub(crate) fn segments(&self) -> &[SegmentRecord] {
        &self.segments
    }

    pub(crate) fn dispatches(&self) -> &[Dispatch] {
        &self.dispatches
    }
}

fn accesses(
    values: &[ValueInfo],
    tiles: &[MatmulTile],
    tasks: &[Task],
    mut visit: impl FnMut(u32, u32),
) {
    let mut writers = vec![Vec::<(u32, Region)>::new(); values.len()];
    let mut readers = vec![Vec::<(u32, Region)>::new(); values.len()];
    for (index, task) in tasks.iter().enumerate() {
        let index = index as u32;
        let touches = region::touches(values, tiles, task);
        for (storage, region) in &touches.reads {
            for (writer, written) in &writers[*storage as usize] {
                if region.overlaps(*written) {
                    visit(*writer, index);
                }
            }
        }
        for (storage, region) in &touches.writes {
            for (reader, read) in &readers[*storage as usize] {
                if region.overlaps(*read) {
                    visit(*reader, index);
                }
            }
            if task.in_place {
                for (writer, written) in &writers[*storage as usize] {
                    if region.overlaps(*written) {
                        visit(*writer, index);
                    }
                }
            }
        }
        for (storage, region) in touches.reads {
            readers[storage as usize].push((index, region));
        }
        for (storage, region) in touches.writes {
            writers[storage as usize].push((index, region));
            readers[storage as usize].clear();
        }
    }
}

fn conflicts(values: &[ValueInfo], tiles: &[MatmulTile], tasks: &[Task]) -> Vec<Vec<u32>> {
    let mut conflicts = vec![Vec::new(); tasks.len()];
    accesses(values, tiles, tasks, |before, after| {
        conflicts[after as usize].push(before);
    });
    barriers(values, tasks, &mut conflicts);
    for conflict in &mut conflicts {
        conflict.sort_unstable();
        conflict.dedup();
    }
    conflicts
}

fn recomputes(values: &[ValueInfo], task: &Task) -> bool {
    let access = Access::of(values, task);
    access
        .reads()
        .iter()
        .chain(access.writes())
        .any(|storage| values[*storage as usize].recomputes.is_some())
}

fn barriers(values: &[ValueInfo], tasks: &[Task], conflicts: &mut [Vec<u32>]) {
    let mut touched = vec![Vec::<u32>::new(); values.len()];
    for (index, task) in tasks.iter().enumerate() {
        let access = Access::of(values, task);
        for storage in access.reads().iter().chain(access.writes()) {
            touched[*storage as usize].push(index as u32);
        }
    }
    for (index, task) in tasks.iter().enumerate() {
        let Some(original) = values[task.out as usize].recomputes else {
            continue;
        };
        for before in &touched[original as usize] {
            if *before != index as u32 {
                conflicts[index].push(*before);
            }
        }
    }
}

enum Placement {
    Fold { wave: u32, segment: u32 },
    Merge { wave: u32, segments: Vec<u32> },
    Open { wave: u32 },
}

fn pack(values: &[ValueInfo], tasks: &[Task], conflicts: &[Vec<u32>], workgroups: u32) -> Packed {
    let lonely = lonely(conflicts);
    let mut waves = vec![0u32; tasks.len()];
    let mut segments = Vec::<Vec<u32>>::new();
    let mut work = Vec::<u64>::new();
    let mut created = Vec::<u32>::new();
    let mut rebuilt = Vec::<bool>::new();
    let mut segment_of = vec![0u32; tasks.len()];
    let mut prefix = 0u32;
    let mut stage = 0u32;
    let mut in_recompute = false;
    for index in 0..tasks.len() {
        let index = index as u32;
        let task = &tasks[index as usize];
        let recomputed = recomputes(values, task);
        if recomputed && !in_recompute {
            stage = prefix + 1;
        }
        in_recompute = recomputed;
        let conflicts = &conflicts[index as usize];
        let earliest = conflicts
            .iter()
            .map(|dependency| waves[*dependency as usize])
            .max();
        let folded = (lonely[index as usize] && !recomputed)
            .then(|| {
                let earliest = earliest?;
                (0..segments.len())
                    .filter(|segment| {
                        conflicts
                            .iter()
                            .any(|dependency| segment_of[*dependency as usize] == *segment as u32)
                    })
                    .filter(|segment| {
                        conflicts.iter().all(|dependency| {
                            segment_of[*dependency as usize] == *segment as u32
                                || waves[*dependency as usize] < earliest
                        })
                    })
                    .min_by_key(|segment| (work[*segment], *segment))
                    .map(|segment| (earliest, segment as u32))
            })
            .flatten();
        let placement = match folded {
            Some((wave, segment)) => Placement::Fold { wave, segment },
            None => close(
                task,
                conflicts,
                earliest,
                &waves,
                &segment_of,
                &work,
                &created,
                &rebuilt,
                recomputed,
                workgroups,
            ),
        };
        let (wave, segment) = match placement {
            Placement::Fold { wave, segment } => (wave, segment),
            Placement::Merge {
                wave,
                segments: merged,
            } => {
                let target = merged[0];
                for segment in &merged[1..] {
                    let moved = std::mem::take(&mut segments[*segment as usize]);
                    for task in &moved {
                        segment_of[*task as usize] = target;
                    }
                    work[target as usize] += work[*segment as usize];
                    work[*segment as usize] = 0;
                    segments[target as usize].extend(moved);
                }
                (wave, target)
            }
            Placement::Open { wave } => {
                segments.push(Vec::new());
                work.push(0);
                rebuilt.push(false);
                (wave, (segments.len() - 1) as u32)
            }
        };
        let wave = if recomputed { wave.max(stage) } else { wave };
        waves[index as usize] = wave;
        segment_of[index as usize] = segment;
        if segments[segment as usize].is_empty() {
            if created.len() <= wave as usize {
                created.resize(wave as usize + 1, 0);
            }
            created[wave as usize] += 1;
        }
        work[segment as usize] += task.work;
        rebuilt[segment as usize] |= recomputed;
        segments[segment as usize].push(index);
        prefix = prefix.max(wave);
    }
    Packed { waves, segments }
}

#[allow(clippy::too_many_arguments)]
fn close(
    task: &Task,
    conflicts: &[u32],
    earliest: Option<u32>,
    waves: &[u32],
    segment_of: &[u32],
    work: &[u64],
    created: &[u32],
    rebuilt: &[bool],
    recomputed: bool,
    workgroups: u32,
) -> Placement {
    let Some(earliest) = earliest else {
        return Placement::Open { wave: 0 };
    };
    let mut dependencies = conflicts
        .iter()
        .filter(|dependency| waves[**dependency as usize] == earliest)
        .map(|dependency| segment_of[*dependency as usize])
        .collect::<Vec<u32>>();
    dependencies.sort_unstable();
    dependencies.dedup();
    let heaviest = dependencies
        .iter()
        .map(|segment| work[*segment as usize])
        .max()
        .unwrap_or(0);
    let claimed = dependencies
        .iter()
        .map(|segment| work[*segment as usize])
        .sum::<u64>();
    let saturated = created.get(earliest as usize).copied().unwrap_or(0) >= workgroups;
    if dependencies.len() > 1
        && heaviest > 0
        && !saturated
        && !recomputed
        && dependencies
            .iter()
            .all(|segment| !rebuilt[*segment as usize])
        && claimed + task.work <= MERGE_SPREAD * heaviest
    {
        return Placement::Merge {
            wave: earliest,
            segments: dependencies,
        };
    }
    Placement::Open { wave: earliest + 1 }
}

fn lonely(conflicts: &[Vec<u32>]) -> Vec<bool> {
    let mut depths = vec![0u32; conflicts.len()];
    for (index, conflicts) in conflicts.iter().enumerate() {
        depths[index] = conflicts
            .iter()
            .map(|dependency| depths[*dependency as usize] + 1)
            .max()
            .unwrap_or(0);
    }
    let levels = depths.iter().copied().max().map_or(0, |depth| depth + 1);
    let mut width = vec![0u32; levels as usize];
    for depth in &depths {
        width[*depth as usize] += 1;
    }
    depths
        .iter()
        .map(|depth| width[*depth as usize] == 1)
        .collect()
}

fn assert_ordered(values: &[ValueInfo], tiles: &[MatmulTile], tasks: &[Task], schedule: &Schedule) {
    let mut dispatch_of = vec![0u32; schedule.segments.len()];
    for (index, dispatch) in schedule.dispatches.iter().enumerate() {
        for segment in dispatch.first_segment..dispatch.first_segment + dispatch.segments {
            dispatch_of[segment as usize] = index as u32;
        }
    }
    let mut dispatch = vec![0u32; tasks.len()];
    let mut segment = vec![0u32; tasks.len()];
    let mut position = vec![0u32; tasks.len()];
    for (index, record) in schedule.segments.iter().enumerate() {
        for (within, task) in (record.first..record.first + record.count).enumerate() {
            let task = schedule.order[task as usize] as usize;
            dispatch[task] = dispatch_of[index];
            segment[task] = index as u32;
            position[task] = within as u32;
        }
    }
    accesses(values, tiles, tasks, |before, after| {
        let (before, after) = (before as usize, after as usize);
        let ordered = dispatch[before] < dispatch[after]
            || (segment[before] == segment[after] && position[before] < position[after]);
        assert!(
            ordered,
            "task {after} touches a tensor that task {before} only reaches later on the tape",
        );
    });
}
