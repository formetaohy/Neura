use crate::access::Access;
use crate::lower::Task;
use neura_abi::{MAX_DISPATCH_SEGMENTS, SegmentFields, SegmentRecord};
use neura_graph::ValueInfo;

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
    pub(crate) fn of(values: &[ValueInfo], tasks: &[Task]) -> Self {
        let conflicts = conflicts(values, tasks);
        let packed = pack(tasks, &conflicts);
        let schedule = Self::layout(&packed);
        assert_ordered(values, tasks, &schedule);
        schedule
    }

    fn layout(packed: &Packed) -> Self {
        let mut grouped = (0..packed.segments.len()).collect::<Vec<_>>();
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

fn accesses(values: &[ValueInfo], tasks: &[Task], mut visit: impl FnMut(u32, u32)) {
    let mut writers = vec![Vec::<u32>::new(); values.len()];
    let mut readers = vec![Vec::<u32>::new(); values.len()];
    for (index, task) in tasks.iter().enumerate() {
        let index = index as u32;
        let access = Access::of(values, task);
        for storage in access.reads() {
            for writer in &writers[*storage as usize] {
                visit(*writer, index);
            }
        }
        for storage in access.writes() {
            for reader in &readers[*storage as usize] {
                visit(*reader, index);
            }
            if access.in_place() {
                for writer in &writers[*storage as usize] {
                    visit(*writer, index);
                }
            }
        }
        for storage in access.reads() {
            readers[*storage as usize].push(index);
        }
        for storage in access.writes() {
            writers[*storage as usize].push(index);
            readers[*storage as usize].clear();
        }
    }
}

fn conflicts(values: &[ValueInfo], tasks: &[Task]) -> Vec<Vec<u32>> {
    let mut conflicts = vec![Vec::new(); tasks.len()];
    accesses(values, tasks, |before, after| {
        conflicts[after as usize].push(before);
    });
    for conflict in &mut conflicts {
        conflict.sort_unstable();
        conflict.dedup();
    }
    conflicts
}

fn pack(tasks: &[Task], conflicts: &[Vec<u32>]) -> Packed {
    let lonely = lonely(conflicts);
    let mut waves = vec![0u32; tasks.len()];
    let mut segments = Vec::<Vec<u32>>::new();
    let mut segment_of = vec![0u32; tasks.len()];
    let mut work = Vec::<u64>::new();
    for index in 0..tasks.len() {
        let index = index as u32;
        let conflicts = &conflicts[index as usize];
        let folded = lonely[index as usize]
            .then(|| {
                let earliest = conflicts
                    .iter()
                    .map(|dependency| waves[*dependency as usize])
                    .max()?;
                (0..segments.len())
                    .filter(|segment| waves[segments[*segment][0] as usize] == earliest)
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
        let (wave, segment) = match folded {
            Some(folded) => folded,
            None => {
                segments.push(Vec::new());
                work.push(0);
                (
                    conflicts
                        .iter()
                        .map(|dependency| waves[*dependency as usize] + 1)
                        .max()
                        .unwrap_or(0),
                    (segments.len() - 1) as u32,
                )
            }
        };
        waves[index as usize] = wave;
        segment_of[index as usize] = segment;
        work[segment as usize] += tasks[index as usize].work;
        segments[segment as usize].push(index);
    }
    Packed { waves, segments }
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

fn assert_ordered(values: &[ValueInfo], tasks: &[Task], schedule: &Schedule) {
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
    accesses(values, tasks, |before, after| {
        let (before, after) = (before as usize, after as usize);
        let ordered = dispatch[before] < dispatch[after]
            || (segment[before] == segment[after] && position[before] < position[after]);
        assert!(
            ordered,
            "task {after} touches a tensor that task {before} only reaches later on the tape",
        );
    });
}
