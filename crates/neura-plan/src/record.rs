use crate::access::Reads;
use crate::region::Walk;
use crate::span::{Extents, Split};
use neura_abi::{Kind, NO_VALUE, StepRecord, TaskRecord, split};
use neura_graph::Window;
use std::mem::size_of;

#[derive(Clone, Copy)]
pub(crate) struct Recorded<'a> {
    record: TaskRecord,
    steps: &'a [u8],
}

impl<'a> Recorded<'a> {
    pub(crate) fn with(record: TaskRecord, steps: &'a [u8]) -> Self {
        Self { record, steps }
    }

    pub(crate) fn of(tasks: &'a [u8], steps: &'a [u8]) -> impl Iterator<Item = Self> + 'a {
        assert!(
            tasks.len().is_multiple_of(size_of::<TaskRecord>()),
            "the records of a plan carry {} bytes of tasks, and a task record holds {}",
            tasks.len(),
            size_of::<TaskRecord>(),
        );
        (0..tasks.len() / size_of::<TaskRecord>()).map(move |at| Self {
            record: read::<TaskRecord>(tasks, at),
            steps,
        })
    }

    pub(crate) fn walked(&self, extents: &Extents, bound: &[u32]) -> (u32, u32) {
        match self.split() {
            Split::Range { first, count } => (first, count),
            split => extents.span(split, bound),
        }
    }

    pub(crate) fn record(&self) -> TaskRecord {
        self.record
    }

    pub(crate) fn split(&self) -> Split {
        let record = &self.record;
        match record.split {
            split::RANGE => Split::Range {
                first: record.first,
                count: record.count,
            },
            split::UNIFORM => Split::Uniform {
                measure: record.measure,
                index: record.index,
                group: record.group,
            },
            split::PLANE => Split::Plane {
                measure: record.measure,
                planes: record.planes,
                plane: record.plane,
                index: record.index,
                group: record.group,
            },
            split::SEGMENT => Split::Segment {
                measure: record.measure,
                plane: record.plane,
                index: record.index,
                group: record.group,
            },
            split::RAGGED => Split::Ragged {
                planes: record.planes,
                plane: record.plane,
                index: record.index,
                group: record.group,
            },
            other => panic!("task record {other} names no split of a task"),
        }
    }

    fn window(&self) -> Window {
        Window::new(
            [self.record.reach_rows, self.record.reach_columns],
            [self.record.stride_rows, self.record.stride_columns],
            [self.record.pad_rows, self.record.pad_columns],
        )
    }

    fn step(&self, at: u32) -> StepRecord {
        read::<StepRecord>(self.steps, at as usize)
    }
}

fn read<T: bytemuck::Pod>(bytes: &[u8], at: usize) -> T {
    let start = at * size_of::<T>();
    bytemuck::pod_read_unaligned(&bytes[start..start + size_of::<T>()])
}

impl Reads for Recorded<'_> {
    fn out(&self) -> u32 {
        self.record.out
    }

    fn extra(&self) -> u32 {
        self.record.extra
    }

    fn in_place(&self) -> bool {
        self.record.in_place != 0
    }

    fn reads(&self) -> impl Iterator<Item = u32> {
        let record = self.record;
        [
            record.a,
            record.b,
            record.c,
            record.d,
            record.e,
            record.f,
            record.origin,
            record.segment,
            record.queries,
        ]
        .into_iter()
        .chain(self.prelude())
        .chain(self.chain())
        .filter(|value| *value != NO_VALUE)
    }
}

impl Walk for Recorded<'_> {
    fn kind(&self) -> Kind {
        Kind::of(self.record.kind)
    }

    fn geometry(&self) -> u32 {
        self.record.geometry
    }

    fn splits(&self) -> u32 {
        self.record.splits
    }

    fn slot(&self) -> u32 {
        self.record.slot
    }

    fn input(&self, slot: usize) -> u32 {
        match slot {
            0 => self.record.a,
            1 => self.record.b,
            2 => self.record.c,
            3 => self.record.d,
            4 => self.record.e,
            _ => self.record.f,
        }
    }

    fn prelude(&self) -> impl Iterator<Item = u32> {
        (0..self.record.prelude_steps).map(|step| self.step(self.record.prelude + step).operand)
    }

    fn chain(&self) -> impl Iterator<Item = u32> {
        (0..self.record.steps).map(|step| self.step(self.record.chain + step).operand)
    }

    fn depends(&self) -> impl Iterator<Item = u32> {
        std::iter::empty()
    }

    fn split(&self) -> Split {
        self.split()
    }
}

pub(crate) fn assert_records_hold_the_tasks_they_carry(
    tasks: &[super::lower::Task],
    order: &[u32],
    records: &[u8],
    steps: &[u8],
) {
    let mut decoded = Recorded::of(records, steps);
    for (position, index) in order.iter().enumerate() {
        let task = &tasks[*index as usize];
        let found = decoded
            .next()
            .expect("the records of a plan carry one task per position of its order");
        if let Some(mismatch) = mismatched(&found, task) {
            panic!(
                "task {index} stands at position {position} of the plan, and the record there walks another task: {mismatch}",
            );
        }
    }
    assert!(
        decoded.next().is_none(),
        "the records of a plan carry no task beyond its order",
    );
}

fn mismatched(found: &Recorded<'_>, task: &super::lower::Task) -> Option<&'static str> {
    let record = &found.record;
    if found.kind() != task.kind {
        return Some("kind");
    }
    if record.geometry != task.geometry {
        return Some("geometry");
    }
    if record.splits != task.splits {
        return Some("splits");
    }
    if record.slot != task.slot {
        return Some("slot");
    }
    if record.out != task.out {
        return Some("out");
    }
    if record.extra != task.extra {
        return Some("extra");
    }
    if record.origin != task.origin {
        return Some("origin");
    }
    if record.axis != task.axis {
        return Some("axis");
    }
    if record.offset != task.offset {
        return Some("offset");
    }
    if record.segment != task.segments {
        return Some("segment");
    }
    if record.queries != task.queries {
        return Some("queries");
    }
    if record.plane != task.plane {
        return Some("plane");
    }
    if record.keys != task.keys {
        return Some("keys");
    }
    if record.tokens != task.tokens {
        return Some("tokens");
    }
    if record.grid != task.grid {
        return Some("grid");
    }
    if record.patch != task.patch {
        return Some("patch");
    }
    if record.in_place != u32::from(task.in_place) {
        return Some("in_place");
    }
    if record.first != task.first {
        return Some("first");
    }
    if record.count != task.count {
        return Some("count");
    }
    if record.param.to_bits() != task.param.to_bits() {
        return Some("param");
    }
    if found.window() != task.window {
        return Some("window");
    }
    match task.split {
        Split::Range { .. } => {
            if (record.first, record.count) != (task.first, task.count) {
                return Some("range");
            }
        }
        split => {
            if found.split() != split {
                return Some("split");
            }
        }
    }
    if (0..6).any(|slot| found.input(slot) != task.inputs[slot]) {
        return Some("inputs");
    }
    if !found
        .prelude()
        .eq(task.prelude.iter().map(|step| step.operand))
    {
        return Some("prelude");
    }
    if !found.chain().eq(task.chain.iter().map(|step| step.operand)) {
        return Some("chain");
    }
    None
}
