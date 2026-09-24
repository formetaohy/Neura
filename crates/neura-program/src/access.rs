use crate::graph::ValueInfo;

pub(crate) trait Reads {
    fn out(&self) -> u32;
    fn in_place(&self) -> bool;
    fn reads(&self) -> impl Iterator<Item = u32> + '_;
}

pub(crate) fn storage(values: &[ValueInfo], value: u32) -> u32 {
    values[value as usize].storage
}

pub(crate) fn owner(values: &[ValueInfo], value: u32) -> bool {
    storage(values, value) == value
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Access {
    reads: Vec<u32>,
    write: u32,
    in_place: bool,
}

impl Access {
    pub(crate) fn of(values: &[ValueInfo], task: &impl Reads) -> Self {
        let mut reads = task
            .reads()
            .map(|value| storage(values, value))
            .collect::<Vec<u32>>();
        reads.sort_unstable();
        reads.dedup();
        Self {
            reads,
            write: storage(values, task.out()),
            in_place: task.in_place(),
        }
    }

    pub(crate) fn reads(&self) -> &[u32] {
        &self.reads
    }

    pub(crate) fn write(&self) -> u32 {
        self.write
    }

    pub(crate) fn in_place(&self) -> bool {
        self.in_place
    }
}
