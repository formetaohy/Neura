use crate::access::{self, Access, Reads};
use neura_abi::{Kind, NO_VALUE, StepFields, StepRecord};
use neura_graph::{GraphSnapshot, Residency, TaskInfo, ValueInfo};

const CHAIN_SLOT: u32 = u32::MAX;

pub(crate) fn fuse(state: &GraphSnapshot) -> Vec<TaskInfo> {
    Fold::of(state.values(), state.tasks()).fold()
}

#[derive(Clone, Copy)]
struct Use {
    task: usize,
    slot: u32,
}

struct Fold<'a> {
    authored: &'a [TaskInfo],
    values: &'a [ValueInfo],
    tasks: Vec<Option<TaskInfo>>,
    times: Vec<u32>,
    producers: Vec<Option<usize>>,
    reads: Vec<Vec<Use>>,
    readers: Vec<u32>,
    write_times: Vec<Vec<u32>>,
}

impl<'a> Fold<'a> {
    fn of(values: &'a [ValueInfo], authored: &'a [TaskInfo]) -> Self {
        let mut fold = Self {
            authored,
            values,
            tasks: authored.iter().cloned().map(Some).collect(),
            times: (0..authored.len() as u32).collect(),
            producers: vec![None; values.len()],
            reads: vec![Vec::new(); values.len()],
            readers: vec![0; values.len()],
            write_times: vec![Vec::new(); values.len()],
        };
        for (index, task) in authored.iter().enumerate() {
            fold.producers[task.out as usize] = Some(index);
            fold.write_times[access::storage(values, task.out) as usize].push(index as u32);
            for (slot, value) in task.inputs.iter().enumerate() {
                if *value != NO_VALUE {
                    fold.reads[*value as usize].push(Use {
                        task: index,
                        slot: slot as u32,
                    });
                }
            }
            for storage in Access::of(values, task).reads() {
                fold.readers[*storage as usize] += 1;
            }
        }
        fold
    }

    fn fold(mut self) -> Vec<TaskInfo> {
        for value in 0..self.values.len() {
            self.absorb(value as u32);
        }
        let mut tape = Vec::with_capacity(self.tasks.len());
        for (task, time) in self.tasks.into_iter().zip(self.times) {
            if let Some(task) = task {
                tape.push((time, task));
            }
        }
        tape.sort_by_key(|(time, _)| *time);
        let (times, tasks): (Vec<u32>, Vec<TaskInfo>) = tape.into_iter().unzip();
        assert_sources(self.values, self.authored, &tasks, &times);
        tasks
    }

    fn absorb(&mut self, value: u32) {
        if pinned(self.values, value) {
            return;
        }
        let Some(producer) = self.producers[value as usize] else {
            return;
        };
        let uses = &self.reads[value as usize];
        if uses.len() != 1 {
            return;
        }
        let use_ = uses[0];
        if use_.slot == CHAIN_SLOT || use_.slot > 1 {
            return;
        }
        let consumer = use_.task;
        let Some((fused, step)) = self.merge(producer, consumer, use_.slot) else {
            return;
        };
        let retired = {
            let producer = self.tasks[producer]
                .as_ref()
                .expect("a producer survives the fold that absorbs its value");
            access::storage(self.values, producer.out)
        };
        let head = self.tasks[consumer]
            .take()
            .expect("a consumer is live while it is absorbed");
        for operand in head.inputs {
            if operand != NO_VALUE {
                drop_use(&mut self.reads, operand, consumer);
            }
        }
        if step.operand != NO_VALUE {
            self.reads[step.operand as usize].push(Use {
                task: producer,
                slot: CHAIN_SLOT,
            });
        }
        self.tasks[producer] = Some(fused);
        self.times[producer] = self.times[consumer];
        self.readers[retired as usize] -= 1;
        self.reads[value as usize].clear();
        self.producers[value as usize] = None;
        self.producers[head.out as usize] = Some(producer);
    }

    fn merge(&self, producer: usize, consumer: usize, slot: u32) -> Option<(TaskInfo, StepRecord)> {
        let head = self.tasks[producer].as_ref()?;
        let tail = self.tasks[consumer].as_ref()?;
        if head.kind == Kind::SumChunk
            || head.in_place
            || !matches!(tail.kind, Kind::Unary | Kind::Binary)
        {
            return None;
        }
        if self.values[head.out as usize].shape != self.values[tail.out as usize].shape {
            return None;
        }
        let retired = access::storage(self.values, head.out);
        if self.readers[retired as usize] != 1 {
            return None;
        }
        let step = consumer_step(tail, slot)?;
        let (from, to) = (self.times[producer], self.times[consumer]);
        for storage in Access::of(self.values, head).reads() {
            if self.write_times[*storage as usize]
                .iter()
                .any(|write| *write > from && *write < to)
            {
                return None;
            }
        }
        let mut fused = head.clone();
        fused.out = tail.out;
        fused.in_place = tail.in_place;
        fused.chain.push(step);
        let written = access::storage(self.values, fused.out);
        for value in fused.reads() {
            let read = access::storage(self.values, value);
            if read == retired {
                return None;
            }
            if read == written
                && self.values[value as usize].strides != self.values[fused.out as usize].strides
            {
                return None;
            }
        }
        Some((fused, step))
    }
}

fn consumer_step(consumer: &TaskInfo, slot: u32) -> Option<StepRecord> {
    match consumer.kind {
        Kind::Unary => Some(StepRecord::of(StepFields {
            op: consumer.op,
            operand: NO_VALUE,
            swapped: 0,
        })),
        Kind::Binary => {
            let operand = consumer.inputs[slot as usize ^ 1];
            if operand == NO_VALUE {
                return None;
            }
            Some(StepRecord::of(StepFields {
                op: consumer.op,
                operand,
                swapped: u32::from(slot == 1),
            }))
        }
        _ => None,
    }
}

fn pinned(values: &[ValueInfo], value: u32) -> bool {
    let info = &values[value as usize];
    info.retained || matches!(info.residency, Residency::Input | Residency::Parameter)
}

fn drop_use(reads: &mut [Vec<Use>], value: u32, task: usize) {
    reads[value as usize].retain(|use_| use_.task != task);
}

fn assert_sources(values: &[ValueInfo], authored: &[TaskInfo], tape: &[TaskInfo], times: &[u32]) {
    let mut authored_writer = vec![NO_VALUE; values.len()];
    let mut tape_writer = vec![NO_VALUE; values.len()];
    let mut at = 0usize;
    for (index, (task, time)) in tape.iter().zip(times).enumerate() {
        while at < *time as usize {
            authored_writer[access::storage(values, authored[at].out) as usize] = authored[at].out;
            at += 1;
        }
        for storage in Access::of(values, task).reads() {
            assert_eq!(
                tape_writer[*storage as usize], authored_writer[*storage as usize],
                "task {index} of the tape reads tensor {storage} past the task that rewrites it",
            );
        }
        tape_writer[access::storage(values, task.out) as usize] = task.out;
    }
}
