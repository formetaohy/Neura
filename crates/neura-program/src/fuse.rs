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
            for out in task.writes() {
                fold.producers[out as usize] = Some(index);
                fold.write_times[access::storage(values, out) as usize].push(index as u32);
            }
            for (slot, value) in task.inputs.iter().enumerate() {
                if *value != NO_VALUE {
                    fold.reads[*value as usize].push(Use {
                        task: index,
                        slot: slot as u32,
                    });
                }
            }
            if task.origin != NO_VALUE {
                fold.reads[task.origin as usize].push(Use {
                    task: index,
                    slot: CHAIN_SLOT,
                });
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
        let mut carried = value;
        loop {
            if pinned(self.values, carried) {
                return;
            }
            let Some(producer) = self.producers[carried as usize] else {
                return;
            };
            let uses = &self.reads[carried as usize];
            if uses.len() != 1 {
                return;
            }
            let use_ = uses[0];
            if use_.slot == CHAIN_SLOT {
                return;
            }
            let consumer = use_.task;
            let opens = self.tasks[consumer]
                .as_ref()
                .is_some_and(|tail| tail.kind.takes_prelude());
            if opens {
                if use_.slot != 0 {
                    return;
                }
                match self.prepend(producer, consumer) {
                    Some(source) => carried = source,
                    None => return,
                }
                continue;
            }
            if use_.slot > 1 {
                return;
            }
            self.absorb_on_the_way_out(producer, consumer, use_.slot, carried);
            return;
        }
    }

    fn prepend(&mut self, producer: usize, consumer: usize) -> Option<u32> {
        let head = self.tasks[producer].as_ref()?.clone();
        let tail = self.tasks[consumer].as_ref()?;
        if head.in_place || tail.in_place || !head.prelude.is_empty() {
            return None;
        }
        if head.extra != NO_VALUE {
            return None;
        }
        let shape = self.values[head.out as usize].shape;
        let (source, step) = match head.kind {
            Kind::Unary => {
                let source = head.inputs[0];
                if source == NO_VALUE {
                    return None;
                }
                (
                    source,
                    StepRecord::of(StepFields {
                        op: head.op,
                        operand: NO_VALUE,
                        swapped: 0,
                    }),
                )
            }
            Kind::Binary => {
                let (left, right) = (head.inputs[0], head.inputs[1]);
                if left == NO_VALUE || right == NO_VALUE {
                    return None;
                }
                if self.values[left as usize].shape == shape {
                    (
                        left,
                        StepRecord::of(StepFields {
                            op: head.op,
                            operand: right,
                            swapped: 0,
                        }),
                    )
                } else if self.values[right as usize].shape == shape {
                    (
                        right,
                        StepRecord::of(StepFields {
                            op: head.op,
                            operand: left,
                            swapped: 1,
                        }),
                    )
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        let retired = access::storage(self.values, head.out);
        if self.readers[retired as usize] != 1 {
            return None;
        }
        if tail
            .inputs
            .iter()
            .skip(1)
            .copied()
            .filter(|operand| *operand != NO_VALUE)
            .chain(tail.chain.iter().map(|step| step.operand))
            .any(|operand| access::storage(self.values, operand) == retired)
        {
            return None;
        }
        let (from, to) = (self.times[producer], self.times[consumer]);
        for storage in Access::of(self.values, &head).reads() {
            if *storage == retired {
                return None;
            }
            if self.write_times[*storage as usize]
                .iter()
                .any(|write| *write > from && *write < to)
            {
                return None;
            }
        }
        let mut tail = self.tasks[consumer]
            .take()
            .expect("a consumer is live while its prelude grows");
        for operand in head.inputs {
            if operand != NO_VALUE {
                drop_use(&mut self.reads, operand, producer);
            }
        }
        for step in &head.chain {
            if step.operand != NO_VALUE {
                drop_use(&mut self.reads, step.operand, producer);
            }
        }
        tail.prelude = head
            .prelude
            .iter()
            .copied()
            .chain(std::iter::once(step))
            .chain(head.chain.iter().copied())
            .chain(tail.prelude.iter().copied())
            .collect();
        tail.inputs[0] = source;
        self.reads[source as usize].push(Use {
            task: consumer,
            slot: 0,
        });
        if step.operand != NO_VALUE {
            self.reads[step.operand as usize].push(Use {
                task: consumer,
                slot: CHAIN_SLOT,
            });
        }
        for step in &head.chain {
            if step.operand != NO_VALUE {
                self.reads[step.operand as usize].push(Use {
                    task: consumer,
                    slot: CHAIN_SLOT,
                });
            }
        }
        self.tasks[consumer] = Some(tail);
        self.tasks[producer] = None;
        self.readers[retired as usize] -= 1;
        self.reads[head.out as usize].clear();
        self.producers[head.out as usize] = None;
        Some(source)
    }

    fn absorb_on_the_way_out(&mut self, producer: usize, consumer: usize, slot: u32, value: u32) {
        let Some((fused, step)) = self.merge(producer, consumer, slot) else {
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
            || head.extra != NO_VALUE
            || !head.kind.takes_chain()
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
            for out in authored[at].writes() {
                authored_writer[access::storage(values, out) as usize] = out;
            }
            at += 1;
        }
        for storage in Access::of(values, task).reads() {
            assert_eq!(
                tape_writer[*storage as usize], authored_writer[*storage as usize],
                "task {index} of the tape reads tensor {storage} past the task that rewrites it",
            );
        }
        for out in task.writes() {
            tape_writer[access::storage(values, out) as usize] = out;
        }
    }
}
