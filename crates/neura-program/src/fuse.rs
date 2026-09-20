use crate::graph::{GraphState, Residency, TaskInfo};
use neura_abi::{KIND_BINARY, KIND_SUM_CHUNK, KIND_UNARY, NO_VALUE, StepRecord, chain_op};

const CHAIN_SLOT: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Use {
    task: usize,
    slot: u32,
}

struct Group {
    consumers: Vec<usize>,
    step: StepRecord,
    chain: Vec<StepRecord>,
    out: u32,
    in_place: bool,
    work: u64,
    time: u32,
}

pub(crate) fn fuse(state: &GraphState) -> Vec<TaskInfo> {
    let mut tasks = state
        .tasks
        .iter()
        .cloned()
        .map(Some)
        .collect::<Vec<Option<TaskInfo>>>();
    let mut producers = vec![Vec::new(); state.values.len()];
    let mut reads = vec![Vec::new(); state.values.len()];
    for (time, task) in tasks.iter_mut().enumerate() {
        let task = task
            .as_mut()
            .expect("a task survives until the pass reaches the value it writes");
        task.time = time as u32;
        producers[task.out as usize].push(time);
        for (slot, value) in task.inputs.iter().enumerate() {
            if *value != NO_VALUE {
                reads[*value as usize].push(Use {
                    task: time,
                    slot: slot as u32,
                });
            }
        }
    }
    for value in 0..state.values.len() {
        absorb(state, &mut tasks, &mut producers, &mut reads, value as u32);
    }
    order_by_time(tasks.into_iter().flatten().collect())
}

fn order_by_time(mut tasks: Vec<TaskInfo>) -> Vec<TaskInfo> {
    tasks.sort_by_key(|task| task.time);
    tasks
}

fn absorb(
    state: &GraphState,
    tasks: &mut [Option<TaskInfo>],
    producers: &mut [Vec<usize>],
    reads: &mut [Vec<Use>],
    value: u32,
) {
    if producers[value as usize].is_empty() || pinned(state, value) {
        return;
    }
    if producers[value as usize]
        .iter()
        .any(|producer| !chainable(live(tasks, *producer).kind))
    {
        return;
    }
    let Some(group) = group_of(state, tasks, reads, value) else {
        return;
    };
    let fused = producers[value as usize].clone();
    for producer in &fused {
        let task = tasks[*producer]
            .as_mut()
            .expect("a producer survives the pass that absorbs its value");
        task.time = group.time;
        task.out = group.out;
        task.in_place = group.in_place;
        task.work += group.work;
        task.chain.push(group.step);
        task.chain.extend(group.chain.iter().copied());
        for step in std::iter::once(&group.step).chain(group.chain.iter()) {
            if step.operand != NO_VALUE {
                reads[step.operand as usize].push(Use {
                    task: *producer,
                    slot: CHAIN_SLOT,
                });
            }
        }
    }
    for consumer in &group.consumers {
        let task = tasks[*consumer]
            .take()
            .expect("a consumer is live while it is absorbed");
        for value in task.inputs {
            if value != NO_VALUE {
                drop_use(reads, value, *consumer);
            }
        }
        for step in &task.chain {
            if step.operand != NO_VALUE {
                drop_use(reads, step.operand, *consumer);
            }
        }
    }
    producers[group.out as usize].retain(|index| !group.consumers.contains(index));
    producers[group.out as usize].extend(fused);
    producers[value as usize].clear();
    reads[value as usize].clear();
}

fn group_of(
    state: &GraphState,
    tasks: &[Option<TaskInfo>],
    reads: &[Vec<Use>],
    value: u32,
) -> Option<Group> {
    let uses = &reads[value as usize];
    if uses.is_empty() || uses.iter().any(|use_| use_.slot == CHAIN_SLOT) {
        return None;
    }
    let head = tasks[uses[0].task].as_ref()?;
    if !absorbable(head.kind)
        || state.values[head.out as usize].shape != state.values[value as usize].shape
    {
        return None;
    }
    let operand = match head.kind {
        KIND_UNARY => NO_VALUE,
        KIND_BINARY => {
            let operand = *head.inputs.get((uses[0].slot ^ 1) as usize)?;
            if uses[0].slot > 1 || operand == NO_VALUE {
                return None;
            }
            operand
        }
        _ => return None,
    };
    for use_ in uses {
        let task = tasks[use_.task].as_ref()?;
        if use_.slot != uses[0].slot
            || task.kind != head.kind
            || task.flags != head.flags
            || task.out != head.out
            || task.in_place != head.in_place
            || task.chain != head.chain
            || task.inputs.iter().filter(|input| **input == value).count() != 1
        {
            return None;
        }
        if head.kind == KIND_BINARY && *task.inputs.get((use_.slot ^ 1) as usize)? != operand {
            return None;
        }
    }
    Some(Group {
        consumers: uses.iter().map(|use_| use_.task).collect(),
        step: StepRecord {
            op: chain_op(head.kind, head.flags),
            operand,
        },
        chain: head.chain.clone(),
        out: head.out,
        in_place: head.in_place,
        work: head.work,
        time: head.time,
    })
}

fn live(tasks: &[Option<TaskInfo>], index: usize) -> &TaskInfo {
    tasks[index]
        .as_ref()
        .expect("a task that writes a taped value is still on the tape")
}

fn absorbable(kind: u32) -> bool {
    matches!(kind, KIND_BINARY | KIND_UNARY)
}

fn chainable(kind: u32) -> bool {
    kind != KIND_SUM_CHUNK
}

fn pinned(state: &GraphState, value: u32) -> bool {
    let info = &state.values[value as usize];
    info.retained || matches!(info.residency, Residency::Input | Residency::Parameter)
}

fn drop_use(reads: &mut [Vec<Use>], value: u32, task: usize) {
    reads[value as usize].retain(|use_| use_.task != task);
}
