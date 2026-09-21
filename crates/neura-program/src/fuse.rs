use crate::graph::{GraphState, Residency, TaskInfo};
use neura_abi::{Kind, NO_VALUE, StepRecord};

const CHAIN_SLOT: u32 = u32::MAX;

#[derive(Clone, Copy)]
struct Use {
    task: usize,
    slot: u32,
}

pub(crate) fn fuse(state: &GraphState) -> Vec<TaskInfo> {
    let mut tasks = state
        .tasks
        .iter()
        .cloned()
        .map(Some)
        .collect::<Vec<Option<TaskInfo>>>();
    let mut writers = vec![None; state.values.len()];
    let mut reads = vec![Vec::new(); state.values.len()];
    for (index, task) in tasks.iter_mut().enumerate() {
        let task = task
            .as_mut()
            .expect("a task survives until the pass reaches the value it writes");
        task.time = index as u32;
        writers[task.out as usize] = Some(index);
        for (slot, value) in task.inputs.iter().enumerate() {
            if *value != NO_VALUE {
                reads[*value as usize].push(Use {
                    task: index,
                    slot: slot as u32,
                });
            }
        }
    }
    for value in 0..state.values.len() {
        absorb(state, &mut tasks, &mut writers, &mut reads, value as u32);
    }
    let mut fused = tasks.into_iter().flatten().collect::<Vec<TaskInfo>>();
    fused.sort_by_key(|task| task.time);
    fused
}

fn absorb(
    state: &GraphState,
    tasks: &mut [Option<TaskInfo>],
    writers: &mut [Option<usize>],
    reads: &mut [Vec<Use>],
    value: u32,
) {
    if pinned(state, value) {
        return;
    }
    let Some(producer_index) = writers[value as usize] else {
        return;
    };
    let uses = &reads[value as usize];
    if uses.len() != 1 || uses[0].slot == CHAIN_SLOT || uses[0].slot > 1 {
        return;
    }
    let consumer_index = uses[0].task;
    let (head, step) = {
        let producer = tasks[producer_index]
            .as_ref()
            .expect("a producer survives the pass that absorbs its value");
        let consumer = tasks[consumer_index]
            .as_ref()
            .expect("a consumer is live while it is absorbed");
        if producer.kind == Kind::SumChunk
            || producer.in_place
            || !consumer.kind.chainable()
            || state.values[consumer.out as usize].shape != state.values[value as usize].shape
            || consumer
                .inputs
                .iter()
                .filter(|input| **input == value)
                .count()
                != 1
        {
            return;
        }
        let Some(step) = consumer_step(consumer, value) else {
            return;
        };
        (consumer.clone(), step)
    };
    {
        let producer = tasks[producer_index]
            .as_mut()
            .expect("a producer survives the pass that absorbs its value");
        producer.time = head.time;
        producer.out = head.out;
        producer.in_place = head.in_place;
        producer.chain.push(step);
        producer.chain.extend(head.chain.iter().copied());
    }
    {
        let producer = tasks[producer_index]
            .as_ref()
            .expect("a producer survives the pass that absorbs its value");
        for step in &producer.chain {
            if step.operand != NO_VALUE {
                reads[step.operand as usize].push(Use {
                    task: producer_index,
                    slot: CHAIN_SLOT,
                });
            }
        }
    }
    let consumer = tasks[consumer_index]
        .take()
        .expect("a consumer is live while it is absorbed");
    for operand in consumer.inputs {
        if operand != NO_VALUE {
            drop_use(reads, operand, consumer_index);
        }
    }
    for step in &consumer.chain {
        if step.operand != NO_VALUE {
            drop_use(reads, step.operand, consumer_index);
        }
    }
    reads[value as usize].clear();
    writers[value as usize] = None;
    writers[head.out as usize] = Some(producer_index);
}

fn consumer_step(consumer: &TaskInfo, value: u32) -> Option<StepRecord> {
    match consumer.kind {
        Kind::Unary => Some(StepRecord {
            op: consumer.op,
            operand: NO_VALUE,
            swapped: 0,
        }),
        Kind::Binary => {
            let slot = consumer
                .inputs
                .iter()
                .position(|input| *input == value)
                .expect("an absorbed value is one of the consumer's operands");
            let operand = consumer.inputs[slot ^ 1];
            if operand == NO_VALUE {
                return None;
            }
            Some(StepRecord {
                op: consumer.op,
                operand,
                swapped: u32::from(slot == 1),
            })
        }
        Kind::Matmul
        | Kind::MatmulFold
        | Kind::Partial
        | Kind::Fill
        | Kind::Broadcast
        | Kind::SumChunk
        | Kind::SumAxis
        | Kind::Softmax
        | Kind::SoftmaxGrad
        | Kind::LogSoftmax
        | Kind::LogSoftmaxGrad
        | Kind::Argmax
        | Kind::Categorical
        | Kind::OneHot
        | Kind::Gather
        | Kind::Conv2d
        | Kind::Conv2dInputGrad
        | Kind::Conv2dWeightGrad => None,
    }
}

fn pinned(state: &GraphState, value: u32) -> bool {
    let info = &state.values[value as usize];
    info.retained || matches!(info.residency, Residency::Input | Residency::Parameter)
}

fn drop_use(reads: &mut [Vec<Use>], value: u32, task: usize) {
    reads[value as usize].retain(|use_| use_.task != task);
}
