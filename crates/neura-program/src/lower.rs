use crate::graph::{TaskInfo, ValueInfo};
use crate::shape::Shape;
use neura_abi::{
    KIND_MATMUL, KIND_SOFTMAX, KIND_SOFTMAX_GRAD, KIND_SUM_CHUNK, KIND_SUM_TO, NO_VALUE, Schedule,
    StepRecord,
};

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Task {
    pub(crate) kind: u32,
    pub(crate) flags: u32,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) slot: u32,
    pub(crate) out: u32,
    pub(crate) inputs: [u32; 3],
    pub(crate) param: f32,
    pub(crate) work: u64,
    pub(crate) in_place: bool,
    pub(crate) chain: Vec<StepRecord>,
    pub(crate) time: u32,
}

impl Task {
    fn span(unit: &TaskInfo, first: u32, count: u32, work: u64) -> Self {
        Self {
            kind: unit.kind,
            flags: unit.flags,
            first,
            count,
            slot: 0,
            out: unit.out,
            inputs: unit.inputs,
            param: unit.param,
            work,
            in_place: unit.in_place,
            chain: unit.chain.clone(),
            time: 0,
        }
    }

    pub(crate) fn reads(&self) -> impl Iterator<Item = u32> + '_ {
        self.inputs
            .iter()
            .copied()
            .chain(self.chain.iter().map(|step| step.operand))
            .filter(|value| *value != NO_VALUE)
    }
}

pub(crate) struct Plan {
    pub(crate) values: Vec<ValueInfo>,
    pub(crate) tasks: Vec<Task>,
}

pub(crate) fn lower(values: &[ValueInfo], units: &[TaskInfo], schedule: Schedule) -> Plan {
    let mut plan = Plan {
        values: values.to_vec(),
        tasks: Vec::new(),
    };
    for unit in units {
        schedule_unit(&mut plan, unit, schedule);
    }
    for (time, task) in plan.tasks.iter_mut().enumerate() {
        task.time = time as u32;
    }
    plan
}

impl Plan {
    fn shape(&self, value: u32) -> Shape {
        self.values[value as usize].shape
    }

    fn publish(&mut self, shape: Shape) -> u32 {
        let id = self.values.len() as u32;
        self.values.push(ValueInfo::derived(shape, id));
        id
    }
}

fn schedule_unit(plan: &mut Plan, unit: &TaskInfo, schedule: Schedule) {
    match unit.kind {
        KIND_MATMUL => {
            let out = plan.shape(unit.out);
            let tile = schedule.matmul();
            let column_blocks = out.columns().div_ceil(tile.columns());
            for tile_index in 0..out.rows().div_ceil(tile.rows()) * column_blocks {
                plan.tasks
                    .push(Task::span(unit, tile_index, 1, tile.tile_work()));
            }
        }
        KIND_SOFTMAX | KIND_SOFTMAX_GRAD => {
            let out = plan.shape(unit.out);
            for (first, count) in spans(out.rows(), schedule.rows_per_task()) {
                plan.tasks.push(Task::span(
                    unit,
                    first,
                    count,
                    u64::from(count) * u64::from(out.columns()),
                ));
            }
        }
        KIND_SUM_CHUNK => reduce(plan, unit, schedule),
        KIND_SUM_TO => {
            let out = plan.shape(unit.out);
            let source = plan.shape(unit.inputs[0]);
            let replicas = (0..4)
                .map(|axis| u64::from(source.dims()[axis]) / u64::from(out.dims()[axis]))
                .product::<u64>();
            for (first, count) in spans(out.elements(), schedule.elements_per_task()) {
                plan.tasks
                    .push(Task::span(unit, first, count, u64::from(count) * replicas));
            }
        }
        _ => {
            let out = plan.shape(unit.out);
            for (first, count) in spans(out.elements(), schedule.elements_per_task()) {
                plan.tasks
                    .push(Task::span(unit, first, count, u64::from(count)));
            }
        }
    }
}

fn reduce(plan: &mut Plan, unit: &TaskInfo, schedule: Schedule) {
    let mut source = unit.inputs[0];
    loop {
        let elements = plan.shape(source).elements();
        if elements <= schedule.elements_per_reduction() {
            let mut task = Task::span(unit, 0, elements, u64::from(elements));
            task.inputs = [source, NO_VALUE, NO_VALUE];
            plan.tasks.push(task);
            return;
        }
        let chunks = elements.div_ceil(schedule.elements_per_reduction());
        let partials = plan.publish(Shape::vector(chunks));
        for (slot, (first, count)) in spans(elements, schedule.elements_per_reduction()).enumerate()
        {
            let mut task = Task::span(unit, first, count, u64::from(count));
            task.inputs = [source, NO_VALUE, NO_VALUE];
            task.slot = slot as u32;
            task.out = partials;
            plan.tasks.push(task);
        }
        source = partials;
    }
}

fn spans(units: u32, per_task: u32) -> impl Iterator<Item = (u32, u32)> {
    let mut first = 0;
    std::iter::from_fn(move || {
        if first >= units {
            return None;
        }
        let count = per_task.min(units - first);
        let span = (first, count);
        first += count;
        Some(span)
    })
}
