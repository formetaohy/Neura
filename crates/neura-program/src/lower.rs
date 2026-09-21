use crate::graph::{TaskInfo, ValueInfo};
use crate::shape::Shape;
use neura_abi::{Kind, MAX_RANK, MatmulTile, NO_VALUE, Profile, StepRecord, Window, strategy};

const TARGET_TASKS: u32 = 256;
const TASK_ELEMENTS_FLOOR: u32 = 2048;
const TASK_ELEMENTS_CEILING: u32 = 65536;
const REDUCTION_FLOOR: u32 = 8192;
const REDUCTION_CEILING: u32 = 65536;
const SOFTMAX_ROW_CEILING: u32 = 8;
const FOLD_ROW_CEILING: u32 = 8;
const MATMUL_TILES_FLOOR: u32 = 128;

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Task {
    pub(crate) kind: Kind,
    pub(crate) op: u32,
    pub(crate) geometry: u32,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) slot: u32,
    pub(crate) out: u32,
    pub(crate) inputs: [u32; 3],
    pub(crate) param: f32,
    pub(crate) window: Window,
    pub(crate) work: u64,
    pub(crate) in_place: bool,
    pub(crate) chain: Vec<StepRecord>,
    pub(crate) time: u32,
}

impl Task {
    fn span(unit: &TaskInfo, first: u32, count: u32, work: u64) -> Self {
        Self {
            kind: unit.kind,
            op: unit.op,
            geometry: 0,
            first,
            count,
            slot: unit.slot,
            out: unit.out,
            inputs: unit.inputs,
            param: unit.param,
            window: unit.window,
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

pub(crate) fn lower(values: &[ValueInfo], units: &[TaskInfo], profile: Profile) -> Plan {
    let mut plan = Plan {
        values: values.to_vec(),
        tasks: Vec::new(),
    };
    for unit in units {
        schedule_unit(&mut plan, unit, profile);
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

fn schedule_unit(plan: &mut Plan, unit: &TaskInfo, profile: Profile) {
    match unit.kind {
        Kind::Matmul => {
            let dims = plan.shape(unit.out).dims();
            let geometry = matmul_geometry(profile, dims[2], dims[3]);
            let tile = profile.ladder()[geometry as usize];
            let planes = dims[0] * dims[1];
            let row_blocks = dims[2].div_ceil(tile.rows());
            let column_blocks = dims[3].div_ceil(tile.columns());
            for tile_index in 0..planes * row_blocks * column_blocks {
                let mut task = Task::span(unit, tile_index, 1, tile.tile_work());
                task.geometry = geometry;
                plan.tasks.push(task);
            }
        }
        Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad => {
            let out = plan.shape(unit.out);
            for (first, count) in spans(out.rows(), softmax_rows_per_task(out.rows())) {
                plan.tasks.push(Task::span(
                    unit,
                    first,
                    count,
                    u64::from(count) * u64::from(out.columns()),
                ));
            }
        }
        Kind::Argmax | Kind::Categorical => choice(plan, unit, profile),
        Kind::SumChunk => reduce(plan, unit),
        Kind::SumAxis => fold(plan, unit, profile),
        Kind::Conv2d => {
            let out = plan.shape(unit.out);
            let filter = plan.shape(unit.inputs[1]);
            let dims = filter.dims();
            let taps = u64::from(dims[1] * dims[2] * dims[3]);
            for (first, count) in spans(out.elements(), task_elements(out.elements())) {
                plan.tasks
                    .push(Task::span(unit, first, count, u64::from(count) * taps));
            }
        }
        Kind::Conv2dInputGrad => {
            let out = plan.shape(unit.out);
            let filter = plan.shape(unit.inputs[0]);
            let dims = filter.dims();
            let taps = u64::from(dims[0] * dims[2] * dims[3]);
            for (first, count) in spans(out.elements(), task_elements(out.elements())) {
                plan.tasks
                    .push(Task::span(unit, first, count, u64::from(count) * taps));
            }
        }
        Kind::Conv2dWeightGrad => conv_weight_grad(plan, unit),
        Kind::Binary
        | Kind::Unary
        | Kind::Partial
        | Kind::Fill
        | Kind::Broadcast
        | Kind::OneHot
        | Kind::Gather => {
            let out = plan.shape(unit.out);
            for (first, count) in spans(out.elements(), task_elements(out.elements())) {
                plan.tasks
                    .push(Task::span(unit, first, count, u64::from(count)));
            }
        }
    }
}

fn conv_weight_grad(plan: &mut Plan, unit: &TaskInfo) {
    let input = plan.shape(unit.inputs[0]);
    let gradient = plan.shape(unit.inputs[1]);
    let filters = plan.shape(unit.inputs[2]).elements();
    let positions = input.dims()[0] * gradient.dims()[2] * gradient.dims()[3];
    let per_task = task_elements(filters);
    let spans_per_chunk = filters.div_ceil(per_task);
    let chunks = (TARGET_TASKS / spans_per_chunk).clamp(1, positions);
    let partials = plan.publish(Shape::of([1, 1, chunks, filters]));
    for chunk in 0..chunks {
        for (first, count) in spans(filters, per_task) {
            let mut task = Task::span(
                unit,
                first,
                count,
                u64::from(count) * u64::from(positions.div_ceil(chunks)),
            );
            task.geometry = strategy::WEIGHT_CHUNK;
            task.slot = chunk;
            task.inputs = [unit.inputs[0], unit.inputs[1], unit.inputs[2]];
            task.out = partials;
            plan.tasks.push(task);
        }
    }
    for (first, count) in spans(filters, per_task) {
        let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(chunks));
        task.geometry = strategy::WEIGHT_FOLD;
        task.inputs = [partials, NO_VALUE, NO_VALUE];
        plan.tasks.push(task);
    }
}

fn choice(plan: &mut Plan, unit: &TaskInfo, profile: Profile) {
    let out = plan.shape(unit.out);
    let columns = plan.shape(unit.inputs[0]).columns();
    let rows = out.rows();
    let geometry = if columns <= profile.workgroup() {
        strategy::THREAD_ROW
    } else {
        strategy::WORKGROUP_ROW
    };
    for (first, count) in spans(rows, choice_rows_per_task(rows)) {
        let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(columns));
        task.geometry = geometry;
        plan.tasks.push(task);
    }
}

fn matmul_geometry(profile: Profile, rows: u32, columns: u32) -> u32 {
    profile
        .ladder()
        .iter()
        .rposition(|tile| {
            rows.is_multiple_of(tile.rows())
                && columns.is_multiple_of(tile.columns())
                && matmul_tiles(tile, rows, columns) >= MATMUL_TILES_FLOOR
        })
        .unwrap_or(0) as u32
}

fn matmul_tiles(tile: &MatmulTile, rows: u32, columns: u32) -> u32 {
    rows.div_ceil(tile.rows()) * columns.div_ceil(tile.columns())
}

fn reduce(plan: &mut Plan, unit: &TaskInfo) {
    let mut source = unit.inputs[0];
    loop {
        let elements = plan.shape(source).elements();
        let per_reduction = reduction_elements(elements);
        if elements <= per_reduction {
            let mut task = Task::span(unit, 0, elements, u64::from(elements));
            task.inputs = [source, NO_VALUE, NO_VALUE];
            plan.tasks.push(task);
            return;
        }
        let chunks = elements.div_ceil(per_reduction);
        let partials = plan.publish(Shape::vector(chunks));
        for (slot, (first, count)) in spans(elements, per_reduction).enumerate() {
            let mut task = Task::span(unit, first, count, u64::from(count));
            task.inputs = [source, NO_VALUE, NO_VALUE];
            task.slot = slot as u32;
            task.out = partials;
            plan.tasks.push(task);
        }
        source = partials;
    }
}

fn task_elements(elements: u32) -> u32 {
    (elements / TARGET_TASKS)
        .next_power_of_two()
        .clamp(TASK_ELEMENTS_FLOOR, TASK_ELEMENTS_CEILING)
}

fn fold(plan: &mut Plan, unit: &TaskInfo, profile: Profile) {
    let (shape, strides) = {
        let source = &plan.values[unit.inputs[0] as usize];
        (source.shape, source.strides)
    };
    let axis = unit.slot;
    let folds = shape.dims()[axis as usize];
    let out = plan.shape(unit.out);
    if axis == MAX_RANK - 1 && strides == shape.strides() {
        let columns = shape.columns();
        let geometry = if columns <= profile.workgroup() {
            strategy::THREAD_ROW
        } else {
            strategy::WORKGROUP_ROW
        };
        for (first, count) in spans(out.elements(), fold_rows_per_task(out.elements())) {
            let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(columns));
            task.geometry = geometry;
            plan.tasks.push(task);
        }
        return;
    }
    for (first, count) in spans(out.elements(), task_elements(out.elements())) {
        let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(folds));
        task.geometry = strategy::THREAD_ELEMENT;
        plan.tasks.push(task);
    }
}

fn reduction_elements(elements: u32) -> u32 {
    (elements / TARGET_TASKS)
        .next_power_of_two()
        .clamp(REDUCTION_FLOOR, REDUCTION_CEILING)
}

fn softmax_rows_per_task(rows: u32) -> u32 {
    rows.div_ceil(TARGET_TASKS).clamp(1, SOFTMAX_ROW_CEILING)
}

fn fold_rows_per_task(rows: u32) -> u32 {
    rows.div_ceil(TARGET_TASKS).clamp(1, FOLD_ROW_CEILING)
}

fn choice_rows_per_task(rows: u32) -> u32 {
    rows.div_ceil(TARGET_TASKS).max(1)
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
