use crate::access::Reads;
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
const MATMUL_SPLITS_CEILING: u32 = 64;
const MATMUL_SPLIT_BLOCKS: u32 = 4;
const MATMUL_PARTIALS_CEILING: u32 = 1 << 20;

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
    pub(crate) splits: u32,
    pub(crate) work: u64,
    pub(crate) in_place: bool,
    pub(crate) chain: Vec<StepRecord>,
    pub(crate) unit: u32,
}

impl Reads for Task {
    fn out(&self) -> u32 {
        self.out
    }

    fn in_place(&self) -> bool {
        self.in_place
    }

    fn reads(&self) -> impl Iterator<Item = u32> + '_ {
        self.inputs
            .iter()
            .copied()
            .chain(self.chain.iter().map(|step| step.operand))
            .filter(|value| *value != NO_VALUE)
    }
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
            splits: 1,
            work,
            in_place: unit.in_place,
            chain: unit.chain.clone(),
            unit: 0,
        }
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
    for (unit, task) in units.iter().enumerate() {
        let mark = plan.tasks.len();
        schedule_unit(&mut plan, task, profile);
        for task in &mut plan.tasks[mark..] {
            task.unit = unit as u32;
        }
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
        Kind::Matmul => matmul(plan, unit, profile),
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
        Kind::MatmulFold => {
            panic!("a fold of depth partials comes from the product whose depth split")
        }
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
        Kind::Scatter => {
            let rows = plan.shape(unit.inputs[1]).elements();
            let width = plan.shape(unit.out).columns();
            for (first, count) in spans(rows, scatter_rows_per_task(rows)) {
                plan.tasks.push(Task::span(
                    unit,
                    first,
                    count,
                    u64::from(count) * u64::from(width),
                ));
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

fn matmul(plan: &mut Plan, unit: &TaskInfo, profile: Profile) {
    let dims = plan.shape(unit.out).dims();
    let (rows, columns) = (dims[2], dims[3]);
    let depth = plan.shape(unit.inputs[0]).dims()[3];
    let planes = dims[0] * dims[1];
    let tile = matmul_tile(profile, rows, columns);
    let geometry = profile
        .tiles()
        .iter()
        .position(|candidate| *candidate == tile)
        .expect("every tile a plan chooses lies in the profile that chose it")
        as u32;
    let tiles = planes * rows.div_ceil(tile.rows()) * columns.div_ceil(tile.columns());
    let splits = matmul_splits(tile, rows, columns, depth, planes);
    let partials =
        (splits > 1).then(|| plan.publish(Shape::vector(splits * planes * rows * columns)));
    for split in 0..splits {
        for index in 0..tiles {
            let mut task = Task::span(unit, index, 1, tile.tile_work());
            task.geometry = geometry;
            if let Some(partials) = partials {
                task.out = partials;
                task.slot = split;
                task.splits = splits;
                task.chain.clear();
                task.in_place = false;
            }
            plan.tasks.push(task);
        }
    }
    let Some(partials) = partials else {
        return;
    };
    let elements = planes * rows * columns;
    for (first, count) in spans(elements, task_elements(elements)) {
        let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(splits));
        task.kind = Kind::MatmulFold;
        task.inputs = [partials, NO_VALUE, NO_VALUE];
        task.splits = splits;
        plan.tasks.push(task);
    }
}

fn matmul_tile(profile: Profile, rows: u32, columns: u32) -> MatmulTile {
    let mut chosen = profile.tiles()[0];
    let mut cheapest = u128::MAX;
    for tile in profile.tiles() {
        let cost = matmul_cost(*tile, rows, columns);
        if cost < cheapest {
            cheapest = cost;
            chosen = *tile;
        }
    }
    chosen
}

fn matmul_cost(tile: MatmulTile, rows: u32, columns: u32) -> u128 {
    let computed = u128::from(tile.rows())
        * u128::from(rows.div_ceil(tile.rows()))
        * u128::from(tile.columns())
        * u128::from(columns.div_ceil(tile.columns()));
    let staged = u128::from(tile.rows() + tile.columns());
    let loaded =
        u128::from(tile.threads()) * u128::from(tile.register_rows() + tile.register_columns());
    let multiplied = u128::from(tile.threads()) * u128::from(tile.registers());
    computed * (staged + loaded) / multiplied
}

fn matmul_splits(tile: MatmulTile, rows: u32, columns: u32, depth: u32, planes: u32) -> u32 {
    let elements = u64::from(planes) * u64::from(rows) * u64::from(columns);
    let tiles = u64::from(planes)
        * u64::from(rows.div_ceil(tile.rows()))
        * u64::from(columns.div_ceil(tile.columns()));
    let splits = u64::from(TARGET_TASKS).div_ceil(tiles);
    let splits = splits.min(u64::from(
        depth.div_ceil(tile.depth()) / MATMUL_SPLIT_BLOCKS,
    ));
    let splits = splits.min(u64::from(MATMUL_SPLITS_CEILING));
    let splits = splits.min(u64::from(MATMUL_PARTIALS_CEILING) / elements);
    let splits = splits.min(u64::from(depth) * u64::from(rows + columns) / elements);
    splits.max(1) as u32
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

const SCATTER_ROW_CEILING: u32 = 4096;

fn scatter_rows_per_task(rows: u32) -> u32 {
    rows.min(SCATTER_ROW_CEILING)
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
