use crate::access::Reads;
use crate::product::Product;
use crate::span::{self, Measure, Split};
use neura_abi::{Kind, MAX_RANK, NO_VALUE, StepFields, StepRecord, strategy};
use neura_graph::{Shape, TaskInfo, ValueInfo, Window};
use neura_pointwise as op;
use neura_profile::{AttentionTile, MatmulTile, Profile};

const TASK_ELEMENTS_FLOOR: u32 = 2048;
const TASK_ELEMENTS_CEILING: u32 = 65536;
const REDUCTION_FLOOR: u32 = 8192;
const REDUCTION_CEILING: u32 = 65536;
const SOFTMAX_ROW_CEILING: u32 = 8;
const FOLD_ROW_CEILING: u32 = 8;

#[derive(Clone, PartialEq, Debug)]
pub(crate) struct Task {
    pub(crate) kind: Kind,
    pub(crate) op: u32,
    pub(crate) geometry: u32,
    pub(crate) first: u32,
    pub(crate) count: u32,
    pub(crate) slot: u32,
    pub(crate) out: u32,
    pub(crate) extra: u32,
    pub(crate) inputs: [u32; 6],
    pub(crate) origin: u32,
    pub(crate) param: f32,
    pub(crate) window: Window,
    pub(crate) splits: u32,
    pub(crate) work: u64,
    pub(crate) in_place: bool,
    pub(crate) axis: u32,
    pub(crate) offset: u32,
    pub(crate) prelude: Vec<StepRecord>,
    pub(crate) chain: Vec<StepRecord>,
    pub(crate) unit: u32,
    pub(crate) split: Split,
}

impl Reads for Task {
    fn out(&self) -> u32 {
        self.out
    }

    fn extra(&self) -> u32 {
        self.extra
    }

    fn in_place(&self) -> bool {
        self.in_place
    }

    fn reads(&self) -> impl Iterator<Item = u32> + '_ {
        self.inputs
            .iter()
            .copied()
            .chain([self.origin])
            .chain(self.prelude.iter().map(|step| step.operand))
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
            extra: unit.extra,
            inputs: unit.inputs,
            origin: unit.origin,
            param: unit.param,
            window: unit.window,
            splits: 1,
            work,
            in_place: unit.in_place,
            axis: unit.axis,
            offset: unit.offset,
            prelude: unit.prelude.clone(),
            chain: unit.chain.clone(),
            unit: 0,
            split: Split::Range { first, count },
        }
    }
}

pub(crate) struct Plan {
    pub(crate) values: Vec<ValueInfo>,
    pub(crate) tasks: Vec<Task>,
    pub(crate) tiles: Vec<MatmulTile>,
    pub(crate) products: Vec<Product>,
    pub(crate) attention: Vec<AttentionTile>,
    pub(crate) measures: Vec<Measure>,
    pub(crate) measured: Vec<(u32, u32)>,
}

pub(crate) fn lower(
    values: &[ValueInfo],
    units: &[TaskInfo],
    profile: Profile,
    chosen: &[(Product, MatmulTile)],
) -> Plan {
    let mut plan = Plan {
        values: values.to_vec(),
        tasks: Vec::new(),
        tiles: profile.tiles().to_vec(),
        products: Vec::new(),
        attention: Vec::new(),
        measures: Vec::new(),
        measured: Vec::new(),
    };
    for (unit, task) in units.iter().enumerate() {
        let mark = plan.tasks.len();
        if writes_narrow(&plan.values, task) {
            schedule_narrow(&mut plan, task, profile, chosen);
        } else {
            schedule_unit(&mut plan, task, profile, chosen);
        }
        for task in &mut plan.tasks[mark..] {
            task.unit = unit as u32;
        }
    }
    walk(&mut plan);
    plan
}

fn walk(plan: &mut Plan) {
    for task in &mut plan.tasks {
        if task.kind.geometry() == neura_abi::Geometry::Access {
            task.geometry = if walks_by_index(&plan.values, task) {
                strategy::INDEX
            } else {
                strategy::FRAME
            };
        }
    }
}

fn walks_by_index(values: &[ValueInfo], task: &Task) -> bool {
    let out = values[task.out as usize].shape;
    task.reads().all(|value| {
        let info = &values[value as usize];
        info.strides == [0u32; MAX_RANK as usize]
            || (info.shape == out
                && info.strides == info.shape.strides()
                && info.strides[MAX_RANK as usize - 1] == 1)
    })
}

fn writes_narrow(values: &[ValueInfo], task: &TaskInfo) -> bool {
    values[task.out as usize].element.narrow()
}

fn schedule_narrow(
    plan: &mut Plan,
    unit: &TaskInfo,
    profile: Profile,
    chosen: &[(Product, MatmulTile)],
) {
    if let Some((source, steps)) = pointwise_steps(plan, unit) {
        let tasks = convert(plan, unit, source, steps, profile);
        plan.tasks.extend(tasks);
        return;
    }
    let target = unit.out;
    let image = plan.publish(plan.shape(target));
    if !writes_every_element(unit.kind) {
        let elements = plan.shape(target).elements();
        let mut copy = Task::span(unit, 0, elements, u64::from(elements));
        copy.split = match measured(plan, target, Measure::Elements) {
            Some(measure) => Split::Uniform {
                measure,
                index: 0,
                group: 1,
            },
            None => Split::Range {
                first: 0,
                count: elements,
            },
        };
        copy.kind = Kind::Unary;
        copy.op = op::IDENTITY;
        copy.geometry = 0;
        copy.slot = 0;
        copy.out = image;
        copy.inputs = [target, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
        copy.origin = NO_VALUE;
        copy.extra = NO_VALUE;
        copy.param = 0.0;
        copy.splits = 1;
        copy.in_place = false;
        copy.prelude.clear();
        copy.chain.clear();
        plan.tasks.push(copy);
    }
    schedule_unit(plan, &redirected(unit, image), profile, chosen);
    let tasks = convert(plan, unit, image, Vec::new(), profile);
    plan.tasks.extend(tasks);
}

fn pointwise_steps(plan: &Plan, unit: &TaskInfo) -> Option<(u32, Vec<StepRecord>)> {
    if unit.extra != NO_VALUE || unit.origin != NO_VALUE {
        return None;
    }
    let (source, step) = match unit.kind {
        Kind::Unary => {
            let source = unit.inputs[0];
            if source == NO_VALUE {
                return None;
            }
            (
                source,
                StepRecord::of(StepFields {
                    op: unit.op,
                    operand: NO_VALUE,
                    swapped: 0,
                }),
            )
        }
        Kind::Binary => {
            let (left, right) = (unit.inputs[0], unit.inputs[1]);
            if left == NO_VALUE || right == NO_VALUE {
                return None;
            }
            let shape = plan.shape(unit.out);
            let (source, operand, swapped) = if plan.shape(left) == shape {
                (left, right, 0)
            } else if plan.shape(right) == shape {
                (right, left, 1)
            } else {
                (left, right, 0)
            };
            (
                source,
                StepRecord::of(StepFields {
                    op: unit.op,
                    operand,
                    swapped,
                }),
            )
        }
        _ => return None,
    };
    let mut steps = unit.prelude.clone();
    steps.push(step);
    steps.extend(unit.chain.iter().copied());
    Some((source, steps))
}

fn redirected(unit: &TaskInfo, image: u32) -> TaskInfo {
    let mut redirected = unit.clone();
    redirected.out = image;
    redirected
}

fn writes_every_element(kind: Kind) -> bool {
    !matches!(kind, Kind::Scatter | Kind::ScatterWrite)
}

fn convert(
    plan: &mut Plan,
    unit: &TaskInfo,
    source: u32,
    steps: Vec<StepRecord>,
    profile: Profile,
) -> Vec<Task> {
    let target = unit.out;
    let element = plan.values[target as usize].element;
    let words = element.payload_words(u64::from(plan.shape(target).elements()));
    let words = u32::try_from(words).expect("a tensor of words fits the device word space");
    let per_task = task_elements(words, device_workgroups(profile));
    span::chunks(words, per_task, measured(plan, target, Measure::Words))
        .into_iter()
        .map(|(first, count, split)| {
            let mut task = Task::span(unit, first, count, u64::from(count));
            task.kind = Kind::Convert;
            task.op = op::NONE;
            task.geometry = 0;
            task.slot = 0;
            task.out = target;
            task.extra = NO_VALUE;
            task.inputs = [source, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
            task.origin = NO_VALUE;
            task.param = plan.values[target as usize].scale;
            task.splits = 1;
            task.in_place = true;
            task.prelude.clear();
            task.chain = steps.clone();
            task.split = split;
            task
        })
        .collect()
}

impl Plan {
    fn measure(&mut self, measure: Measure) -> u32 {
        if let Some(index) = self.measures.iter().position(|kept| *kept == measure) {
            return index as u32;
        }
        self.measures.push(measure);
        (self.measures.len() - 1) as u32
    }

    fn shape(&self, value: u32) -> Shape {
        self.values[value as usize].shape
    }

    fn geometry(&self, tile: MatmulTile) -> u32 {
        self.tiles
            .iter()
            .position(|candidate| *candidate == tile)
            .unwrap_or_else(|| {
                panic!("a plan walks a matmul tile of {tile:?} its profile offers no geometry for")
            })
            .try_into()
            .expect("a profile carries fewer tiles than a device word holds")
    }

    fn publish(&mut self, shape: Shape) -> u32 {
        let id = self.values.len() as u32;
        self.values.push(ValueInfo::derived(shape, id));
        id
    }
}

fn schedule_unit(
    plan: &mut Plan,
    unit: &TaskInfo,
    profile: Profile,
    chosen: &[(Product, MatmulTile)],
) {
    let target = device_workgroups(profile);
    match unit.kind {
        Kind::Matmul => matmul(plan, unit, profile, chosen),
        Kind::Attention | Kind::AttentionQueryGrad => {
            let rows = plan.shape(unit.out).dims();
            let tokens = rows[2];
            let planes = rows[0] * rows[1];
            let geometry = attention_geometry(plan, profile, rows[3]);
            let tile = plan.attention[geometry as usize];
            let measure = measured(plan, unit.out, Measure::Tokens);
            for (first, count, split) in attention_spans(tokens, planes, profile, measure) {
                let mut task = Task::span(unit, first, count, attention_work(count, tokens, tile));
                task.geometry = geometry;
                task.split = split;
                plan.tasks.push(task);
            }
        }
        Kind::AttentionKeyGrad | Kind::AttentionValueGrad => {
            let keys = plan.shape(unit.inputs[1]).dims();
            let tokens = keys[2];
            let planes = keys[0] * keys[1];
            let geometry = attention_geometry(plan, profile, keys[3]);
            let tile = plan.attention[geometry as usize];
            let measure = measured(plan, unit.inputs[1], Measure::Tokens);
            for (first, count, split) in attention_spans(tokens, planes, profile, measure) {
                let mut task = Task::span(unit, first, count, attention_work(count, tokens, tile));
                task.geometry = geometry;
                task.split = split;
                plan.tasks.push(task);
            }
        }
        Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad => {
            let out = plan.shape(unit.out);
            let rows = out.rows();
            let columns = out.columns();
            let measure = measured(plan, unit.out, Measure::Rows);
            spread(
                plan,
                unit,
                rows,
                softmax_rows_per_task(rows, target),
                measure,
                |_, count| u64::from(count) * u64::from(columns),
            );
        }
        Kind::Argmax | Kind::Categorical => choice(plan, unit, profile, target),
        Kind::SumChunk => reduce(plan, unit, target),
        Kind::SumAxis => fold(plan, unit, profile, target),
        Kind::Conv2d => {
            let out = plan.shape(unit.out);
            let filter = plan.shape(unit.inputs[1]);
            let dims = filter.dims();
            let taps = u64::from(dims[1] * dims[2] * dims[3]);
            let measure = measured(plan, unit.out, Measure::Elements);
            spread(
                plan,
                unit,
                out.elements(),
                task_elements(out.elements(), target),
                measure,
                |_, count| u64::from(count) * taps,
            );
        }
        Kind::Conv2dInputGrad => {
            let out = plan.shape(unit.out);
            let filter = plan.shape(unit.inputs[0]);
            let dims = filter.dims();
            let groups = out.dims()[1] / dims[1];
            let taps = u64::from(dims[0] / groups * dims[2] * dims[3]);
            let measure = measured(plan, unit.out, Measure::Elements);
            spread(
                plan,
                unit,
                out.elements(),
                task_elements(out.elements(), target),
                measure,
                |_, count| u64::from(count) * taps,
            );
        }
        Kind::PoolMax2d | Kind::PoolMean2d => {
            let out = plan.shape(unit.out);
            let taps = window_taps(unit.window);
            let measure = measured(plan, unit.out, Measure::Elements);
            spread(
                plan,
                unit,
                out.elements(),
                task_elements(out.elements(), target),
                measure,
                |_, count| u64::from(count) * taps,
            );
        }
        Kind::PoolMax2dInputGrad | Kind::PoolMean2dInputGrad => {
            let out = plan.shape(unit.out);
            let covering = covering_windows(unit.window);
            let scans = if unit.kind == Kind::PoolMax2dInputGrad {
                covering * window_taps(unit.window)
            } else {
                covering
            };
            let measure = measured(plan, unit.out, Measure::Elements);
            spread(
                plan,
                unit,
                out.elements(),
                task_elements(out.elements(), target),
                measure,
                |_, count| u64::from(count) * scans,
            );
        }
        Kind::Conv2dWeightGrad => conv_weight_grad(plan, unit, profile),
        Kind::MatmulFold => {
            panic!("a fold of depth partials comes from the product whose depth split")
        }
        Kind::Rope | Kind::RopeGrad => {
            let out = plan.shape(unit.out);
            let measure = measured(plan, unit.out, Measure::Elements);
            spread(
                plan,
                unit,
                out.elements(),
                task_elements(out.elements(), target),
                measure,
                |_, count| u64::from(count) * 4,
            );
        }
        Kind::Binary
        | Kind::Unary
        | Kind::Partial
        | Kind::Fill
        | Kind::Broadcast
        | Kind::Layout
        | Kind::OneHot
        | Kind::Gather => {
            let out = plan.shape(unit.out);
            let measure = measured(plan, unit.out, Measure::Elements);
            spread(
                plan,
                unit,
                out.elements(),
                task_elements(out.elements(), target),
                measure,
                |_, count| u64::from(count),
            );
        }
        Kind::Concat => {
            let source = plan.shape(unit.inputs[0]);
            let measure = measured(plan, unit.inputs[0], Measure::Elements);
            spread(
                plan,
                unit,
                source.elements(),
                task_elements(source.elements(), target),
                measure,
                |_, count| u64::from(count),
            );
        }
        Kind::Slice => {
            let out = plan.shape(unit.out);
            let measure = measured(plan, unit.out, Measure::Elements);
            spread(
                plan,
                unit,
                out.elements(),
                task_elements(out.elements(), target),
                measure,
                |_, count| u64::from(count),
            );
        }
        Kind::Scatter | Kind::ScatterWrite => {
            let rows = plan.shape(unit.inputs[1]).elements();
            let width = plan.shape(unit.out).columns();
            let measure = measured(plan, unit.inputs[1], Measure::Elements);
            spread(
                plan,
                unit,
                rows,
                scatter_rows_per_task(rows),
                measure,
                |_, count| u64::from(count) * u64::from(width),
            );
        }
        Kind::Convert => panic!("a narrow tensor is written by the convert its task schedules"),
    }
}

fn attention_spans(
    tokens: u32,
    planes: u32,
    profile: Profile,
    measure: Option<u32>,
) -> Vec<(u32, u32, Split)> {
    let per_task = profile.workgroup();
    match measure {
        None => {
            let mut spans = Vec::new();
            for plane in 0..planes {
                spans.extend(spans_of(tokens, per_task).map(|(first, count)| {
                    (plane * tokens + first, count, Split::Range { first, count })
                }));
            }
            spans
        }
        Some(measure) => span::plane_chunks(tokens, per_task, planes, measure),
    }
}

fn spans_of(units: u32, per_task: u32) -> impl Iterator<Item = (u32, u32)> {
    spans(units, per_task)
}

fn attention_work(count: u32, tokens: u32, tile: AttentionTile) -> u64 {
    let keys = tile.keys();
    let blocks = tokens.div_ceil(keys).max(1);
    u64::from(count) * u64::from(keys) * u64::from(tile.width()) * u64::from(blocks)
}

fn attention_geometry(plan: &mut Plan, profile: Profile, width: u32) -> u32 {
    if let Some(index) = plan.attention.iter().position(|tile| tile.width() == width) {
        return index as u32;
    }
    plan.attention
        .push(AttentionTile::fit(profile.scratch_bytes(), width));
    (plan.attention.len() - 1) as u32
}

fn device_workgroups(profile: Profile) -> u32 {
    profile.workgroups()
}

fn window_taps(window: Window) -> u64 {
    u64::from(window.reach_rows()) * u64::from(window.reach_columns())
}

fn covering_windows(window: Window) -> u64 {
    u64::from(window.reach_rows().div_ceil(window.stride_rows()))
        * u64::from(window.reach_columns().div_ceil(window.stride_columns()))
}

fn conv_weight_grad(plan: &mut Plan, unit: &TaskInfo, profile: Profile) {
    let input = plan.shape(unit.inputs[0]);
    let gradient = plan.shape(unit.inputs[1]);
    let filters = plan.shape(unit.inputs[2]).elements();
    let positions = input.dims()[0] * gradient.dims()[2] * gradient.dims()[3];
    let per_task = task_elements(filters, device_workgroups(profile));
    let spans_per_chunk = filters.div_ceil(per_task);
    let chunks = if plan.values[unit.inputs[0] as usize].shape.dynamic() {
        1
    } else {
        (device_workgroups(profile) / spans_per_chunk).clamp(1, positions)
    };
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
            task.inputs = unit.inputs;
            task.out = partials;
            plan.tasks.push(task);
        }
    }
    for (first, count) in spans(filters, per_task) {
        let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(chunks));
        task.geometry = strategy::WEIGHT_FOLD;
        task.inputs = [partials, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
        plan.tasks.push(task);
    }
}

fn choice(plan: &mut Plan, unit: &TaskInfo, profile: Profile, target: u32) {
    let out = plan.shape(unit.out);
    let columns = plan.shape(unit.inputs[0]).columns();
    let rows = out.rows();
    let geometry = if columns <= profile.workgroup() {
        strategy::THREAD_ROW
    } else {
        strategy::WORKGROUP_ROW
    };
    let measure = measured(plan, unit.out, Measure::Rows);
    for (first, count, split) in span::chunks(rows, choice_rows_per_task(rows, target), measure) {
        let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(columns));
        task.geometry = geometry;
        task.split = split;
        plan.tasks.push(task);
    }
}

fn matmul(plan: &mut Plan, unit: &TaskInfo, profile: Profile, chosen: &[(Product, MatmulTile)]) {
    let dims = plan.shape(unit.out).dims();
    let (rows, columns) = (dims[2], dims[3]);
    let depth = plan.shape(unit.inputs[0]).dims()[3];
    let planes = dims[0] * dims[1];
    let product = Product::of(planes, rows, columns, depth);
    if !plan.products.contains(&product) {
        plan.products.push(product);
    }
    let tile = chosen
        .iter()
        .find(|(shape, _)| *shape == product)
        .map(|(_, tile)| *tile)
        .unwrap_or_else(|| product.planned(profile));
    assert!(
        profile.tiles().contains(&tile),
        "a plan walks {tile:?} for {product:?} where its profile offers {:?}",
        profile.tiles(),
    );
    let geometry = plan.geometry(tile);
    let tiles = planes * rows.div_ceil(tile.rows()) * columns.div_ceil(tile.columns());
    let splits = product.splits(tile, profile);
    let partials =
        (splits > 1).then(|| plan.publish(Shape::vector(splits * planes * rows * columns)));
    let measure = measured(plan, unit.out, |value| Measure::Tiles { value, geometry });
    for split in 0..splits {
        for index in 0..tiles {
            let mut task = Task::span(unit, index, 1, tile.tile_work());
            if let Some(measure) = measure {
                task.split = Split::Uniform {
                    measure,
                    index,
                    group: tiles,
                };
            }
            task.geometry = geometry;
            if let Some(partials) = partials {
                task.out = partials;
                task.slot = split;
                task.splits = splits;
                task.prelude.clear();
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
    let per_task = task_elements(elements, device_workgroups(profile));
    let measure = measured(plan, unit.out, Measure::Elements);
    for (first, count, split) in span::chunks(elements, per_task, measure) {
        let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(splits));
        task.kind = Kind::MatmulFold;
        task.inputs = [partials, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
        task.splits = splits;
        task.prelude.clear();
        task.split = split;
        plan.tasks.push(task);
    }
}

fn reduce(plan: &mut Plan, unit: &TaskInfo, target: u32) {
    let mut source = unit.inputs[0];
    let dynamic = plan.values[unit.inputs[0] as usize].shape.dynamic();
    let mut extent = plan.measure(Measure::Elements(unit.inputs[0]));
    loop {
        let elements = plan.shape(source).elements();
        let per_reduction = reduction_elements(elements, target);
        let opens = source == unit.inputs[0];
        let measure = dynamic.then_some(extent);
        if elements <= per_reduction {
            let mut task = Task::span(unit, 0, elements, u64::from(elements));
            task.inputs = [source, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
            if let Some(measure) = measure {
                task.split = Split::Uniform {
                    measure,
                    index: 0,
                    group: 1,
                };
            }
            if !opens {
                task.prelude.clear();
            }
            plan.tasks.push(task);
            return;
        }
        let chunks = elements.div_ceil(per_reduction);
        let partials = plan.publish(Shape::vector(chunks));
        if dynamic {
            plan.measured.push((partials, extent));
        }
        for (slot, (first, count, split)) in span::chunks(elements, per_reduction, measure)
            .into_iter()
            .enumerate()
        {
            let mut task = Task::span(unit, first, count, u64::from(count));
            task.inputs = [source, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE, NO_VALUE];
            if !opens {
                task.prelude.clear();
            }
            task.slot = slot as u32;
            task.out = partials;
            task.split = split;
            plan.tasks.push(task);
        }
        extent = plan.measure(Measure::Chunks {
            source: extent,
            divisor: per_reduction,
        });
        source = partials;
    }
}

fn task_elements(elements: u32, target: u32) -> u32 {
    (elements / target)
        .next_power_of_two()
        .clamp(TASK_ELEMENTS_FLOOR, TASK_ELEMENTS_CEILING)
}

fn fold(plan: &mut Plan, unit: &TaskInfo, profile: Profile, target: u32) {
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
        let measure = measured(plan, unit.out, Measure::Elements);
        let per_task = fold_rows_per_task(out.elements(), target);
        for (first, count, split) in span::chunks(out.elements(), per_task, measure) {
            let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(columns));
            task.geometry = geometry;
            task.split = split;
            plan.tasks.push(task);
        }
        return;
    }
    let measure = measured(plan, unit.out, Measure::Elements);
    let per_task = task_elements(out.elements(), target);
    for (first, count, split) in span::chunks(out.elements(), per_task, measure) {
        let mut task = Task::span(unit, first, count, u64::from(count) * u64::from(folds));
        task.geometry = strategy::THREAD_ELEMENT;
        task.split = split;
        plan.tasks.push(task);
    }
}

fn reduction_elements(elements: u32, target: u32) -> u32 {
    (elements / target)
        .next_power_of_two()
        .clamp(REDUCTION_FLOOR, REDUCTION_CEILING)
}

fn softmax_rows_per_task(rows: u32, target: u32) -> u32 {
    rows.div_ceil(target).clamp(1, SOFTMAX_ROW_CEILING)
}

fn fold_rows_per_task(rows: u32, target: u32) -> u32 {
    rows.div_ceil(target).clamp(1, FOLD_ROW_CEILING)
}

fn choice_rows_per_task(rows: u32, target: u32) -> u32 {
    rows.div_ceil(target).max(1)
}

const SCATTER_ROW_CEILING: u32 = 4096;

fn scatter_rows_per_task(rows: u32) -> u32 {
    rows.min(SCATTER_ROW_CEILING)
}

fn measured(plan: &mut Plan, value: u32, measure: impl FnOnce(u32) -> Measure) -> Option<u32> {
    if !plan.values[value as usize].shape.dynamic() {
        return None;
    }
    Some(plan.measure(measure(value)))
}

fn spread(
    plan: &mut Plan,
    unit: &TaskInfo,
    total: u32,
    per_task: u32,
    measure: Option<u32>,
    work: impl Fn(u32, u32) -> u64,
) {
    for (first, count, split) in span::chunks(total, per_task, measure) {
        let mut task = Task::span(unit, first, count, work(first, count));
        task.split = split;
        plan.tasks.push(task);
    }
}

pub(crate) fn spans(units: u32, per_task: u32) -> impl Iterator<Item = (u32, u32)> {
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
