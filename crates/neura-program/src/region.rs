use crate::lower::Task;
use neura_abi::{Kind, NO_VALUE};
use neura_graph::ValueInfo;
use neura_profile::MatmulTile;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Region {
    Whole,
    Run { first: u64, count: u64 },
}

impl Region {
    pub(crate) fn run(first: u64, count: u64) -> Self {
        if count == 0 {
            return Self::Whole;
        }
        Self::Run { first, count }
    }

    pub(crate) fn overlaps(self, other: Self) -> bool {
        match (self, other) {
            (Self::Whole, _) | (_, Self::Whole) => true,
            (
                Self::Run {
                    first: left,
                    count: l,
                },
                Self::Run {
                    first: right,
                    count: r,
                },
            ) => left < right + r && right < left + l,
        }
    }
}

#[derive(Default)]
pub(crate) struct Touches {
    pub(crate) writes: Vec<(u32, Region)>,
    pub(crate) reads: Vec<(u32, Region)>,
}

pub(crate) fn touches(values: &[ValueInfo], tiles: &[MatmulTile], task: &Task) -> Touches {
    let mut touches = Touches::default();
    let out = values[task.out as usize].clone();
    if task.in_place || !owned(values, task.out) || !dense(&out) {
        whole(values, task, &mut touches);
        return touches;
    }
    match task.kind {
        Kind::Matmul => product(values, tiles, task, &mut touches),
        Kind::MatmulFold => fold(values, task, &mut touches),
        Kind::Convert => convert(values, task, &mut touches),
        Kind::Binary | Kind::Unary | Kind::Fill | Kind::Broadcast | Kind::Partial => {
            let range = Region::run(u64::from(task.first), u64::from(task.count));
            for value in writes(task) {
                touches.writes.push((values[value as usize].storage, range));
            }
            for value in reads(task) {
                let source = &values[value as usize];
                let region = if source.shape == out.shape && dense(source) && owned(values, value) {
                    range
                } else {
                    Region::Whole
                };
                touches.reads.push((source.storage, region));
            }
        }
        Kind::Rope | Kind::RopeGrad => {
            let half = u64::from(out.shape.dims()[3] / 2);
            let first = u64::from(task.first);
            let count = u64::from(task.count);
            for value in writes(task) {
                touches
                    .writes
                    .push((values[value as usize].storage, Region::run(first, count)));
            }
            for value in task
                .inputs
                .iter()
                .copied()
                .filter(|value| *value != NO_VALUE)
            {
                let source = &values[value as usize];
                let region = if source.shape == out.shape && dense(source) && owned(values, value) {
                    Region::run(first.saturating_sub(half), count + 2 * half)
                } else {
                    Region::Whole
                };
                touches.reads.push((source.storage, region));
            }
            for value in std::iter::once(task.origin)
                .chain(task.prelude.iter().map(|step| step.operand))
                .chain(task.chain.iter().map(|step| step.operand))
                .filter(|value| *value != NO_VALUE)
            {
                touches
                    .reads
                    .push((values[value as usize].storage, Region::Whole));
            }
        }
        Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad => {
            let columns = u64::from(out.shape.dims()[3]);
            let range = Region::run(
                u64::from(task.first) * columns,
                u64::from(task.count) * columns,
            );
            touches.writes.push((out.storage, range));
            for value in task
                .inputs
                .iter()
                .copied()
                .filter(|value| *value != NO_VALUE)
            {
                let source = &values[value as usize];
                let region = if source.shape.dims()[3] == out.shape.dims()[3] {
                    range
                } else {
                    Region::Whole
                };
                touches.reads.push((source.storage, region));
            }
            for value in std::iter::once(task.origin)
                .chain(task.prelude.iter().map(|step| step.operand))
                .chain(task.chain.iter().map(|step| step.operand))
                .filter(|value| *value != NO_VALUE)
            {
                touches
                    .reads
                    .push((values[value as usize].storage, Region::Whole));
            }
        }
        _ => whole(values, task, &mut touches),
    }
    touches
}

fn whole(values: &[ValueInfo], task: &Task, touches: &mut Touches) {
    for value in writes(task) {
        touches
            .writes
            .push((values[value as usize].storage, Region::Whole));
    }
    for value in reads(task) {
        touches
            .reads
            .push((values[value as usize].storage, Region::Whole));
    }
}

fn owned(values: &[ValueInfo], value: u32) -> bool {
    values[value as usize].storage == value
}

fn dense(info: &ValueInfo) -> bool {
    info.strides == info.shape.strides()
}

fn writes(task: &Task) -> impl Iterator<Item = u32> + '_ {
    [task.out, task.extra]
        .into_iter()
        .filter(|value| *value != NO_VALUE)
}

fn reads(task: &Task) -> impl Iterator<Item = u32> + '_ {
    task.inputs
        .iter()
        .copied()
        .chain([task.origin])
        .chain(task.prelude.iter().map(|step| step.operand))
        .chain(task.chain.iter().map(|step| step.operand))
        .filter(|value| *value != NO_VALUE)
}

fn product(values: &[ValueInfo], tiles: &[MatmulTile], task: &Task, touches: &mut Touches) {
    let left = values[task.inputs[0] as usize].clone();
    let right = values[task.inputs[1] as usize].clone();
    let rows = u64::from(left.shape.dims()[2]);
    let columns = u64::from(right.shape.dims()[3]);
    let plane_columns = u64::from(left.shape.dims()[1].max(right.shape.dims()[1]));
    let planes = u64::from(left.shape.dims()[0].max(right.shape.dims()[0])) * plane_columns;
    let tile = tiles[task.geometry as usize];
    let row_blocks = rows.div_ceil(u64::from(tile.rows()));
    let column_blocks = columns.div_ceil(u64::from(tile.columns()));
    let tiles_per_plane = row_blocks * column_blocks;
    let plane = u64::from(task.first) / tiles_per_plane;
    let within = u64::from(task.first) % tiles_per_plane;
    let base_row = (within / column_blocks) * u64::from(tile.rows());
    let band_rows = u64::from(tile.rows()).min(rows - base_row);
    let write = if dense(&values[task.out as usize]) {
        Region::run(
            (u64::from(task.slot) * planes + plane) * rows * columns + base_row * columns,
            band_rows * columns,
        )
    } else {
        Region::Whole
    };
    touches
        .writes
        .push((values[task.out as usize].storage, write));
    for extra in [task.extra].into_iter().filter(|value| *value != NO_VALUE) {
        touches
            .writes
            .push((values[extra as usize].storage, Region::Whole));
    }
    let strides = left.shape.strides();
    let band = if dense(&left) && left.shape.dims()[3] > 1 {
        Region::run(
            (plane / plane_columns) * u64::from(strides[0])
                + (plane % plane_columns) * u64::from(strides[1])
                + base_row * u64::from(strides[2]),
            band_rows * u64::from(strides[2]),
        )
    } else {
        Region::Whole
    };
    touches.reads.push((left.storage, band));
    for value in task.inputs[1..].iter().copied() {
        if value != NO_VALUE {
            touches
                .reads
                .push((values[value as usize].storage, Region::Whole));
        }
    }
    for value in std::iter::once(task.origin)
        .chain(task.prelude.iter().map(|step| step.operand))
        .chain(task.chain.iter().map(|step| step.operand))
        .filter(|value| *value != NO_VALUE)
    {
        touches
            .reads
            .push((values[value as usize].storage, Region::Whole));
    }
}

fn fold(values: &[ValueInfo], task: &Task, touches: &mut Touches) {
    let out = values[task.out as usize].clone();
    let elements = u64::from(out.shape.elements());
    let partials = &values[task.inputs[0] as usize];
    let span = u64::from(task.splits) * elements;
    touches.writes.push((
        out.storage,
        Region::run(u64::from(task.first), u64::from(task.count)),
    ));
    let read = if dense(partials) {
        Region::run(u64::from(task.first), span - u64::from(task.first))
    } else {
        Region::Whole
    };
    touches.reads.push((partials.storage, read));
    for value in task
        .chain
        .iter()
        .map(|step| step.operand)
        .filter(|value| *value != NO_VALUE)
    {
        touches
            .reads
            .push((values[value as usize].storage, Region::Whole));
    }
}

fn convert(values: &[ValueInfo], task: &Task, touches: &mut Touches) {
    let out = values[task.out as usize].clone();
    let stride = out.element.elements_per_word();
    let first = u64::from(task.first) * stride;
    let count = u64::from(task.count) * stride;
    let range = if dense(&out) {
        Region::run(first, count)
    } else {
        Region::Whole
    };
    touches.writes.push((out.storage, range));
    for value in task
        .inputs
        .iter()
        .copied()
        .chain(task.prelude.iter().map(|step| step.operand))
        .chain(task.chain.iter().map(|step| step.operand))
        .filter(|value| *value != NO_VALUE)
    {
        let source = &values[value as usize];
        let region = if source.shape == out.shape && dense(source) {
            Region::run(first, count)
        } else {
            Region::Whole
        };
        touches.reads.push((source.storage, region));
    }
}
