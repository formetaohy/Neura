use crate::access::Reads;
use crate::lower::Task;
use crate::span::Split;
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
}

#[derive(Default)]
pub(crate) struct Touches {
    pub(crate) writes: Vec<(u32, Region)>,
    pub(crate) reads: Vec<(u32, Region)>,
}

#[derive(Default)]
struct Narrowed {
    writes: Vec<(u32, Region)>,
    reads: Vec<(u32, Region)>,
}

impl Narrowed {
    fn narrow_write(&mut self, value: u32, region: Region) {
        self.writes.push((value, region));
    }

    fn narrow_read(&mut self, value: u32, region: Region) {
        self.reads.push((value, region));
    }

    fn read_whole(&mut self, value: u32) {
        if value != NO_VALUE {
            self.reads.push((value, Region::Whole));
        }
    }

    fn region_of(regions: &[(u32, Region)], value: u32) -> Region {
        regions
            .iter()
            .rev()
            .find(|(kept, _)| *kept == value)
            .map_or(Region::Whole, |(_, region)| *region)
    }
}

pub(crate) fn touches(values: &[ValueInfo], tiles: &[MatmulTile], task: &Task) -> Touches {
    let narrowed = narrowed(values, tiles, task);
    let touches = Touches {
        writes: task
            .writes()
            .map(|value| {
                (
                    values[value as usize].storage,
                    Narrowed::region_of(&narrowed.writes, value),
                )
            })
            .collect(),
        reads: task
            .reads()
            .map(|value| {
                (
                    values[value as usize].storage,
                    Narrowed::region_of(&narrowed.reads, value),
                )
            })
            .collect(),
    };
    assert_names_held_numbers(values, task, &touches);
    touches
}

fn assert_names_held_numbers(values: &[ValueInfo], task: &Task, touches: &Touches) {
    for (storage, region) in touches.reads.iter().chain(&touches.writes) {
        let Region::Run { first, .. } = *region else {
            continue;
        };
        let info = &values[*storage as usize];
        assert!(
            first < u64::from(info.shape.elements()),
            "a {} task narrows a walk of storage {storage} to {first}, and that tensor holds {} numbers; a narrowed walk names numbers the tensor it names holds",
            task.kind.name(),
            info.shape.elements(),
        );
    }
}

fn narrowed(values: &[ValueInfo], tiles: &[MatmulTile], task: &Task) -> Narrowed {
    let mut narrowed = Narrowed::default();
    if owned(values, task.out)
        && dense(&values[task.out as usize])
        && !matches!(task.split, Split::Segment { .. })
    {
        match task.kind {
            Kind::Matmul => product(values, tiles, task, &mut narrowed),
            Kind::MatmulFold => fold(values, task, &mut narrowed),
            Kind::Convert => convert(values, task, &mut narrowed),
            Kind::Rope | Kind::RopeGrad => rope(values, task, &mut narrowed),
            Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad => {
                softmax(values, task, &mut narrowed)
            }
            Kind::Binary | Kind::Unary | Kind::Fill | Kind::Broadcast | Kind::Partial => {
                elementwise(values, task, &mut narrowed)
            }
            _ => {}
        }
    }
    for count in &task.depends {
        narrowed.read_whole(*count);
    }
    narrowed
}

fn owned(values: &[ValueInfo], value: u32) -> bool {
    values[value as usize].storage == value
}

fn walks_its_range(values: &[ValueInfo], value: u32, out: &ValueInfo) -> bool {
    let source = &values[value as usize];
    source.shape == out.shape && dense(source) && owned(values, value)
}

fn dense(info: &ValueInfo) -> bool {
    info.strides == info.shape.strides()
}

fn elementwise(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let range = Region::run(u64::from(task.first), u64::from(task.count));
    for value in task.writes() {
        narrowed.narrow_write(value, range);
    }
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            range
        } else {
            Region::Whole
        };
        narrowed.narrow_read(value, region);
    }
}

fn product(values: &[ValueInfo], tiles: &[MatmulTile], task: &Task, narrowed: &mut Narrowed) {
    let left = &values[task.inputs[0] as usize];
    let right = &values[task.inputs[1] as usize];
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
    narrowed.narrow_write(
        task.out,
        Region::run(
            (u64::from(task.slot) * planes + plane) * rows * columns + base_row * columns,
            band_rows * columns,
        ),
    );
    let strides = left.shape.strides();
    let band = if dense(left) && left.shape.dims()[3] > 1 {
        Region::run(
            (plane / plane_columns) * u64::from(strides[0])
                + (plane % plane_columns) * u64::from(strides[1])
                + base_row * u64::from(strides[2]),
            band_rows * u64::from(strides[2]),
        )
    } else {
        Region::Whole
    };
    narrowed.narrow_read(task.inputs[0], band);
}

fn fold(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let elements = u64::from(out.shape.elements());
    let partials = &values[task.inputs[0] as usize];
    let span = u64::from(task.splits) * elements;
    narrowed.narrow_write(
        task.out,
        Region::run(u64::from(task.first), u64::from(task.count)),
    );
    let read = if dense(partials) {
        Region::run(u64::from(task.first), span - u64::from(task.first))
    } else {
        Region::Whole
    };
    narrowed.narrow_read(task.inputs[0], read);
}

fn convert(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let stride = out.element.elements_per_word();
    let first = u64::from(task.first) * stride;
    let count = u64::from(task.count) * stride;
    narrowed.narrow_write(task.out, Region::run(first, count));
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            Region::run(first, count)
        } else {
            Region::Whole
        };
        narrowed.narrow_read(value, region);
    }
}

fn rope(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let half = u64::from(out.shape.dims()[3] / 2);
    let first = u64::from(task.first);
    let count = u64::from(task.count);
    let range = Region::run(first, count);
    for value in task.writes() {
        narrowed.narrow_write(value, range);
    }
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            Region::run(first.saturating_sub(half), count + 2 * half)
        } else {
            Region::Whole
        };
        narrowed.narrow_read(value, region);
    }
}

fn softmax(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let columns = u64::from(out.shape.dims()[3]);
    let range = Region::run(
        u64::from(task.first) * columns,
        u64::from(task.count) * columns,
    );
    narrowed.narrow_write(task.out, range);
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            range
        } else {
            Region::Whole
        };
        narrowed.narrow_read(value, region);
    }
}
