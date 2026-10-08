use crate::access::Reads;
use crate::lower::Task;
use crate::span::Split;
use neura_abi::{Kind, NO_VALUE};
use neura_graph::ValueInfo;
use neura_profile::MatmulTile;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Region {
    Whole,
    Run {
        first: u64,
        count: u64,
    },
    Band {
        first: u64,
        span: u64,
        stride: u64,
        count: u64,
    },
}

impl Region {
    pub(crate) fn run(first: u64, count: u64) -> Self {
        if count == 0 {
            return Self::Whole;
        }
        Self::Run { first, count }
    }

    pub(crate) fn band(first: u64, span: u64, stride: u64, count: u64) -> Self {
        if span == 0 || count == 0 {
            return Self::Whole;
        }
        if count == 1 || stride == 0 {
            return Self::run(first, span);
        }
        if stride <= span {
            let hull = (count - 1).saturating_mul(stride).saturating_add(span);
            return Self::run(first, hull);
        }
        Self::Band {
            first,
            span,
            stride,
            count,
        }
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
    fn write(&mut self, value: u32, region: Region) {
        self.writes.push((value, region));
    }

    fn read(&mut self, value: u32, region: Region) {
        self.reads.push((value, region));
    }

    fn read_whole(&mut self, value: u32) {
        if value != NO_VALUE {
            self.reads.push((value, Region::Whole));
        }
    }

    fn push_regions(
        out: &mut Vec<(u32, Region)>,
        narrowed: &[(u32, Region)],
        value: u32,
        storage: u32,
    ) {
        let mut named = false;
        for (kept, region) in narrowed {
            if *kept == value {
                out.push((storage, *region));
                named = true;
            }
        }
        if !named {
            out.push((storage, Region::Whole));
        }
    }
}

pub(crate) fn touches(values: &[ValueInfo], tiles: &[MatmulTile], task: &Task) -> Touches {
    let narrowed = narrowed(values, tiles, task);
    let mut touches = Touches::default();
    for value in task.writes() {
        let storage = values[value as usize].storage;
        Narrowed::push_regions(&mut touches.writes, &narrowed.writes, value, storage);
    }
    for value in task.reads() {
        let storage = values[value as usize].storage;
        Narrowed::push_regions(&mut touches.reads, &narrowed.reads, value, storage);
    }
    assert_names_held_numbers(values, task, &touches);
    touches
}

fn assert_names_held_numbers(values: &[ValueInfo], task: &Task, touches: &Touches) {
    for (storage, region) in touches.reads.iter().chain(&touches.writes) {
        let first = match *region {
            Region::Whole => continue,
            Region::Run { first, .. } | Region::Band { first, .. } => first,
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
    if frozen(values, task) {
        match task.kind {
            Kind::Matmul => product(values, tiles, task, &mut narrowed),
            Kind::MatmulFold => fold(values, task, &mut narrowed),
            Kind::Convert => convert(values, task, &mut narrowed),
            Kind::Rope | Kind::RopeGrad => rope(values, task, &mut narrowed),
            Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad => {
                softmax(values, task, &mut narrowed)
            }
            Kind::Binary
            | Kind::Unary
            | Kind::Select
            | Kind::Fill
            | Kind::Broadcast
            | Kind::Partial => elementwise(values, task, &mut narrowed),
            _ => {}
        }
    }
    for count in &task.depends {
        narrowed.read_whole(*count);
    }
    narrowed
}

fn frozen(values: &[ValueInfo], task: &Task) -> bool {
    matches!(task.split, Split::Range { .. })
        && owned(values, task.out)
        && dense(&values[task.out as usize])
        && task
            .reads()
            .chain(task.writes())
            .all(|value| !walks_a_free_axis(values, value))
}

fn walks_a_free_axis(values: &[ValueInfo], value: u32) -> bool {
    let info = &values[value as usize];
    info.shape.dynamic() || values[info.storage as usize].shape.dynamic()
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
        narrowed.write(value, range);
    }
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            range
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn product(values: &[ValueInfo], tiles: &[MatmulTile], task: &Task, narrowed: &mut Narrowed) {
    let left = &values[task.inputs[0] as usize];
    let right = &values[task.inputs[1] as usize];
    let rows = u64::from(left.shape.dims()[2]);
    let depth = u64::from(left.shape.dims()[3]);
    let columns = u64::from(right.shape.dims()[3]);
    let plane_columns = u64::from(left.shape.dims()[1].max(right.shape.dims()[1]));
    let planes = u64::from(left.shape.dims()[0].max(right.shape.dims()[0])) * plane_columns;
    let tile = tiles[task.geometry as usize];
    let row_blocks = rows.div_ceil(u64::from(tile.rows()));
    let column_blocks = columns.div_ceil(u64::from(tile.columns()));
    let tiles_per_plane = row_blocks * column_blocks;
    let plane = u64::from(task.first) / tiles_per_plane;
    let within = u64::from(task.first) % tiles_per_plane;
    let plane_row = plane / plane_columns;
    let plane_column = plane % plane_columns;
    let base_row = (within / column_blocks) * u64::from(tile.rows());
    let band_rows = u64::from(tile.rows()).min(rows - base_row);
    let base_column = (within % column_blocks) * u64::from(tile.columns());
    let band_columns = u64::from(tile.columns()).min(columns - base_column);
    let split = task.splits.max(1);
    let slot = if split > 1 { u64::from(task.slot) } else { 0 };
    let depth_blocks = depth.div_ceil(u64::from(tile.depth()));
    let first_block = slot * depth_blocks / u64::from(split);
    let last_block = ((slot + 1) * depth_blocks / u64::from(split)).min(depth_blocks);
    let base_depth = (first_block * u64::from(tile.depth())).min(depth);
    let end_depth = (last_block * u64::from(tile.depth())).min(depth);
    let band_depth = end_depth - base_depth;
    let out_plane = (slot * planes + plane) * rows * columns;
    narrowed.write(
        task.out,
        spans(
            out_plane + base_row * columns + base_column,
            columns,
            1,
            band_rows,
            band_columns,
        ),
    );
    narrowed.read(
        task.inputs[0],
        plane_box(
            left,
            plane_row,
            plane_column,
            base_row,
            band_rows,
            base_depth,
            band_depth,
        ),
    );
    narrowed.read(
        task.inputs[1],
        plane_box(
            right,
            plane_row,
            plane_column,
            base_depth,
            band_depth,
            base_column,
            band_columns,
        ),
    );
    for step in &task.chain {
        if step.operand == NO_VALUE {
            continue;
        }
        let operand = &values[step.operand as usize];
        narrowed.read(
            step.operand,
            plane_box(
                operand,
                plane_row,
                plane_column,
                base_row,
                band_rows,
                base_column,
                band_columns,
            ),
        );
    }
    for step in &task.prelude {
        if step.operand == NO_VALUE {
            continue;
        }
        let operand = &values[step.operand as usize];
        narrowed.read(
            step.operand,
            plane_box(
                operand,
                plane_row,
                plane_column,
                base_row,
                band_rows,
                base_depth,
                band_depth,
            ),
        );
    }
}

fn plane_box(
    info: &ValueInfo,
    plane_row: u64,
    plane_column: u64,
    base_outer: u64,
    outer: u64,
    base_inner: u64,
    inner: u64,
) -> Region {
    let (x, y, z, w) = (
        u64::from(info.strides[0]),
        u64::from(info.strides[1]),
        u64::from(info.strides[2]),
        u64::from(info.strides[3]),
    );
    let base = plane_row
        .saturating_mul(x)
        .saturating_add(plane_column.saturating_mul(y))
        .saturating_add(base_outer.saturating_mul(z))
        .saturating_add(base_inner.saturating_mul(w));
    spans(base, z, w, outer, inner)
}

fn spans(base: u64, outer_stride: u64, inner_stride: u64, outer: u64, inner: u64) -> Region {
    if outer == 0 || inner == 0 {
        return Region::run(base, 0);
    }
    if inner == 1 {
        return match (outer, outer_stride) {
            (1, _) | (_, 0) => Region::run(base, 1),
            (_, stride) => Region::band(base, 1, stride, outer),
        };
    }
    if outer == 1 {
        return match inner_stride {
            0 => Region::run(base, 1),
            1 => Region::run(base, inner),
            stride => Region::band(base, 1, stride, inner),
        };
    }
    match (inner_stride, outer_stride) {
        (0, 0) => Region::run(base, 1),
        (0, stride) => Region::band(base, 1, stride, outer),
        (1, 0) => Region::run(base, inner),
        (1, 1) => Region::run(base, outer + inner - 1),
        (1, stride) if stride >= inner => Region::band(base, inner, stride, outer),
        (stride, 0) => Region::run(base, (inner - 1).saturating_mul(stride) + 1),
        (stride, 1) if stride >= outer => Region::band(base, outer, stride, inner),
        (stride, outer_stride) => {
            let hull = (outer - 1)
                .saturating_mul(outer_stride)
                .saturating_add((inner - 1).saturating_mul(stride))
                + 1;
            Region::run(base, hull)
        }
    }
}

fn fold(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let elements = u64::from(out.shape.elements());
    let count = u64::from(task.count);
    let splits = u64::from(task.splits.max(1));
    let partials = &values[task.inputs[0] as usize];
    narrowed.write(task.out, Region::run(u64::from(task.first), count));
    let read = if dense(partials) {
        Region::band(u64::from(task.first), count, elements, splits)
    } else {
        Region::Whole
    };
    narrowed.read(task.inputs[0], read);
    for step in &task.chain {
        if step.operand == NO_VALUE {
            continue;
        }
        let value = step.operand;
        let region = if walks_its_range(values, value, out) {
            Region::run(u64::from(task.first), count)
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn convert(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let stride = out.element.elements_per_word();
    let first = u64::from(task.first) * stride;
    let count = u64::from(task.count) * stride;
    narrowed.write(task.out, Region::run(first, count));
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            Region::run(first, count)
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn rope(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let half = u64::from(out.shape.dims()[3] / 2);
    let first = u64::from(task.first);
    let count = u64::from(task.count);
    let range = Region::run(first, count);
    for value in task.writes() {
        narrowed.write(value, range);
    }
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            Region::run(first.saturating_sub(half), count + 2 * half)
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn softmax(values: &[ValueInfo], task: &Task, narrowed: &mut Narrowed) {
    let out = &values[task.out as usize];
    let columns = u64::from(out.shape.dims()[3]);
    let range = Region::run(
        u64::from(task.first) * columns,
        u64::from(task.count) * columns,
    );
    narrowed.write(task.out, range);
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            range
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}
