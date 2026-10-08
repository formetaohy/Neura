use crate::access::Reads;
use crate::lower::Task;
use crate::span::{Extents, Split};
use neura_abi::{Element, Kind, NO_VALUE};
use neura_graph::{Shape, ValueInfo};
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
    narrowed: Vec<(u32, Region)>,
    narrowed_reads: Vec<(u32, Region)>,
}

pub(crate) trait Walk: Reads {
    fn kind(&self) -> Kind;
    fn geometry(&self) -> u32;
    fn splits(&self) -> u32;
    fn slot(&self) -> u32;
    fn input(&self, slot: usize) -> u32;
    fn prelude(&self) -> impl Iterator<Item = u32>;
    fn chain(&self) -> impl Iterator<Item = u32>;
    fn depends(&self) -> impl Iterator<Item = u32>;
}

impl Walk for Task {
    fn kind(&self) -> Kind {
        self.kind
    }

    fn geometry(&self) -> u32 {
        self.geometry
    }

    fn splits(&self) -> u32 {
        self.splits
    }

    fn slot(&self) -> u32 {
        self.slot
    }

    fn input(&self, slot: usize) -> u32 {
        self.inputs[slot]
    }

    fn prelude(&self) -> impl Iterator<Item = u32> {
        self.prelude.iter().map(|step| step.operand)
    }

    fn chain(&self) -> impl Iterator<Item = u32> {
        self.chain.iter().map(|step| step.operand)
    }

    fn depends(&self) -> impl Iterator<Item = u32> {
        self.depends.iter().copied()
    }
}

pub(crate) trait Values {
    fn dims(&self, value: u32) -> [u32; 4];
    fn strides(&self, value: u32) -> [u32; 4];
    fn bounds(&self, value: u32) -> [u32; 4];
    fn element(&self, value: u32) -> Element;
    fn storage(&self, value: u32) -> u32;

    fn elements(&self, value: u32) -> u64 {
        self.dims(value).iter().map(|dim| u64::from(*dim)).product()
    }

    fn owned(&self, value: u32) -> bool {
        self.storage(value) == value
    }

    fn dense(&self, value: u32) -> bool {
        self.strides(value) == Shape::dense_strides(self.dims(value))
    }
}

impl Values for &[ValueInfo] {
    fn dims(&self, value: u32) -> [u32; 4] {
        self[value as usize].shape.dims()
    }

    fn strides(&self, value: u32) -> [u32; 4] {
        self[value as usize].strides
    }

    fn bounds(&self, value: u32) -> [u32; 4] {
        self[value as usize].shape.dims()
    }

    fn element(&self, value: u32) -> Element {
        self[value as usize].element
    }

    fn storage(&self, value: u32) -> u32 {
        self[value as usize].storage
    }
}

pub(crate) struct Resolved {
    dims: Vec<[u32; 4]>,
    strides: Vec<[u32; 4]>,
    bounds: Vec<[u32; 4]>,
    storage: Vec<u32>,
    element: Vec<Element>,
}

impl Resolved {
    pub(crate) fn of(records: &[u8], extents: &Extents, bound: &[u32]) -> Self {
        let count = records.len() / size_of::<neura_abi::ValueRecord>();
        let mut resolved = Self {
            dims: Vec::with_capacity(count),
            strides: Vec::with_capacity(count),
            bounds: Vec::with_capacity(count),
            storage: Vec::with_capacity(count),
            element: Vec::with_capacity(count),
        };
        for value in 0..count as u32 {
            let at = value as usize * size_of::<neura_abi::ValueRecord>();
            let record: neura_abi::ValueRecord = bytemuck::pod_read_unaligned(
                &records[at..at + size_of::<neura_abi::ValueRecord>()],
            );
            resolved.bounds.push(record.bounds);
            resolved.storage.push(record.storage);
            resolved.element.push(Element::of(record.element));
            resolved.dims.push(extents.dims(value, bound));
            resolved.strides.push(extents.strides(value, bound));
        }
        resolved
    }
}

impl Values for Resolved {
    fn dims(&self, value: u32) -> [u32; 4] {
        self.dims[value as usize]
    }

    fn strides(&self, value: u32) -> [u32; 4] {
        self.strides[value as usize]
    }

    fn bounds(&self, value: u32) -> [u32; 4] {
        self.bounds[value as usize]
    }

    fn element(&self, value: u32) -> Element {
        self.element[value as usize]
    }

    fn storage(&self, value: u32) -> u32 {
        self.storage[value as usize]
    }
}

pub(crate) fn touches(values: &[ValueInfo], tiles: &[MatmulTile], task: &Task) -> Touches {
    let mut touched = Touches::default();
    if frozen(values, task) {
        walked(&mut touched, task, &values, tiles, (task.first, task.count));
    } else {
        whole(&mut touched, task, &values);
    }
    touched
}

pub(crate) fn walked<W: Walk, V: Values>(
    touched: &mut Touches,
    task: &W,
    values: &V,
    tiles: &[MatmulTile],
    span: (u32, u32),
) {
    touched.clear();
    let (first, count) = span;
    if count == 0 {
        return;
    }
    let Touches {
        writes,
        reads,
        narrowed,
        narrowed_reads,
    } = touched;
    narrow(task, values, tiles, first, count, narrowed, narrowed_reads);
    for value in task.writes() {
        push_regions(writes, narrowed, value, values.storage(value));
    }
    for value in task.reads() {
        push_regions(reads, narrowed_reads, value, values.storage(value));
    }
    assert_names_held_numbers(values, task, writes, reads);
}

pub(crate) fn whole<W: Walk, V: Values>(touched: &mut Touches, task: &W, values: &V) {
    touched.clear();
    for value in task.writes() {
        touched.writes.push((values.storage(value), Region::Whole));
    }
    for value in task.reads() {
        touched.reads.push((values.storage(value), Region::Whole));
    }
}

impl Touches {
    fn clear(&mut self) {
        self.writes.clear();
        self.reads.clear();
        self.narrowed.clear();
        self.narrowed_reads.clear();
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

fn assert_names_held_numbers<W: Walk, V: Values>(
    values: &V,
    task: &W,
    writes: &[(u32, Region)],
    reads: &[(u32, Region)],
) {
    for (storage, region) in reads.iter().chain(writes) {
        let first = match *region {
            Region::Whole => continue,
            Region::Run { first, .. } | Region::Band { first, .. } => first,
        };
        assert!(
            first < values.elements(*storage),
            "a {} task narrows a walk of storage {storage} to {first}, and that tensor holds {} numbers; a narrowed walk names numbers the tensor it names holds",
            task.kind().name(),
            values.elements(*storage),
        );
    }
}

fn frozen(values: &[ValueInfo], task: &Task) -> bool {
    matches!(task.split, Split::Range { .. })
        && values[task.out as usize].storage == task.out
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

fn dense(info: &ValueInfo) -> bool {
    info.strides == info.shape.strides()
}

struct Narrowed<'a> {
    writes: &'a mut Vec<(u32, Region)>,
    reads: &'a mut Vec<(u32, Region)>,
}

impl Narrowed<'_> {
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
}

fn narrow<W: Walk, V: Values>(
    task: &W,
    values: &V,
    tiles: &[MatmulTile],
    first: u32,
    count: u32,
    writes: &mut Vec<(u32, Region)>,
    reads: &mut Vec<(u32, Region)>,
) {
    let mut narrowed = Narrowed { writes, reads };
    match task.kind() {
        Kind::Matmul => product(values, tiles, task, first, count, &mut narrowed),
        Kind::MatmulFold => fold(values, task, first, count, &mut narrowed),
        Kind::Convert => convert(values, task, first, count, &mut narrowed),
        Kind::Rope | Kind::RopeGrad => rope(values, task, first, count, &mut narrowed),
        Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad => {
            softmax(values, task, first, count, &mut narrowed)
        }
        Kind::Binary
        | Kind::Unary
        | Kind::Select
        | Kind::Fill
        | Kind::Broadcast
        | Kind::Partial => elementwise(values, task, first, count, &mut narrowed),
        _ => {}
    }
    for count in task.depends() {
        narrowed.read_whole(count);
    }
}

fn walks_its_range<V: Values>(values: &V, value: u32, out: u32) -> bool {
    values.dims(value) == values.dims(out) && values.dense(value) && values.owned(value)
}

fn elementwise<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    let range = Region::run(u64::from(first), u64::from(count));
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

fn product<W: Walk, V: Values>(
    values: &V,
    tiles: &[MatmulTile],
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    assert_eq!(
        count, 1,
        "a product task walks {count} tiles from tile {first}, and the rectangle of a product names one tile",
    );
    let left = task.input(0);
    let right = task.input(1);
    let (left_dims, right_dims) = (values.dims(left), values.dims(right));
    let rows = u64::from(left_dims[2]);
    let depth = u64::from(left_dims[3]);
    let columns = u64::from(right_dims[3]);
    let plane_columns = u64::from(left_dims[1].max(right_dims[1]));
    let planes = u64::from(left_dims[0].max(right_dims[0])) * plane_columns;
    let tile = tiles[task.geometry() as usize];
    let row_blocks = rows.div_ceil(u64::from(tile.rows()));
    let column_blocks = columns.div_ceil(u64::from(tile.columns()));
    let tiles_per_plane = row_blocks * column_blocks;
    let plane = u64::from(first) / tiles_per_plane;
    let within = u64::from(first) % tiles_per_plane;
    let plane_row = plane / plane_columns;
    let plane_column = plane % plane_columns;
    let base_row = (within / column_blocks) * u64::from(tile.rows());
    let band_rows = u64::from(tile.rows()).min(rows - base_row);
    let base_column = (within % column_blocks) * u64::from(tile.columns());
    let band_columns = u64::from(tile.columns()).min(columns - base_column);
    let split = task.splits().max(1);
    let slot = if split > 1 { u64::from(task.slot()) } else { 0 };
    let depth_blocks = depth.div_ceil(u64::from(tile.depth()));
    let first_block = slot * depth_blocks / u64::from(split);
    let last_block = ((slot + 1) * depth_blocks / u64::from(split)).min(depth_blocks);
    let base_depth = (first_block * u64::from(tile.depth())).min(depth);
    let end_depth = (last_block * u64::from(tile.depth())).min(depth);
    let band_depth = end_depth - base_depth;
    let out_plane = (slot * planes + plane) * rows * columns;
    let output = task.out();
    narrowed.write(
        output,
        spans(
            out_plane + base_row * columns + base_column,
            columns,
            1,
            band_rows,
            band_columns,
        ),
    );
    narrowed.read(
        left,
        Rectangle {
            plane_row,
            plane_column,
            base_outer: base_row,
            outer: band_rows,
            base_inner: base_depth,
            inner: band_depth,
        }
        .region(values, left),
    );
    narrowed.read(
        right,
        Rectangle {
            plane_row,
            plane_column,
            base_outer: base_depth,
            outer: band_depth,
            base_inner: base_column,
            inner: band_columns,
        }
        .region(values, right),
    );
    for step in task.chain() {
        if step == NO_VALUE {
            continue;
        }
        narrowed.read(
            step,
            Rectangle {
                plane_row,
                plane_column,
                base_outer: base_row,
                outer: band_rows,
                base_inner: base_column,
                inner: band_columns,
            }
            .region(values, step),
        );
    }
    for step in task.prelude() {
        if step == NO_VALUE {
            continue;
        }
        narrowed.read(
            step,
            Rectangle {
                plane_row,
                plane_column,
                base_outer: base_row,
                outer: band_rows,
                base_inner: base_depth,
                inner: band_depth,
            }
            .region(values, step),
        );
    }
}

#[derive(Clone, Copy)]
struct Rectangle {
    plane_row: u64,
    plane_column: u64,
    base_outer: u64,
    outer: u64,
    base_inner: u64,
    inner: u64,
}

impl Rectangle {
    fn region<V: Values>(self, values: &V, value: u32) -> Region {
        let strides = values.strides(value);
        let (x, y, z, w) = (
            u64::from(strides[0]),
            u64::from(strides[1]),
            u64::from(strides[2]),
            u64::from(strides[3]),
        );
        let base = self
            .plane_row
            .saturating_mul(x)
            .saturating_add(self.plane_column.saturating_mul(y))
            .saturating_add(self.base_outer.saturating_mul(z))
            .saturating_add(self.base_inner.saturating_mul(w));
        spans(base, z, w, self.outer, self.inner)
    }
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

fn fold<W: Walk, V: Values>(values: &V, task: &W, first: u32, count: u32, narrowed: &mut Narrowed) {
    let out = task.out();
    let elements = values.elements(out);
    let partials = task.input(0);
    narrowed.write(out, Region::run(u64::from(first), u64::from(count)));
    let read = if values.dense(partials) {
        Region::band(
            u64::from(first),
            u64::from(count),
            elements,
            u64::from(task.splits().max(1)),
        )
    } else {
        Region::Whole
    };
    narrowed.read(partials, read);
    for step in task.chain() {
        if step == NO_VALUE {
            continue;
        }
        let region = if walks_its_range(values, step, out) {
            Region::run(u64::from(first), u64::from(count))
        } else {
            Region::Whole
        };
        narrowed.read(step, region);
    }
}

fn convert<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    let stride = values.element(out).elements_per_word();
    let first = u64::from(first) * stride;
    let count = u64::from(count) * stride;
    narrowed.write(out, Region::run(first, count));
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            Region::run(first, count)
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn rope<W: Walk, V: Values>(values: &V, task: &W, first: u32, count: u32, narrowed: &mut Narrowed) {
    let out = task.out();
    let half = u64::from(values.dims(out)[3] / 2);
    let first = u64::from(first);
    let count = u64::from(count);
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

fn softmax<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    let columns = u64::from(values.dims(out)[3]);
    let range = Region::run(u64::from(first) * columns, u64::from(count) * columns);
    narrowed.write(out, range);
    for value in task.reads() {
        let region = if walks_its_range(values, value, out) {
            range
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}
