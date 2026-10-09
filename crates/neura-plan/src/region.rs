use crate::access::Reads;
use crate::lower::Task;
use crate::span::{Extents, Split};
use neura_abi::{Element, Kind, NO_VALUE};
use neura_graph::{Shape, ValueInfo};
use neura_profile::MatmulTile;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Region {
    Empty,
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
    pub(crate) fn empty() -> Self {
        Self::Empty
    }

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
    fn split(&self) -> Split;
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

    fn split(&self) -> Split {
        self.split
    }
}

pub(crate) trait Values {
    fn len(&self) -> usize;
    fn dims(&self, value: u32) -> [u32; 4];
    fn strides(&self, value: u32) -> [u32; 4];
    fn bounds(&self, value: u32) -> [u32; 4];
    fn element(&self, value: u32) -> Element;
    fn storage(&self, value: u32) -> u32;
    fn recomputes(&self, value: u32) -> Option<u32>;

    fn exact(&self, value: u32) -> bool;

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
    fn len(&self) -> usize {
        <[ValueInfo]>::len(self)
    }

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

    fn recomputes(&self, value: u32) -> Option<u32> {
        self[value as usize].recomputes
    }

    fn exact(&self, value: u32) -> bool {
        let info = &self[value as usize];
        !info.shape.dynamic() && !self[info.storage as usize].shape.dynamic()
    }
}

pub(crate) struct Resolved {
    dims: Vec<[u32; 4]>,
    strides: Vec<[u32; 4]>,
    bounds: Vec<[u32; 4]>,
    storage: Vec<u32>,
    element: Vec<Element>,
    recomputes: Vec<Option<u32>>,
    exact: Vec<bool>,
}

impl Resolved {
    pub(crate) fn of(records: &[u8], extents: &Extents, bound: &[u32], authored: &[u32]) -> Self {
        let count = records.len() / size_of::<neura_abi::ValueRecord>();
        let mut resolved = Self {
            dims: Vec::with_capacity(count),
            strides: Vec::with_capacity(count),
            bounds: Vec::with_capacity(count),
            storage: Vec::with_capacity(count),
            element: Vec::with_capacity(count),
            recomputes: Vec::with_capacity(count),
            exact: Vec::with_capacity(count),
        };
        for value in 0..count as u32 {
            let at = value as usize * size_of::<neura_abi::ValueRecord>();
            let record: neura_abi::ValueRecord = bytemuck::pod_read_unaligned(
                &records[at..at + size_of::<neura_abi::ValueRecord>()],
            );
            resolved.bounds.push(record.bounds);
            resolved.storage.push(record.storage);
            resolved.element.push(Element::of(record.element));
            resolved.recomputes.push(extents.recomputes(value));
            resolved.dims.push(extents.dims(value, bound));
            resolved.strides.push(extents.strides(value, bound));
            resolved
                .exact
                .push(extents.sealed(value, authored) && extents.sealed(record.storage, authored));
        }
        resolved
    }
}

impl Values for Resolved {
    fn len(&self) -> usize {
        self.dims.len()
    }

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

    fn recomputes(&self, value: u32) -> Option<u32> {
        self.recomputes[value as usize]
    }

    fn exact(&self, value: u32) -> bool {
        self.exact[value as usize]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TableRows<'a> {
    value: u32,
    rows: &'a [u32],
}

impl<'a> TableRows<'a> {
    pub fn new(value: u32, rows: &'a [u32]) -> Self {
        assert!(
            !rows.is_empty(),
            "a host declares the rows of a table walk, and value {value} holds none",
        );
        let mut sorted = true;
        for pair in rows.windows(2) {
            sorted &= pair[0] < pair[1];
        }
        assert!(
            sorted,
            "a host names the rows of a table walk once each and in order, and the rows of value {value} repeat or descend",
        );
        Self { value, rows }
    }

    pub const fn value(self) -> u32 {
        self.value
    }

    pub const fn rows(self) -> &'a [u32] {
        self.rows
    }
}

pub(crate) fn touches<W: Walk, V: Values>(
    values: &V,
    tiles: &[MatmulTile],
    task: &W,
    span: (u32, u32),
    touched: &mut Touches,
) {
    if exact(values, task) {
        walked(touched, task, values, tiles, span, &[]);
    } else {
        whole(touched, task, values);
    }
}

pub(crate) fn exact<W: Walk, V: Values>(values: &V, task: &W) -> bool {
    let out = task.out();
    values.owned(out)
        && values.dense(out)
        && !matches!(task.split(), Split::Ragged { .. })
        && task
            .reads()
            .chain(task.writes())
            .all(|value| values.exact(value))
}

pub(crate) fn walked<W: Walk, V: Values>(
    touched: &mut Touches,
    task: &W,
    values: &V,
    tiles: &[MatmulTile],
    span: (u32, u32),
    rows: &[TableRows<'_>],
) {
    touched.clear();
    let (first, count) = span;
    if count == 0 {
        empty(touched, task, values);
        return;
    }
    let Touches {
        writes,
        reads,
        narrowed,
        narrowed_reads,
    } = touched;
    narrow(
        task,
        values,
        tiles,
        (first, count),
        rows,
        narrowed,
        narrowed_reads,
    );
    for value in task.writes() {
        push_regions(writes, narrowed, value, values.storage(value));
    }
    for value in task.reads() {
        push_regions(reads, narrowed_reads, value, values.storage(value));
    }
    assert_names_held_numbers(values, task, writes, reads);
}

fn empty<W: Walk, V: Values>(touched: &mut Touches, task: &W, values: &V) {
    let identity = matches!(
        task.kind(),
        Kind::SumChunk | Kind::PrefixChunk | Kind::PrefixScan | Kind::PrefixClose | Kind::Length
    );
    if identity {
        for value in task.writes() {
            let storage = values.storage(value);
            let at = match task.kind() {
                Kind::SumChunk | Kind::PrefixChunk => u64::from(task.slot()),
                _ => 0,
            };
            touched.writes.push((
                storage,
                if at < values.elements(storage) {
                    Region::run(at, 1)
                } else {
                    Region::Whole
                },
            ));
        }
    }
    for count in task.depends() {
        touched.reads.push((values.storage(count), Region::Whole));
    }
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
            Region::Empty | Region::Whole => continue,
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

struct Narrowed<'a> {
    writes: &'a mut Vec<(u32, Region)>,
    reads: &'a mut Vec<(u32, Region)>,
}

impl Narrowed<'_> {
    fn names(&self, value: u32) -> bool {
        self.reads.iter().any(|(kept, _)| *kept == value)
    }

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
    span: (u32, u32),
    rows: &[TableRows<'_>],
    writes: &mut Vec<(u32, Region)>,
    reads: &mut Vec<(u32, Region)>,
) {
    let (first, count) = span;
    let mut narrowed = Narrowed { writes, reads };
    match task.kind() {
        Kind::Matmul => product(values, tiles, task, first, count, &mut narrowed),
        Kind::MatmulFold => fold(values, task, first, count, &mut narrowed),
        Kind::Conv2d => convolution(values, task, first, count, &mut narrowed),
        Kind::Conv2dInputGrad => convolution_input_grad(values, task, first, count, &mut narrowed),
        Kind::Conv2dWeightGrad => {
            convolution_weight_grad(values, task, first, count, &mut narrowed)
        }
        Kind::Convert => convert(values, task, first, count, &mut narrowed),
        Kind::Gather => gather(values, task, rows, &mut narrowed),
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
        Kind::Attention
        | Kind::AttentionQueryGrad
        | Kind::AttentionKeyGrad
        | Kind::AttentionValueGrad
        | Kind::PrefixChunk
        | Kind::PrefixScan
        | Kind::PrefixClose
        | Kind::Layout
        | Kind::Extend
        | Kind::SumChunk
        | Kind::SumAxis
        | Kind::Argmax
        | Kind::Categorical
        | Kind::OneHot
        | Kind::Scatter
        | Kind::ScatterWrite
        | Kind::Compact
        | Kind::PoolMax2d
        | Kind::PoolMax2dInputGrad
        | Kind::PoolMean2d
        | Kind::PoolMean2dInputGrad
        | Kind::Concat
        | Kind::Slice
        | Kind::Rows
        | Kind::MatmulWeightGrad
        | Kind::SegmentSum
        | Kind::Length => {}
    }
    for count in task.depends() {
        narrowed.read_whole(count);
    }
}

fn walks_its_range<V: Values>(values: &V, value: u32, out: u32) -> bool {
    values.dims(value) == values.dims(out) && values.dense(value) && values.owned(value)
}

fn gather<W: Walk, V: Values>(
    values: &V,
    task: &W,
    rows: &[TableRows<'_>],
    narrowed: &mut Narrowed<'_>,
) {
    let table = task.input(0);
    let Some(declared) = rows
        .iter()
        .find(|declared| declared.value() == table)
        .map(|declared| declared.rows())
    else {
        return;
    };
    if !values.owned(table) {
        return;
    }
    let columns = u64::from(values.dims(table)[3]);
    let held = values.elements(table) / columns.max(1);
    let mut first = u64::from(declared[0]);
    let mut last = first;
    for row in &declared[1..] {
        let row = u64::from(*row);
        if row == last + 1 {
            last = row;
            continue;
        }
        narrowed.read(table, row_span(first, last, columns, held));
        first = row;
        last = row;
    }
    narrowed.read(table, row_span(first, last, columns, held));
}

fn row_span(first: u64, last: u64, columns: u64, held: u64) -> Region {
    assert!(
        last < held,
        "a host declares row {last} of a table that holds {held} of them",
    );
    Region::run(first * columns, (last - first + 1) * columns)
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

fn convolution<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    let filter = task.input(1);
    let range = Region::run(u64::from(first), u64::from(count));
    narrowed.write(out, range);
    let region = channel_span(values, out, first, count)
        .zip(filter_planes(values, filter))
        .map(|((first_channel, last_channel), (per_channel, _))| {
            Region::run(
                u64::from(first_channel) * per_channel,
                u64::from(last_channel - first_channel + 1) * per_channel,
            )
        })
        .unwrap_or(Region::Whole);
    narrowed.read(filter, region);
    chained_reads(values, task, out, range, narrowed);
}

fn convolution_input_grad<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    let filter = task.input(0);
    let range = Region::run(u64::from(first), u64::from(count));
    narrowed.write(out, range);
    name_filter_bands(values, out, filter, first, count, narrowed);
    chained_reads(values, task, out, range, narrowed);
}

fn name_filter_bands<V: Values>(
    values: &V,
    out: u32,
    filter: u32,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let Some((first_channel, last_channel)) = channel_span(values, out, first, count) else {
        return;
    };
    let Some((per_channel, per_local)) = filter_planes(values, filter) else {
        return;
    };
    let channels = u64::from(values.dims(filter)[1]);
    let groups = u64::from(values.dims(out)[1]) / channels;
    if channels == 0 || groups == 0 {
        return;
    }
    let out_channels = u64::from(values.dims(filter)[0]) / groups;
    if out_channels == 0 {
        return;
    }
    let block = |group: u64, local: (u64, u64)| {
        Region::band(
            group * out_channels * per_channel + local.0 * per_local,
            (local.1 - local.0 + 1) * per_local,
            per_channel,
            out_channels,
        )
    };
    let first_group = u64::from(first_channel) / channels;
    let last_group = u64::from(last_channel) / channels;
    let first_local = u64::from(first_channel) % channels;
    let last_local = u64::from(last_channel) % channels;
    if first_group == last_group {
        narrowed.read(filter, block(first_group, (first_local, last_local)));
        return;
    }
    narrowed.read(filter, block(first_group, (first_local, channels - 1)));
    if first_group + 1 < last_group {
        narrowed.read(
            filter,
            Region::band(
                (first_group + 1) * out_channels * per_channel,
                per_channel,
                per_channel,
                out_channels * (last_group - first_group - 1),
            ),
        );
    }
    narrowed.read(filter, block(last_group, (0, last_local)));
}

fn convolution_weight_grad<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    let filter = task.input(2);
    let range = Region::run(u64::from(first), u64::from(count));
    let written = if filter == NO_VALUE {
        range
    } else {
        if !values.dense(filter) {
            return;
        }
        narrowed.read(filter, Region::empty());
        Region::run(
            u64::from(task.slot()) * values.elements(filter) + u64::from(first),
            u64::from(count),
        )
    };
    narrowed.write(out, written);
    chained_reads(values, task, out, written, narrowed);
}

fn chained_reads<W: Walk, V: Values>(
    values: &V,
    task: &W,
    out: u32,
    range: Region,
    narrowed: &mut Narrowed<'_>,
) {
    for value in task.chain().chain(task.prelude()) {
        if value == NO_VALUE || narrowed.names(value) {
            continue;
        }
        let region = if walks_its_range(values, value, out) {
            range
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn channel_span<V: Values>(values: &V, out: u32, first: u32, count: u32) -> Option<(u32, u32)> {
    let dims = values.dims(out);
    let plane = u64::from(dims[2]).checked_mul(u64::from(dims[3]))?;
    let channels = u64::from(dims[1]);
    if plane == 0 || channels == 0 {
        return None;
    }
    let batch = plane.checked_mul(channels)?;
    let first = u64::from(first);
    let last = first.checked_add(u64::from(count))?.checked_sub(1)?;
    if first / batch != last / batch {
        return None;
    }
    Some((
        ((first / plane) % channels) as u32,
        ((last / plane) % channels) as u32,
    ))
}

fn filter_planes<V: Values>(values: &V, filter: u32) -> Option<(u64, u64)> {
    if !values.dense(filter) {
        return None;
    }
    let dims = values.dims(filter);
    let per_local = u64::from(dims[2]).checked_mul(u64::from(dims[3]))?;
    let per_channel = u64::from(dims[1]).checked_mul(per_local)?;
    if per_channel == 0 {
        return None;
    }
    Some((per_channel, per_local))
}
