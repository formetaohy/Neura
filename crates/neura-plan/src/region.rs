use crate::access::Reads;
use crate::lower::{Task, softmax_words};
use crate::span::{Extents, Split};
use neura_abi::{CANDIDATES, Element, Kind, NO_VALUE, WORD_BYTES};
use neura_graph::{Shape, ValueInfo, Window};
use neura_profile::{ConvolutionInputTile, ConvolutionTile, MatmulTile};

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
    fn section(&self) -> u32;
    fn window(&self) -> Window;
    fn input(&self, slot: usize) -> u32;
    fn prelude(&self) -> impl Iterator<Item = u32>;
    fn chain(&self) -> impl Iterator<Item = u32>;
    fn depends(&self) -> impl Iterator<Item = u32>;
    fn split(&self) -> Split;
    fn plane(&self) -> u32;
    fn axis(&self) -> u32;
    fn offset(&self) -> u32;
    fn segments(&self) -> u32;
    fn queries(&self) -> u32;
}

impl Walk for Task {
    fn kind(&self) -> Kind {
        self.kind
    }

    fn window(&self) -> Window {
        self.window
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

    fn section(&self) -> u32 {
        self.section
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

    fn plane(&self) -> u32 {
        self.plane
    }

    fn axis(&self) -> u32 {
        self.axis
    }

    fn offset(&self) -> u32 {
        self.offset
    }

    fn segments(&self) -> u32 {
        self.segments
    }

    fn queries(&self) -> u32 {
        self.queries
    }
}

pub(crate) trait Values {
    fn len(&self) -> usize;
    fn dims(&self, value: u32) -> [u32; 4];
    fn strides(&self, value: u32) -> [u32; 4];
    fn origin(&self, value: u32) -> u64;
    fn bounds(&self, value: u32) -> [u32; 4];
    fn element(&self, value: u32) -> Element;
    fn storage(&self, value: u32) -> u32;
    fn recomputes(&self, value: u32) -> Option<u32>;

    fn exact(&self, value: u32) -> bool;

    fn bounded(&self, value: u32) -> bool {
        self.exact(value)
    }

    fn strides_hold(&self, value: u32) -> bool {
        self.exact(value)
    }

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

    fn origin(&self, value: u32) -> u64 {
        self[value as usize].layout.origin()
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
    origins: Vec<u64>,
    bounds: Vec<[u32; 4]>,
    storage: Vec<u32>,
    element: Vec<Element>,
    recomputes: Vec<Option<u32>>,
    exact: Vec<bool>,
    bounded: Vec<bool>,
    holds: Vec<bool>,
}

impl Resolved {
    pub(crate) fn of(records: &[u8], extents: &Extents, bound: &[u32], authored: &[u32]) -> Self {
        let count = records.len() / size_of::<neura_abi::ValueRecord>();
        let mut resolved = Self {
            dims: Vec::with_capacity(count),
            strides: Vec::with_capacity(count),
            origins: Vec::with_capacity(count),
            bounds: Vec::with_capacity(count),
            storage: Vec::with_capacity(count),
            element: Vec::with_capacity(count),
            recomputes: Vec::with_capacity(count),
            exact: Vec::with_capacity(count),
            bounded: Vec::with_capacity(count),
            holds: Vec::with_capacity(count),
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
            resolved.origins.push(extents.origin(value));
            resolved
                .exact
                .push(extents.sealed(value, authored) && extents.sealed(record.storage, authored));
            let dense = resolved.strides[value as usize]
                == Shape::dense_strides(resolved.dims[value as usize]);
            let holds = extents.strides_hold(value, authored);
            resolved
                .bounded
                .push(dense && extents.cut_is_outer(value, authored) && holds);
            resolved.holds.push(holds);
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

    fn origin(&self, value: u32) -> u64 {
        self.origins[value as usize]
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

    fn bounded(&self, value: u32) -> bool {
        self.bounded[value as usize]
    }

    fn strides_hold(&self, value: u32) -> bool {
        self.holds[value as usize]
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

#[derive(Clone, Copy)]
pub(crate) struct GeometryTiles<'a> {
    pub(crate) matmul: &'a [MatmulTile],
    pub(crate) convolution: &'a [ConvolutionTile],
    pub(crate) convolution_input: &'a [ConvolutionInputTile],
}

pub(crate) fn touches<W: Walk, V: Values>(
    values: &V,
    tiles: GeometryTiles<'_>,
    task: &W,
    span: (u32, u32),
    touched: &mut Touches,
) {
    if task.kind() == Kind::SegmentSum || exact(values, task) || bounded(values, task) {
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

pub(crate) fn bounded<W: Walk, V: Values>(values: &V, task: &W) -> bool {
    let out = task.out();
    values.owned(out)
        && values.bounded(out)
        && !matches!(task.split(), Split::Ragged { .. })
        && (task.splits() == 1 || !matches!(task.split(), Split::Segment { .. }))
        && task
            .reads()
            .chain(task.writes())
            .all(|value| values.exact(value) || values.strides_hold(value))
}

pub(crate) fn walked<W: Walk, V: Values>(
    touched: &mut Touches,
    task: &W,
    values: &V,
    tiles: GeometryTiles<'_>,
    span: (u32, u32),
    rows: &[TableRows<'_>],
) {
    touched.clear();
    let (first, count) = span;
    if count == 0 && task.kind() != Kind::SegmentSum {
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
    tiles: GeometryTiles<'_>,
    span: (u32, u32),
    rows: &[TableRows<'_>],
    writes: &mut Vec<(u32, Region)>,
    reads: &mut Vec<(u32, Region)>,
) {
    let (first, count) = span;
    let mut narrowed = Narrowed { writes, reads };
    match task.kind() {
        Kind::Matmul => product(values, tiles.matmul, task, first, count, &mut narrowed),
        Kind::MatmulFold => fold(values, task, first, count, &mut narrowed),
        Kind::Conv2d => convolution(values, task, first, count, &mut narrowed),
        Kind::Conv2dTile => {
            convolution_window(values, tiles.convolution, task, first, count, &mut narrowed)
        }
        Kind::Conv2dInputGrad | Kind::Conv2dTranspose
            if task.geometry() == neura_abi::strategy::INPUT_FOLD =>
        {
            fold(values, task, first, count, &mut narrowed)
        }
        Kind::Conv2dInputTile => convolution_input_window(
            values,
            tiles.convolution_input,
            task,
            first,
            count,
            &mut narrowed,
        ),
        Kind::Conv2dInputGrad | Kind::Conv2dTranspose => {
            convolution_input_grad(values, task, first, count, &mut narrowed)
        }
        Kind::Conv2dWeightGrad | Kind::Conv2dWeightTile => {
            convolution_weight_grad(values, task, first, count, &mut narrowed)
        }
        Kind::Convert => convert(values, task, first, count, &mut narrowed),
        Kind::Gather => gather(values, task, first, count, rows, &mut narrowed),
        Kind::Rope | Kind::RopeGrad => rope(values, task, first, count, &mut narrowed),
        Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad
            if task.splits() > 1 && task.input(2) != NO_VALUE =>
        {
            softmax_apply(values, task, first, count, &mut narrowed)
        }
        Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad
            if task.splits() > 1 =>
        {
            softmax_section(values, task, first, count, &mut narrowed)
        }
        Kind::SoftmaxFold | Kind::SoftmaxGradFold => {
            softmax_fold(values, task, first, count, &mut narrowed)
        }
        Kind::ClassLogSoftmax if task.splits() > 1 => {
            softmax_section(values, task, first, count, &mut narrowed)
        }
        Kind::ClassLogSoftmax => class_log_softmax(values, task, first, count, &mut narrowed),
        Kind::ClassLogSoftmaxGrad => {
            class_log_softmax_grad(values, task, first, count, &mut narrowed)
        }
        Kind::ClassLogSoftmaxFold => {
            class_log_softmax_fold(values, task, first, count, &mut narrowed)
        }
        Kind::Delta => delta(values, task, first, count, &mut narrowed),
        Kind::Softmax | Kind::SoftmaxGrad | Kind::LogSoftmax | Kind::LogSoftmaxGrad => {
            softmax(values, task, first, count, &mut narrowed)
        }
        Kind::Norm if task.splits() > 1 => norm_section(values, task, first, count, &mut narrowed),
        Kind::Norm | Kind::NormGrad => norm(values, task, first, count, &mut narrowed),
        Kind::NormFold => norm_fold(values, task, first, count, &mut narrowed),
        Kind::NormApply => norm_apply(values, task, first, count, &mut narrowed),
        Kind::Binary
        | Kind::Unary
        | Kind::Select
        | Kind::Fill
        | Kind::Noise
        | Kind::Broadcast
        | Kind::Partial => elementwise(values, task, first, count, &mut narrowed),
        Kind::Attention | Kind::AttentionQueryGrad if task.splits() > 1 => {
            attention(values, task, first, count, &mut narrowed)
        }
        Kind::AttentionFold | Kind::AttentionQueryGradFold => {
            attention_fold(values, task, first, count, &mut narrowed)
        }
        Kind::Attention
        | Kind::AttentionQueryGrad
        | Kind::AttentionKeyGrad
        | Kind::AttentionValueGrad => rowed(values, task, first, count, &mut narrowed),
        Kind::TopK | Kind::Sample if task.splits() > 1 => {
            choice_chunk(values, task, first, count, &mut narrowed)
        }
        Kind::Argmax | Kind::Categorical if task.splits() > 1 => {
            row_choice_section(values, task, first, count, &mut narrowed)
        }
        Kind::TopK | Kind::Sample | Kind::Argmax | Kind::Categorical => {
            rowed(values, task, first, count, &mut narrowed)
        }
        Kind::TopKFold | Kind::SampleFold => choice_fold(values, task, first, count, &mut narrowed),
        Kind::ChoiceFold => row_choice_fold(values, task, first, count, &mut narrowed),
        Kind::Concat => concat(values, task, first, count, &mut narrowed),
        Kind::MatmulWeightGrad => {
            matmul_weight_grad(values, tiles.matmul, task, first, count, &mut narrowed)
        }
        Kind::PoolMax2d
        | Kind::PoolMax2dInputGrad
        | Kind::PoolMean2d
        | Kind::PoolMean2dInputGrad => pool(values, task, first, count, &mut narrowed),
        Kind::SumAxis => sum_axis(values, task, first, count, &mut narrowed),
        Kind::Clear => clear(values, task, first, count, &mut narrowed),
        Kind::Extend | Kind::Layout | Kind::Slice => {
            linear(values, task, first, count, &mut narrowed)
        }
        Kind::OneHot | Kind::PrefixScan => element_run(values, task, first, count, &mut narrowed),
        Kind::SumChunk | Kind::PrefixChunk => slot(values, task, &mut narrowed),
        Kind::SegmentSum => segment_sum(values, task, &mut narrowed),
        Kind::PrefixClose
        | Kind::Scatter
        | Kind::ScatterWrite
        | Kind::Compact
        | Kind::Rows
        | Kind::Length => {}
    }
    for count in task.depends() {
        narrowed.read_whole(count);
    }
}

fn matmul_weight_grad<W: Walk, V: Values>(
    values: &V,
    tiles: &[MatmulTile],
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count != 1 || !values.owned(out) || !values.dense(out) {
        return;
    }
    let dims = values.dims(out);
    let (planes, rows, columns) = (dims[0], dims[2], dims[3]);
    let tile = tiles[task.geometry() as usize];
    let column_blocks = u64::from(columns).div_ceil(u64::from(tile.columns()));
    let tiles_per_plane = u64::from(rows).div_ceil(u64::from(tile.rows())) * column_blocks;
    if tiles_per_plane == 0 {
        return;
    }
    let plane = u64::from(task.plane()) % u64::from(planes.max(1));
    let within = u64::from(first) % tiles_per_plane;
    let base_row = (within / column_blocks) * u64::from(tile.rows());
    let base_column = (within % column_blocks) * u64::from(tile.columns());
    let band_rows = u64::from(tile.rows()).min(u64::from(rows) - base_row);
    let band_columns = u64::from(tile.columns()).min(u64::from(columns) - base_column);
    narrowed.write(
        out,
        spans(
            plane * u64::from(rows) * u64::from(columns)
                + base_row * u64::from(columns)
                + base_column,
            u64::from(columns),
            1,
            band_rows,
            band_columns,
        ),
    );
}

fn pool<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    narrowed.write(out, Region::run(u64::from(first), u64::from(count)));
}

fn linear<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    narrowed.write(out, Region::run(u64::from(first), u64::from(count)));
}

fn sum_axis<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if task.geometry() == neura_abi::strategy::WORKGROUP_SPREAD {
        if count == 0 || !values.owned(out) || !values.dense(out) {
            return;
        }
        let width = u64::from(values.dims(out)[task.slot() as usize]);
        narrowed.write(
            out,
            Region::band(u64::from(first) * width, width, width, u64::from(count)),
        );
        return;
    }
    fold(values, task, first, count, narrowed);
}

fn clear<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    let element = values.element(out);
    if element.block().is_some() {
        return;
    }
    let per_word = element.elements_per_word();
    narrowed.write(
        out,
        Region::run(u64::from(first) * per_word, u64::from(count) * per_word),
    );
}

fn concat<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    let source = task.input(0);
    let dims = values.dims(source);
    let out_dims = values.dims(out);
    let axis = task.axis() as usize;
    let inner = dims[axis + 1..]
        .iter()
        .map(|dim| u64::from(*dim))
        .product::<u64>();
    let block = u64::from(dims[axis]) * inner;
    let out_block = u64::from(out_dims[axis]) * inner;
    let shift = u64::from(task.offset()) * inner;
    let start = u64::from(first);
    let end = start + u64::from(count);
    let head = if start.is_multiple_of(block) {
        0
    } else {
        (block - start % block).min(end - start)
    };
    if head > 0 {
        narrowed.write(
            out,
            Region::run((start / block) * out_block + shift + start % block, head),
        );
    }
    let middle = (end - start - head) / block;
    if middle > 0 {
        let at = start + head;
        narrowed.write(
            out,
            Region::band((at / block) * out_block + shift, block, out_block, middle),
        );
    }
    let tail = (end - start - head) % block;
    if tail > 0 {
        let at = end - tail;
        narrowed.write(
            out,
            Region::run((at / block) * out_block + shift + at % block, tail),
        );
    }
}

fn delta<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let state = task.input(0);
    let dims = values.dims(state);
    let width = u64::from(dims[3]);
    let rows = u64::from(dims[2]);
    let area = width * width;
    let plane = u64::from(task.plane());
    let first = u64::from(first);
    let count = u64::from(count);
    if values.owned(state) && values.dense(state) {
        let start = plane * area + first;
        narrowed.read(state, Region::band(start, count, width, rows));
        narrowed.write(task.extra(), Region::band(start, count, width, rows));
    }
    let out = task.out();
    let out_dims = values.dims(out);
    if values.owned(out) && values.dense(out) {
        let columns = u64::from(out_dims[3]);
        let out_rows = u64::from(out_dims[2]);
        let start = plane * out_rows * columns + first;
        narrowed.write(out, Region::band(start, count, columns, out_rows));
    }
}

fn choice_chunk<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed,
) {
    if count == 0 {
        return;
    }
    let source = task.input(0);
    let columns = u64::from(values.dims(source)[3].max(1));
    let dims = values.dims(task.out());
    let rows = u64::from(dims[2].max(1));
    let sections = u64::from(task.splits().max(1));
    let section = u64::from(task.section());
    let first = u64::from(first);
    let count = u64::from(count);
    let first_column = columns * section / sections;
    let last_column = columns * (section + 1u64) / sections;
    narrowed.read(
        source,
        if values.dense(source) {
            Region::band(
                first * columns + first_column,
                last_column - first_column,
                columns,
                count,
            )
        } else {
            Region::Whole
        },
    );
    let block = u64::from(CANDIDATES);
    narrowed.write(
        task.out(),
        Region::run((section * rows + first) * block, count * block),
    );
    let carried = task.extra();
    let span = match task.kind() {
        Kind::TopK => block + 4,
        _ => 2 * block + 4,
    };
    narrowed.write(
        carried,
        Region::run((section * rows + first) * span, count * span),
    );
}

fn row_choice_section<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed,
) {
    if count == 0 {
        return;
    }
    let source = task.input(0);
    let columns = u64::from(values.dims(source)[3].max(1));
    let dims = values.dims(task.out());
    let rows = u64::from(dims[2].max(1));
    let sections = u64::from(task.splits().max(1));
    let section = u64::from(task.section());
    let first_column = columns * section / sections;
    let last_column = columns * (section + 1u64) / sections;
    let first = u64::from(first);
    let count = u64::from(count);
    narrowed.read(
        source,
        if values.dense(source) {
            Region::band(
                first * columns + first_column,
                last_column - first_column,
                columns,
                count,
            )
        } else {
            Region::Whole
        },
    );
    narrowed.write(
        task.out(),
        Region::run((section * rows + first) * 2u64, count * 2u64),
    );
}

fn row_choice_fold<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed,
) {
    if count == 0 {
        return;
    }
    let out = task.out();
    let dims = values.dims(out);
    let columns = u64::from(dims[3].max(1));
    let carried = values.dims(task.input(0));
    let rows = u64::from(carried[2].max(1));
    let sections = u64::from(task.splits().max(1));
    let first = u64::from(first);
    let count = u64::from(count);
    narrowed.read(
        task.input(0),
        Region::band(first * 2u64, count * 2u64, rows * 2u64, sections),
    );
    narrowed.write(out, Region::run(first * columns, count * columns));
}

fn choice_fold<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed,
) {
    if count == 0 {
        return;
    }
    let out = task.out();
    let dims = values.dims(out);
    let columns = u64::from(dims[3].max(1));
    let rows = u64::from(dims[0].max(1)) * u64::from(dims[1].max(1)) * u64::from(dims[2].max(1));
    let sections = u64::from(task.splits().max(1));
    let first = u64::from(first);
    let count = u64::from(count);
    let block = u64::from(CANDIDATES);
    narrowed.read(
        task.input(0),
        Region::band(first * block, count * block, rows * block, sections),
    );
    let (carried, span) = match task.kind() {
        Kind::TopKFold => (task.input(1), block + 4),
        _ => (task.input(4), 2 * block + 4),
    };
    narrowed.read(
        carried,
        Region::band(first * span, count * span, rows * span, sections),
    );
    narrowed.write(out, Region::run(first * columns, count * columns));
    if task.kind() == Kind::TopKFold {
        narrowed.write(task.extra(), Region::run(first * columns, count * columns));
    }
}

fn walks_its_range<V: Values>(values: &V, value: u32, out: u32) -> bool {
    values.dims(value) == values.dims(out) && values.dense(value) && values.owned(value)
}

fn gather<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    rows: &[TableRows<'_>],
    narrowed: &mut Narrowed<'_>,
) {
    element_run(values, task, first, count, narrowed);
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

fn element_run<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count != 0 && values.owned(out) && values.dense(out) {
        narrowed.write(out, Region::run(u64::from(first), u64::from(count)));
    }
}

fn rowed<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    if matches!(task.split(), Split::Segment { .. } | Split::Ragged { .. })
        || task.segments() != NO_VALUE
        || task.queries() != NO_VALUE
    {
        return;
    }
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    let columns = u64::from(values.dims(out)[3]);
    narrowed.write(
        out,
        Region::run(u64::from(first) * columns, u64::from(count) * columns),
    );
    let statistic = task.extra();
    if statistic != NO_VALUE && values.owned(statistic) && values.dense(statistic) {
        let columns = u64::from(values.dims(statistic)[3]);
        narrowed.write(
            statistic,
            Region::run(u64::from(first) * columns, u64::from(count) * columns),
        );
    }
}

fn band_at(columns: u64, sections: u64, section: u64) -> (u64, u64) {
    let shared = columns / sections;
    let rest = columns % sections;
    let start = section * shared + section.min(rest);
    let end = (section + 1) * shared + (section + 1).min(rest);
    (start, end - start)
}

fn norm_section<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    let sections = u64::from(task.splits());
    let section = u64::from(task.section());
    let at = (u64::from(first) * sections + section) * 2;
    narrowed.write(out, Region::band(at, 2, sections * 2, u64::from(count)));
}

fn norm_fold<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    narrowed.write(out, Region::run(u64::from(first) * 2, u64::from(count) * 2));
}

fn norm_apply<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    let columns = u64::from(task.axis());
    let (start, span) = band_at(columns, u64::from(task.splits()), u64::from(task.section()));
    narrowed.write(
        out,
        Region::band(
            u64::from(first) * columns + start,
            span,
            columns,
            u64::from(count),
        ),
    );
}

fn norm<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    if matches!(task.split(), Split::Segment { .. } | Split::Ragged { .. })
        || task.segments() != NO_VALUE
        || task.queries() != NO_VALUE
    {
        return;
    }
    let out = task.out();
    if count == 0 || !values.owned(out) || !values.dense(out) {
        return;
    }
    let columns = u64::from(task.axis());
    narrowed.write(
        out,
        Region::run(u64::from(first) * columns, u64::from(count) * columns),
    );
    let statistic = task.extra();
    if statistic != NO_VALUE && values.owned(statistic) && values.dense(statistic) {
        let width = match task.kind() {
            Kind::Norm => u64::from(values.dims(statistic)[3]),
            _ => columns,
        };
        narrowed.write(
            statistic,
            Region::run(u64::from(first) * width, u64::from(count) * width),
        );
    }
}

fn slot<W: Walk, V: Values>(values: &V, task: &W, narrowed: &mut Narrowed<'_>) {
    let out = task.out();
    let at = u64::from(task.slot());
    if values.owned(out) && values.dense(out) && at < values.elements(out) {
        narrowed.write(out, Region::run(at, 1));
    }
}

fn segment_sum<W: Walk, V: Values>(values: &V, task: &W, narrowed: &mut Narrowed<'_>) {
    let Split::Ragged { index, .. } = task.split() else {
        return;
    };
    let out = task.out();
    if !values.owned(out) || !values.dense(out) {
        return;
    }
    let dims = values.dims(out);
    let width = u64::from(dims[3]);
    let chunks = u64::from(dims[2]);
    let first = (u64::from(task.plane()) * chunks + u64::from(index)) * width;
    if first < values.elements(out) {
        narrowed.write(out, Region::run(first, width));
    }
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
        let base = values
            .origin(value)
            .saturating_add(self.plane_row.saturating_mul(x))
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
    let elements = bounded_elements(values, out);
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

fn attention<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed,
) {
    if count == 0 {
        return;
    }
    let query = values.dims(task.input(0));
    let width = values.dims(task.input(1))[3];
    let words = match task.kind() {
        Kind::Attention => u64::from(width) + 2,
        _ => u64::from(width),
    };
    let rows = u64::from(query[0]) * u64::from(query[1]) * u64::from(query[2]);
    let slot = u64::from(task.section()) * rows * words;
    narrowed.write(
        task.out(),
        Region::run(slot + u64::from(first) * words, u64::from(count) * words),
    );
}

fn attention_fold<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed,
) {
    if count == 0 {
        return;
    }
    let out = task.out();
    let dims = values.dims(out);
    let width = u64::from(dims[3].max(1));
    let words = match task.kind() {
        Kind::AttentionFold => width + 2,
        _ => width,
    };
    let rows = u64::from(dims[0]) * u64::from(dims[1]) * u64::from(dims[2]);
    let sections = u64::from(task.splits().max(1));
    let first_row = u64::from(first) / width;
    let last_row = (u64::from(first) + u64::from(count)).div_ceil(width);
    let span = last_row - first_row;
    narrowed.read(
        task.input(0),
        Region::band(first_row * words, span * words, rows * words, sections),
    );
    narrowed.write(
        out,
        if values.dense(out) {
            Region::run(u64::from(first), u64::from(count))
        } else {
            Region::Whole
        },
    );
    let statistic = task.extra();
    if task.kind() == Kind::AttentionFold && statistic != NO_VALUE {
        narrowed.write(
            statistic,
            if values.dense(statistic) {
                Region::run(first_row, span)
            } else {
                Region::Whole
            },
        );
    }
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
    let (first, count) = walked_elements(values.element(out), first, count);
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

fn walked_elements(element: Element, first: u32, count: u32) -> (u64, u64) {
    let Some(block) = element.block() else {
        let stride = element.elements_per_word();
        return (u64::from(first) * stride, u64::from(count) * stride);
    };
    let bytes = u64::from(block.bytes);
    let elements = u64::from(block.elements);
    let opening = u64::from(first) * WORD_BYTES;
    let closing = (u64::from(first) + u64::from(count)) * WORD_BYTES;
    let first = opening / bytes * elements;
    let end = closing.div_ceil(bytes) * elements;
    (first, end - first)
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

fn softmax_section<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    if count == 0 {
        return;
    }
    let source = task.input(0);
    let columns = u64::from(values.dims(source)[3]);
    let sections = u64::from(task.splits().max(1));
    let section = u64::from(task.section());
    let first_column = columns * section / sections;
    let width = columns * (section + 1) / sections - first_column;
    let rows = u64::from(first);
    let count = u64::from(count);
    let words = u64::from(softmax_words(task.kind()));
    let stats = task.out();
    narrowed.write(
        stats,
        Region::band(
            rows * sections * words + section * words,
            words,
            sections * words,
            count,
        ),
    );
    for value in task.reads() {
        let region = if walks_its_range(values, value, source) {
            Region::band(rows * columns + first_column, width, columns, count)
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn softmax_fold<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    if count == 0 {
        return;
    }
    let stats = task.input(0);
    let folded = task.out();
    let sections = u64::from(values.dims(stats)[3]);
    let held = u64::from(values.dims(folded)[3]);
    narrowed.read(
        stats,
        Region::run(u64::from(first) * sections, u64::from(count) * sections),
    );
    narrowed.write(
        folded,
        Region::run(u64::from(first) * held, u64::from(count) * held),
    );
}

fn softmax_apply<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    if count == 0 {
        return;
    }
    let source = task.input(0);
    let columns = u64::from(values.dims(source)[3]);
    let sections = u64::from(task.splits().max(1));
    let section = u64::from(task.section());
    let first_column = columns * section / sections;
    let width = columns * (section + 1) / sections - first_column;
    let rows = u64::from(first);
    let count = u64::from(count);
    let held = u64::from(values.dims(task.input(2))[3]);
    narrowed.read(task.input(2), Region::run(rows * held, count * held));
    narrowed.write(
        task.out(),
        Region::band(rows * columns + first_column, width, columns, count),
    );
    for value in task.reads() {
        if value == task.input(2) {
            continue;
        }
        let region = if walks_its_range(values, value, source) {
            Region::band(rows * columns + first_column, width, columns, count)
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn class_log_softmax<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    if count == 0 {
        return;
    }
    let source = task.input(0);
    let columns = u64::from(values.dims(source)[3]);
    let rows = u64::from(first);
    let count = u64::from(count);
    narrowed.write(task.out(), Region::run(rows, count));
    let statistic = task.extra();
    if statistic != NO_VALUE {
        narrowed.write(statistic, Region::run(rows, count));
    }
    for value in task.reads() {
        let region = if walks_its_range(values, value, task.out()) {
            Region::run(rows, count)
        } else if walks_its_range(values, value, source) {
            Region::band(rows * columns, columns, columns, count)
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn class_log_softmax_grad<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    if count == 0 {
        return;
    }
    let source = task.input(0);
    let columns = u64::from(values.dims(source)[3]);
    let sections = u64::from(task.splits().max(1));
    let section = u64::from(task.section());
    let first_column = columns * section / sections;
    let width = columns * (section + 1) / sections - first_column;
    let rows = u64::from(first);
    let count = u64::from(count);
    narrowed.write(
        task.out(),
        Region::band(rows * columns + first_column, width, columns, count),
    );
    for value in task.reads() {
        let region = if value == source || walks_its_range(values, value, task.out()) {
            Region::band(rows * columns + first_column, width, columns, count)
        } else if values.dims(value)[3] == 1 {
            Region::run(rows, count)
        } else {
            Region::Whole
        };
        narrowed.read(value, region);
    }
}

fn class_log_softmax_fold<W: Walk, V: Values>(
    values: &V,
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    if count == 0 {
        return;
    }
    let stats = task.input(0);
    let source = task.input(1);
    let words = u64::from(values.dims(stats)[3]);
    let columns = u64::from(values.dims(source)[3]);
    let rows = u64::from(first);
    let count = u64::from(count);
    narrowed.read(stats, Region::run(rows * words, count * words));
    narrowed.read(
        source,
        Region::band(rows * columns, columns, columns, count),
    );
    narrowed.read(task.input(2), Region::run(rows, count));
    narrowed.write(task.out(), Region::run(rows, count));
    let statistic = task.extra();
    if statistic != NO_VALUE {
        narrowed.write(statistic, Region::run(rows, count));
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

fn convolution_window<W: Walk, V: Values>(
    values: &V,
    windows: &[ConvolutionTile],
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    let input = task.input(0);
    let filter = task.input(1);
    let dims = values.dims(out);
    let input_dims = values.dims(input);
    let filter_dims = values.dims(filter);
    let tile = windows[task.geometry() as usize];
    let local = filter_dims[1];
    let channels = input_dims[1];
    if local == 0 || channels == 0 || dims[0] == 0 || dims[1] == 0 {
        narrowed.write(out, Region::empty());
        return;
    }
    let groups = (channels / local).max(1);
    let out_per_group = (dims[1] / groups).max(1);
    let out_blocks = out_per_group.div_ceil(tile.outputs());
    let row_blocks = dims[2].div_ceil(tile.rows());
    let column_blocks = dims[3].div_ceil(tile.columns());
    let per_batch = groups * out_blocks * row_blocks * column_blocks;
    let out_strides = values.strides(out);
    let input_strides = values.strides(input);
    let filter_strides = values.strides(filter);
    let out_origin = values.origin(out);
    let input_origin = values.origin(input);
    let filter_origin = values.origin(filter);
    let window = tile.window();
    let pad = task.window();
    let tap = window.taps();
    let mut hull = None::<(u64, u64)>;
    for index in first..first + count {
        let batch = index / per_batch;
        let within = index % per_batch;
        let group = within / (out_blocks * row_blocks * column_blocks);
        let k_block = within / (row_blocks * column_blocks) % out_blocks;
        let p_block = within / column_blocks % row_blocks;
        let q_block = within % column_blocks;
        let p0 = p_block * tile.rows();
        let q0 = q_block * tile.columns();
        let k0 = group * out_per_group + k_block * tile.outputs();
        let rows = tile.rows().min(dims[2] - p0);
        let columns = tile.columns().min(dims[3] - q0);
        for output in 0..tile.outputs().min(dims[1] - k0) {
            let base = out_origin
                + u64::from(batch) * u64::from(out_strides[0])
                + u64::from(k0 + output) * u64::from(out_strides[1])
                + u64::from(p0) * u64::from(out_strides[2])
                + u64::from(q0) * u64::from(out_strides[3]);
            let region = spans(
                base,
                u64::from(out_strides[2]),
                u64::from(out_strides[3]),
                u64::from(rows),
                u64::from(columns),
            );
            hull = Some(match hull {
                None => (base, base + region_hull(region)),
                Some((start, end)) => (start.min(base), end.max(base + region_hull(region))),
            });
            narrowed.write(out, region);
        }
        let row_start = i64::from(p0) * i64::from(window.stride_rows()) - i64::from(pad.pad_rows());
        let column_start =
            i64::from(q0) * i64::from(window.stride_columns()) - i64::from(pad.pad_columns());
        let rows = clip(
            row_start,
            window.span_rows(tile.rows()),
            i64::from(input_dims[2]),
        );
        let columns = clip(
            column_start,
            window.span_columns(tile.columns()),
            i64::from(input_dims[3]),
        );
        if rows.1 > 0 && columns.1 > 0 {
            for channel in 0..local {
                let base = input_origin
                    + u64::from(batch) * u64::from(input_strides[0])
                    + u64::from(group * local + channel) * u64::from(input_strides[1])
                    + rows.0 as u64 * u64::from(input_strides[2])
                    + columns.0 as u64 * u64::from(input_strides[3]);
                narrowed.read(
                    input,
                    spans(
                        base,
                        u64::from(input_strides[2]),
                        u64::from(input_strides[3]),
                        rows.1,
                        columns.1,
                    ),
                );
            }
        }
        for output in 0..tile.outputs().min(dims[1] - k0) {
            let base = filter_origin + u64::from(k0 + output) * u64::from(filter_strides[0]);
            narrowed.read(
                filter,
                Region::run(base, filter_hull(&filter_strides, local, tap)),
            );
        }
    }
    if let Some((start, end)) = hull {
        chained_reads(
            values,
            task,
            out,
            Region::run(start, end - start + 1),
            narrowed,
        );
    }
}

fn filter_hull(strides: &[u32; 4], local: u32, taps: u32) -> u64 {
    (u64::from(local) - 1).saturating_mul(u64::from(strides[1]))
        + u64::from(taps - 1).saturating_mul(u64::from(strides[3]))
        + 1
}

fn region_hull(region: Region) -> u64 {
    match region {
        Region::Empty => 0,
        Region::Whole => panic!("a window narrows no whole walk"),
        Region::Run { count, .. } => count - 1,
        Region::Band {
            span,
            stride,
            count,
            ..
        } => (count - 1) * stride + span - 1,
    }
}

fn clip(start: i64, span: u32, bound: i64) -> (i64, u64) {
    let end = start + i64::from(span);
    let first = start.max(0);
    let last = end.min(bound);
    if last <= first {
        return (0, 0);
    }
    (first, (last - first) as u64)
}

fn convolution_input_window<W: Walk, V: Values>(
    values: &V,
    windows: &[ConvolutionInputTile],
    task: &W,
    first: u32,
    count: u32,
    narrowed: &mut Narrowed<'_>,
) {
    let out = task.out();
    let filter = task.input(0);
    let gradient = task.input(1);
    let dims = values.dims(out);
    let filter_dims = values.dims(filter);
    let gradient_dims = values.dims(gradient);
    let tile = windows[task.geometry() as usize];
    let locals = filter_dims[1];
    if locals == 0 || dims[1] == 0 {
        narrowed.write(out, Region::empty());
        return;
    }
    let groups = dims[1] / locals;
    let outputs = filter_dims[0] / groups;
    let local_blocks = locals.div_ceil(tile.locals());
    let row_blocks = dims[2].div_ceil(tile.rows());
    let column_blocks = dims[3].div_ceil(tile.columns());
    let per_batch = groups * local_blocks * row_blocks * column_blocks;
    let out_strides = values.strides(out);
    let filter_strides = values.strides(filter);
    let gradient_strides = values.strides(gradient);
    let out_origin = values.origin(out);
    let filter_origin = values.origin(filter);
    let gradient_origin = values.origin(gradient);
    let window = tile.window();
    let pad = task.window();
    let tap = window.taps();
    let mut hull = None::<(u64, u64)>;
    for index in first..first + count {
        let batch = index / per_batch;
        let within = index % per_batch;
        let column_block = within % column_blocks;
        let row_block = within / column_blocks % row_blocks;
        let local_block = within / (column_blocks * row_blocks) % local_blocks;
        let group = within / (column_blocks * row_blocks * local_blocks);
        let p0 = row_block * tile.rows();
        let q0 = column_block * tile.columns();
        let local0 = local_block * tile.locals();
        let rows = tile.rows().min(dims[2] - p0);
        let columns = tile.columns().min(dims[3] - q0);
        let inputs = tile.locals().min(locals - local0);
        for local in 0..inputs {
            let base = out_origin
                + u64::from(batch) * u64::from(out_strides[0])
                + u64::from(group * locals + local0 + local) * u64::from(out_strides[1])
                + u64::from(p0) * u64::from(out_strides[2])
                + u64::from(q0) * u64::from(out_strides[3]);
            let region = spans(
                base,
                u64::from(out_strides[2]),
                u64::from(out_strides[3]),
                u64::from(rows),
                u64::from(columns),
            );
            hull = Some(match hull {
                None => (base, base + region_hull(region)),
                Some((start, end)) => (start.min(base), end.max(base + region_hull(region))),
            });
            narrowed.write(out, region);
        }
        let row_start = i64::from(p0) + i64::from(pad.pad_rows())
            - i64::from(window.reach_rows() - 1) * i64::from(window.dilation_rows());
        let column_start = i64::from(q0) + i64::from(pad.pad_columns())
            - i64::from(window.reach_columns() - 1) * i64::from(window.dilation_columns());
        let rows = clip(
            row_start,
            window.span_rows(tile.rows()),
            i64::from(gradient_dims[2]),
        );
        let columns = clip(
            column_start,
            window.span_columns(tile.columns()),
            i64::from(gradient_dims[3]),
        );
        if rows.1 > 0 && columns.1 > 0 {
            for output in 0..outputs {
                let base = gradient_origin
                    + u64::from(batch) * u64::from(gradient_strides[0])
                    + u64::from(group * outputs + output) * u64::from(gradient_strides[1])
                    + rows.0 as u64 * u64::from(gradient_strides[2])
                    + columns.0 as u64 * u64::from(gradient_strides[3]);
                narrowed.read(
                    gradient,
                    spans(
                        base,
                        u64::from(gradient_strides[2]),
                        u64::from(gradient_strides[3]),
                        rows.1,
                        columns.1,
                    ),
                );
            }
        }
        if outputs > 0 && inputs > 0 {
            let span = (u64::from(inputs) - 1) * u64::from(filter_strides[1])
                + (u64::from(tap) - 1) * u64::from(filter_strides[3])
                + 1;
            let base = filter_origin
                + u64::from(group * outputs) * u64::from(filter_strides[0])
                + u64::from(local0) * u64::from(filter_strides[1]);
            narrowed.read(
                filter,
                Region::band(base, span, u64::from(filter_strides[0]), u64::from(outputs)),
            );
        }
    }
    if let Some((start, end)) = hull {
        chained_reads(
            values,
            task,
            out,
            Region::run(start, end - start + 1),
            narrowed,
        );
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
    let runs = channel_runs(values, out, first, count);
    match (runs.is_empty(), filter_planes(values, filter)) {
        (false, Some((per_channel, _))) => {
            for (first_channel, last_channel) in runs.iter() {
                narrowed.read(
                    filter,
                    Region::run(
                        u64::from(first_channel) * per_channel,
                        u64::from(last_channel - first_channel + 1) * per_channel,
                    ),
                );
            }
        }
        _ => narrowed.read(filter, Region::Whole),
    }
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
    let extra = task.extra();
    let filter = task.input(0);
    let range = Region::run(u64::from(first), u64::from(count));
    if extra != NO_VALUE {
        narrowed.write(out, Region::empty());
        narrowed.write(
            extra,
            Region::run(
                u64::from(task.slot()) * bounded_elements(values, out) + u64::from(first),
                u64::from(count),
            ),
        );
    } else {
        narrowed.write(out, range);
    }
    let runs = channel_runs(values, out, first, count);
    if runs.is_empty() {
        narrowed.read(filter, Region::Whole);
    }
    for (first_channel, last_channel) in runs.iter() {
        name_filter_bands(
            values,
            task,
            out,
            filter,
            first_channel,
            last_channel,
            narrowed,
        );
    }
    chained_reads(values, task, out, range, narrowed);
}

fn bounded_elements<V: Values>(values: &V, value: u32) -> u64 {
    values
        .bounds(value)
        .iter()
        .map(|dim| u64::from(*dim))
        .product()
}

fn name_filter_bands<W: Walk, V: Values>(
    values: &V,
    task: &W,
    out: u32,
    filter: u32,
    first_channel: u32,
    last_channel: u32,
    narrowed: &mut Narrowed<'_>,
) {
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
    let chunks = u64::from(task.splits().max(1));
    let per_chunk = out_channels.div_ceil(chunks);
    let first_block = u64::from(task.slot()).min(chunks - 1) * per_chunk;
    let last_block = (first_block + per_chunk).min(out_channels);
    if last_block <= first_block {
        return;
    }
    let block = |group: u64, local: (u64, u64)| {
        Region::band(
            (group * out_channels + first_block) * per_channel + local.0 * per_local,
            (local.1 - local.0 + 1) * per_local,
            per_channel,
            last_block - first_block,
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
                ((first_group + 1) * out_channels + first_block) * per_channel,
                (last_block - first_block) * per_channel,
                out_channels * per_channel,
                last_group - first_group - 1,
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

#[derive(Clone, Copy, Default)]
struct ChannelRuns {
    runs: [(u32, u32); 2],
    held: usize,
}

impl ChannelRuns {
    fn push(&mut self, first: u64, last: u64) {
        self.runs[self.held] = (first as u32, last as u32);
        self.held += 1;
    }

    fn is_empty(&self) -> bool {
        self.held == 0
    }

    fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        self.runs[..self.held].iter().copied()
    }
}

fn channel_runs<V: Values>(values: &V, out: u32, first: u32, count: u32) -> ChannelRuns {
    let mut runs = ChannelRuns::default();
    let dims = values.dims(out);
    let Some(plane) = u64::from(dims[2]).checked_mul(u64::from(dims[3])) else {
        return runs;
    };
    let channels = u64::from(dims[1]);
    if plane == 0 || channels == 0 || count == 0 {
        return runs;
    }
    let first = u64::from(first);
    let Some(last) = first.checked_add(u64::from(count)).map(|end| end - 1) else {
        return runs;
    };
    let a = (first / plane) % channels;
    let b = (last / plane) % channels;
    if a <= b {
        runs.push(a, b);
    } else {
        runs.push(a, channels - 1);
        runs.push(0, b);
    }
    runs
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
