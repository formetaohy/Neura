use neura_abi::{DeviceModule, Kind, WORD_BYTES};
use std::collections::HashMap;

pub const CHOICE_LOCAL_CEILING: u32 = 4;
pub const CHOICE_CAP: u32 = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ChoiceLayout {
    owners: u32,
    local: u32,
}

impl ChoiceLayout {
    pub fn of(workgroup: u32, scratch_bytes: u64) -> Self {
        let owners = workgroup.min(CHOICE_CAP);
        let room = (scratch_bytes / WORD_BYTES) as u32;
        let fixed = 2 * workgroup + 2 * neura_abi::CANDIDATES + neura_abi::CHOICE_MERGE_WORDS;
        let mut local = 1;
        while local < CHOICE_LOCAL_CEILING && fixed + 2 * owners * (local + 1) <= room {
            local += 1;
        }
        let layout = Self { owners, local };
        assert!(
            u64::from(layout.words(workgroup)) * WORD_BYTES <= scratch_bytes,
            "a choice of {owners} owners and {local} ranks spans {} bytes of workgroup scratch where its profile offers {scratch_bytes}",
            u64::from(layout.words(workgroup)) * WORD_BYTES,
        );
        layout
    }

    pub const fn owners(self) -> u32 {
        self.owners
    }

    pub const fn local(self) -> u32 {
        self.local
    }

    pub const fn entries(self) -> u32 {
        self.owners * self.local
    }

    pub const fn sorted_width(self) -> u32 {
        1 << (u32::BITS - 1 - self.entries().leading_zeros())
    }

    pub const fn words(self, workgroup: u32) -> u32 {
        2 * workgroup
            + 2 * neura_abi::CANDIDATES
            + 2 * self.entries()
            + neura_abi::CHOICE_MERGE_WORDS
    }

    pub const fn levels(self) -> u32 {
        self.owners.next_power_of_two().trailing_zeros()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct AttentionTile {
    keys: u32,
    width: u32,
    slices: u32,
    groups: u32,
    panels: u32,
}

impl AttentionTile {
    pub const REGISTER_CEILING: u32 = 200;
    pub const KEYS_CEILING: u32 = 16;
    pub const GROUP_KEYS_CEILING: u32 = 32;
    pub const GROUPS_CEILING: u32 = 32;
    pub const KEYS_PER_GROUP: u32 = 2;
    pub const WIDTH_PER_THREAD: u32 = 64;
    pub const REDUCTIONS: u32 = 1;
    pub const COMBINE_WORDS: u64 = 512;
    pub const PANELS: u32 = 2;
    pub const RECOMPUTED_PANELS: u32 = 3;

    pub fn new(keys: u32, width: u32, groups: u32, panels: u32) -> Self {
        Self::within(
            keys,
            width,
            groups,
            panels,
            Self::slices_of(width),
            Self::REGISTER_CEILING,
        )
    }

    fn within(
        keys: u32,
        width: u32,
        groups: u32,
        panels: u32,
        slices: u32,
        registers: u32,
    ) -> Self {
        let ceiling = if groups == 1 {
            Self::KEYS_CEILING
        } else {
            Self::GROUP_KEYS_CEILING
        };
        assert!(
            keys > 0 && keys <= ceiling && width > 0,
            "an attention tile of {groups} groups walks {keys} keys of {width} numbers where at most {ceiling} keys fit",
        );
        assert!(
            groups > 0 && groups <= Self::GROUPS_CEILING && keys.is_multiple_of(groups),
            "an attention tile hands its {keys} keys to {groups} groups of its workgroup, and no whole number of keys reaches every group",
        );
        assert!(
            (Self::PANELS..=Self::RECOMPUTED_PANELS).contains(&panels),
            "an attention tile stages {panels} operands of its keys, and a body stages {} of them or {}",
            Self::PANELS,
            Self::RECOMPUTED_PANELS,
        );
        let slice_width = width.div_ceil(slices);
        let key_group = keys / groups;
        assert!(
            3 * slice_width + 2 * key_group <= registers,
            "an attention of width {width} spreads one query row over {slices} threads of {slice_width} numbers beside {key_group} keys of one group, and the {} numbers of that row, its gradients and those keys still outrun the {registers} a device thread holds here",
            3 * slice_width + 2 * key_group,
        );
        Self {
            keys,
            width,
            slices,
            groups,
            panels,
        }
    }

    pub fn fit(
        pool: u64,
        workgroup: u32,
        registers: u32,
        width: u32,
        rows: u32,
        panels: u32,
    ) -> Self {
        assert!(rows > 0, "an attention tile serves no row per task");
        Self::try_fit(pool, workgroup, registers, width, rows, panels).unwrap_or_else(|| {
            panic!(
                "an attention of width {width} fits no key of {workgroup} threads beside the {pool} bytes of workgroup scratch its profile offers",
            )
        })
    }

    pub fn try_fit(
        pool: u64,
        workgroup: u32,
        registers: u32,
        width: u32,
        rows: u32,
        panels: u32,
    ) -> Option<Self> {
        if rows == 0 {
            return None;
        }
        let ceiling = registers.min(Self::REGISTER_CEILING);
        let slices = Self::slices_within(width, ceiling);
        if slices > workgroup {
            return None;
        }
        let slice_width = width.div_ceil(slices);
        if 3 * slice_width + 2 * Self::KEYS_PER_GROUP > ceiling {
            return None;
        }
        let wanted = workgroup
            .div_ceil(rows.saturating_mul(slices))
            .next_power_of_two()
            .clamp(1, Self::GROUPS_CEILING);
        let mut panels = panels.clamp(Self::PANELS, Self::RECOMPUTED_PANELS);
        loop {
            let mut groups = wanted;
            loop {
                if let Some(tile) =
                    Self::holding(pool, workgroup, ceiling, width, slices, groups, panels)
                {
                    return Some(tile);
                }
                if groups == 1 {
                    break;
                }
                groups /= 2;
            }
            if panels == Self::PANELS {
                return None;
            }
            panels = Self::PANELS;
        }
    }

    fn holding(
        pool: u64,
        workgroup: u32,
        registers: u32,
        width: u32,
        slices: u32,
        groups: u32,
        panels: u32,
    ) -> Option<Self> {
        if groups.saturating_mul(slices) > workgroup {
            return None;
        }
        let slice_width = width.div_ceil(slices);
        let per_thread = (registers - 3 * slice_width) / 2;
        let ceiling = if groups == 1 {
            Self::KEYS_CEILING
        } else {
            Self::GROUP_KEYS_CEILING
        };
        let mut keys = ceiling.min(groups * per_thread);
        keys -= keys % groups;
        while keys >= groups * Self::KEYS_PER_GROUP {
            let tile = Self::within(keys, width, groups, panels, slices, registers);
            if tile.shared_bytes(workgroup) <= pool {
                return Some(tile);
            }
            keys -= groups;
        }
        None
    }

    fn slices_of(width: u32) -> u32 {
        width.div_ceil(Self::WIDTH_PER_THREAD).next_power_of_two()
    }

    fn slices_within(width: u32, registers: u32) -> u32 {
        let per_thread = ((registers.saturating_sub(2 * Self::KEYS_PER_GROUP)) / 3).max(1);
        width
            .div_ceil(per_thread)
            .next_power_of_two()
            .max(Self::slices_of(width))
    }

    pub const fn keys(self) -> u32 {
        self.keys
    }

    pub const fn width(self) -> u32 {
        self.width
    }

    pub const fn slices(self) -> u32 {
        self.slices
    }

    pub const fn groups(self) -> u32 {
        self.groups
    }

    pub const fn panels(self) -> u32 {
        self.panels
    }

    pub const fn words(self) -> [u32; 5] {
        [self.keys, self.width, self.slices, self.groups, self.panels]
    }

    pub fn of_words(words: [u32; 5]) -> Option<Self> {
        let [keys, width, slices, groups, panels] = words;
        let ceiling = if groups == 1 {
            Self::KEYS_CEILING
        } else {
            Self::GROUP_KEYS_CEILING
        };
        if keys == 0
            || keys > ceiling
            || width == 0
            || slices == 0
            || groups == 0
            || groups > Self::GROUPS_CEILING
            || !keys.is_multiple_of(groups)
            || !(Self::PANELS..=Self::RECOMPUTED_PANELS).contains(&panels)
        {
            return None;
        }
        let slice_width = width.div_ceil(slices);
        let key_group = keys / groups;
        if 3 * slice_width + 2 * key_group > Self::REGISTER_CEILING {
            return None;
        }
        Some(Self {
            keys,
            width,
            slices,
            groups,
            panels,
        })
    }

    pub const fn slice_width(self) -> u32 {
        self.width.div_ceil(self.slices)
    }

    pub const fn key_group(self) -> u32 {
        self.keys / self.groups
    }

    pub const fn rows(self, workgroup: u32) -> u32 {
        workgroup / (self.groups * self.slices)
    }

    pub const fn combine_rows(self, workgroup: u32) -> u32 {
        let rows = self.rows(workgroup);
        let room = (Self::COMBINE_WORDS / self.width as u64) as u32;
        let room = if room == 0 { 1 } else { room };
        if rows < room { rows } else { room }
    }

    pub const fn stride(self) -> u32 {
        if self.groups == 1 {
            return self.width;
        }
        self.width + 1
    }

    pub const fn stage_words(self) -> u32 {
        self.keys * self.stride()
    }

    pub const fn reduction_words(self, workgroup: u32) -> u32 {
        if self.slices == 1 {
            return 0;
        }
        self.rows(workgroup) * self.keys * Self::REDUCTIONS * self.slices
    }

    pub const fn combine_words(self, workgroup: u32) -> u32 {
        if self.groups == 1 {
            return 0;
        }
        self.combine_rows(workgroup) * self.stride() + self.rows(workgroup) * self.groups * 2
    }

    pub const fn row_dot_words(self, workgroup: u32) -> u32 {
        if self.slices == 1 {
            return 0;
        }
        self.rows(workgroup) * self.groups * self.slices
    }

    pub const fn scratch_words(self, workgroup: u32) -> u32 {
        self.panels * self.stage_words()
            + self.reduction_words(workgroup)
            + self.combine_words(workgroup)
            + self.row_dot_words(workgroup)
    }

    pub const fn shared_bytes(self, workgroup: u32) -> u64 {
        self.scratch_words(workgroup) as u64 * WORD_BYTES
    }

    pub const fn registers(self) -> u32 {
        3 * self.slice_width() + 2 * self.key_group()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ConvolutionWindow {
    reach_rows: u32,
    reach_columns: u32,
    stride_rows: u32,
    stride_columns: u32,
    dilation_rows: u32,
    dilation_columns: u32,
}

impl ConvolutionWindow {
    pub fn new(
        reach_rows: u32,
        reach_columns: u32,
        stride_rows: u32,
        stride_columns: u32,
        dilation_rows: u32,
        dilation_columns: u32,
    ) -> Self {
        assert!(
            reach_rows > 0
                && reach_columns > 0
                && stride_rows > 0
                && stride_columns > 0
                && dilation_rows > 0
                && dilation_columns > 0,
            "a window of {reach_rows} by {reach_columns} taps at stride {stride_rows} by {stride_columns} and dilation {dilation_rows} by {dilation_columns} walks no tap",
        );
        Self {
            reach_rows,
            reach_columns,
            stride_rows,
            stride_columns,
            dilation_rows,
            dilation_columns,
        }
    }

    pub const fn reach_rows(self) -> u32 {
        self.reach_rows
    }

    pub const fn reach_columns(self) -> u32 {
        self.reach_columns
    }

    pub const fn stride_rows(self) -> u32 {
        self.stride_rows
    }

    pub const fn stride_columns(self) -> u32 {
        self.stride_columns
    }

    pub const fn dilation_rows(self) -> u32 {
        self.dilation_rows
    }

    pub const fn dilation_columns(self) -> u32 {
        self.dilation_columns
    }

    pub const fn words(self) -> [u32; 6] {
        [
            self.reach_rows,
            self.reach_columns,
            self.stride_rows,
            self.stride_columns,
            self.dilation_rows,
            self.dilation_columns,
        ]
    }

    pub fn of_words(words: [u32; 6]) -> Option<Self> {
        let [
            reach_rows,
            reach_columns,
            stride_rows,
            stride_columns,
            dilation_rows,
            dilation_columns,
        ] = words;
        if reach_rows == 0
            || reach_columns == 0
            || stride_rows == 0
            || stride_columns == 0
            || dilation_rows == 0
            || dilation_columns == 0
        {
            return None;
        }
        Some(Self {
            reach_rows,
            reach_columns,
            stride_rows,
            stride_columns,
            dilation_rows,
            dilation_columns,
        })
    }

    pub const fn taps(self) -> u32 {
        self.reach_rows * self.reach_columns
    }

    pub const fn span_rows(self, rows: u32) -> u32 {
        (rows - 1) * self.stride_rows + (self.reach_rows - 1) * self.dilation_rows + 1
    }

    pub const fn span_columns(self, columns: u32) -> u32 {
        (columns - 1) * self.stride_columns + (self.reach_columns - 1) * self.dilation_columns + 1
    }

    pub const fn slides_columns(self) -> bool {
        self.stride_columns == 1 && self.dilation_columns == 1
    }

    pub const fn strides_one(self) -> bool {
        self.stride_rows == 1 && self.stride_columns == 1
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConvolutionShape {
    pub batch: u32,
    pub groups: u32,
    pub rows: u32,
    pub columns: u32,
    pub outputs: u32,
    pub channels: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ConvolutionInputShape {
    pub batch: u32,
    pub groups: u32,
    pub rows: u32,
    pub columns: u32,
    pub outputs: u32,
    pub locals: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ConvolutionTile {
    window: ConvolutionWindow,
    rows: u32,
    columns: u32,
    channels: u32,
    outputs: u32,
    thread_columns: u32,
    thread_outputs: u32,
}

impl ConvolutionTile {
    pub const THREAD_STEPS: [(u32, u32); 9] = [
        (4, 4),
        (4, 2),
        (4, 1),
        (2, 4),
        (2, 2),
        (2, 1),
        (1, 4),
        (1, 2),
        (1, 1),
    ];
    pub const COLUMN_STEPS: [u32; 4] = [32, 16, 8, 4];
    pub const OUTPUT_STEPS: [u32; 4] = [16, 8, 4, 1];
    pub const CHANNEL_STEPS: [u32; 5] = [16, 8, 4, 2, 1];
    pub const ROW_STEPS: [u32; 5] = [16, 8, 4, 2, 1];

    pub fn new(
        window: ConvolutionWindow,
        rows: u32,
        columns: u32,
        channels: u32,
        outputs: u32,
        thread_columns: u32,
        thread_outputs: u32,
    ) -> Self {
        assert!(
            rows > 0 && columns > 0 && channels > 0 && outputs > 0,
            "a convolution tile walks {rows} rows of {columns} numbers through {channels} channels into {outputs} outputs, and a tile covers at least one of each",
        );
        assert!(
            Self::THREAD_STEPS.contains(&(thread_columns, thread_outputs)),
            "a convolution tile hands {thread_columns} by {thread_outputs} outputs to one thread, where a thread holds {} of them",
            Self::THREAD_STEPS
                .iter()
                .map(|(columns, outputs)| format!("{columns}x{outputs}"))
                .collect::<Vec<String>>()
                .join("/"),
        );
        Self {
            window,
            rows,
            columns,
            channels,
            outputs,
            thread_columns,
            thread_outputs,
        }
    }

    pub fn fit(
        pool: u64,
        workgroup: u32,
        wanted: u32,
        weight_words: u64,
        shape: ConvolutionShape,
        window: ConvolutionWindow,
    ) -> Option<Self> {
        if shape.batch == 0
            || shape.groups == 0
            || shape.rows == 0
            || shape.columns == 0
            || shape.outputs == 0
            || shape.channels == 0
        {
            return None;
        }
        let taps = u64::from(window.taps());
        let mut best: Option<(f64, ConvolutionTile)> = None;
        for (thread_columns, thread_outputs) in Self::THREAD_STEPS {
            for column_step in Self::COLUMN_STEPS {
                let columns = column_step.min(shape.columns);
                let column_groups = columns.div_ceil(thread_columns);
                for output_step in Self::OUTPUT_STEPS {
                    let outputs = output_step.min(shape.outputs);
                    if u64::from(outputs) * u64::from(shape.channels) * taps > weight_words {
                        continue;
                    }
                    let output_groups = outputs.div_ceil(thread_outputs);
                    let lanes = column_groups * output_groups;
                    for row_step in Self::ROW_STEPS {
                        let ceiling = (workgroup / lanes).max(1);
                        let rows = row_step.min(shape.rows).min(ceiling);
                        let span_rows = window.span_rows(rows);
                        let span_columns = window.span_columns(columns);
                        let area = u64::from(span_rows) * u64::from(span_columns);
                        let Some(channels) =
                            Self::staged(pool, area, shape.channels, outputs, taps)
                        else {
                            continue;
                        };
                        let candidate = Self::new(
                            window,
                            rows,
                            columns,
                            channels,
                            outputs,
                            thread_columns,
                            thread_outputs,
                        );
                        let score = candidate.score(workgroup, wanted, shape);
                        if best.as_ref().is_none_or(|(kept, led)| {
                            score > *kept || (score == *kept && candidate.rows > led.rows)
                        }) {
                            best = Some((score, candidate));
                        }
                    }
                }
            }
        }
        best.map(|(_, tile)| tile)
    }

    fn staged(pool: u64, area: u64, local: u32, outputs: u32, taps: u64) -> Option<u32> {
        for step in Self::CHANNEL_STEPS {
            let channels = step.min(local);
            let words =
                u64::from(channels) * area + u64::from(outputs) * u64::from(channels) * taps;
            if words * neura_abi::WORD_BYTES <= pool {
                return Some(channels);
            }
        }
        None
    }

    fn score(self, workgroup: u32, wanted: u32, shape: ConvolutionShape) -> f64 {
        let columns = self.thread_columns.min(self.columns);
        let outputs = self.thread_outputs.min(self.outputs);
        let fma = f64::from(columns * outputs * self.window.reach_columns);
        let window = if self.window.slides_columns() {
            f64::from(columns + self.window.reach_columns - 1)
        } else {
            f64::from(columns * self.window.reach_columns)
        };
        let loads = window + f64::from(outputs * self.window.reach_columns);
        let inner = (fma + loads) / fma;
        let macs = f64::from(self.rows * self.columns * self.outputs)
            * f64::from(shape.channels)
            * f64::from(self.window.taps());
        let staged = f64::from(self.span_rows() * self.span_columns()) * f64::from(shape.channels)
            + f64::from(self.outputs * shape.channels * self.window.taps());
        let throughput = macs / (inner * macs + staged);
        let tiles = self.tiles(shape);
        let fill = f64::from(tiles).min(f64::from(wanted)) / f64::from(wanted.max(1));
        let lanes =
            self.columns.div_ceil(self.thread_columns) * self.outputs.div_ceil(self.thread_outputs);
        let busy = f64::from(self.rows * lanes).min(f64::from(workgroup)) / f64::from(workgroup);
        throughput * (fill * busy).sqrt()
    }

    pub fn tiles(self, shape: ConvolutionShape) -> u32 {
        shape.batch
            * shape.groups
            * shape.outputs.div_ceil(self.outputs)
            * shape.rows.div_ceil(self.rows)
            * shape.columns.div_ceil(self.columns)
    }

    pub const fn window(self) -> ConvolutionWindow {
        self.window
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.columns
    }

    pub const fn channels(self) -> u32 {
        self.channels
    }

    pub const fn outputs(self) -> u32 {
        self.outputs
    }

    pub const fn thread_columns(self) -> u32 {
        self.thread_columns
    }

    pub const fn thread_outputs(self) -> u32 {
        self.thread_outputs
    }

    pub const fn words(self) -> [u32; 12] {
        let window = self.window.words();
        [
            window[0],
            window[1],
            window[2],
            window[3],
            window[4],
            window[5],
            self.rows,
            self.columns,
            self.channels,
            self.outputs,
            self.thread_columns,
            self.thread_outputs,
        ]
    }

    pub fn of_words(words: [u32; 12]) -> Option<Self> {
        let window = ConvolutionWindow::of_words(words[..6].try_into().ok()?)?;
        let [
            rows,
            columns,
            channels,
            outputs,
            thread_columns,
            thread_outputs,
        ] = words[6..].try_into().ok()?;
        if rows == 0
            || columns == 0
            || channels == 0
            || outputs == 0
            || !Self::THREAD_STEPS.contains(&(thread_columns, thread_outputs))
        {
            return None;
        }
        Some(Self {
            window,
            rows,
            columns,
            channels,
            outputs,
            thread_columns,
            thread_outputs,
        })
    }

    pub const fn span_rows(self) -> u32 {
        self.window.span_rows(self.rows)
    }

    pub const fn span_columns(self) -> u32 {
        self.window.span_columns(self.columns)
    }

    pub const fn scratch_words(self) -> u32 {
        self.channels * self.span_rows() * self.span_columns()
            + self.outputs * self.channels * self.window.taps()
    }

    pub const fn items(self) -> u32 {
        self.rows
            * self.columns.div_ceil(self.thread_columns)
            * self.outputs.div_ceil(self.thread_outputs)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ConvolutionInputTile {
    window: ConvolutionWindow,
    rows: u32,
    columns: u32,
    channels: u32,
    locals: u32,
    thread_columns: u32,
    thread_locals: u32,
}

impl ConvolutionInputTile {
    pub const THREAD_STEPS: [(u32, u32); 9] = [
        (4, 4),
        (4, 2),
        (4, 1),
        (2, 4),
        (2, 2),
        (2, 1),
        (1, 4),
        (1, 2),
        (1, 1),
    ];
    pub const COLUMN_STEPS: [u32; 4] = [32, 16, 8, 4];
    pub const LOCAL_STEPS: [u32; 5] = [16, 8, 4, 2, 1];
    pub const CHANNEL_STEPS: [u32; 5] = [16, 8, 4, 2, 1];
    pub const ROW_STEPS: [u32; 5] = [16, 8, 4, 2, 1];

    pub fn new(
        window: ConvolutionWindow,
        rows: u32,
        columns: u32,
        channels: u32,
        locals: u32,
        thread_columns: u32,
        thread_locals: u32,
    ) -> Self {
        assert!(
            rows > 0 && columns > 0 && channels > 0 && locals > 0,
            "a convolution input tile walks {rows} rows of {columns} positions through {channels} gradient channels into {locals} local channels, and a tile covers at least one of each",
        );
        assert!(
            Self::THREAD_STEPS.contains(&(thread_columns, thread_locals)),
            "a convolution input tile hands {thread_columns} by {thread_locals} positions to one thread, where a thread holds one of {}",
            Self::THREAD_STEPS
                .iter()
                .map(|(columns, locals)| format!("{columns}x{locals}"))
                .collect::<Vec<String>>()
                .join("/"),
        );
        Self {
            window,
            rows,
            columns,
            channels,
            locals,
            thread_columns,
            thread_locals,
        }
    }

    pub fn fit(
        pool: u64,
        workgroup: u32,
        wanted: u32,
        weight_words: u64,
        shape: ConvolutionInputShape,
        window: ConvolutionWindow,
    ) -> Option<Self> {
        if !window.strides_one()
            || shape.batch == 0
            || shape.groups == 0
            || shape.rows == 0
            || shape.columns == 0
            || shape.outputs == 0
            || shape.locals == 0
        {
            return None;
        }
        let taps = u64::from(window.taps());
        let mut best: Option<(f64, ConvolutionInputTile)> = None;
        for (thread_columns, thread_locals) in Self::THREAD_STEPS {
            for column_step in Self::COLUMN_STEPS {
                let columns = column_step.min(shape.columns);
                let column_groups = columns.div_ceil(thread_columns);
                for local_step in Self::LOCAL_STEPS {
                    let locals = local_step.min(shape.locals);
                    let local_groups = locals.div_ceil(thread_locals);
                    let lanes = column_groups * local_groups;
                    if lanes > workgroup {
                        continue;
                    }
                    for row_step in Self::ROW_STEPS {
                        let ceiling = (workgroup / lanes).max(1);
                        let rows = row_step.min(shape.rows).min(ceiling);
                        let span_rows = window.span_rows(rows);
                        let span_columns = window.span_columns(columns);
                        let area = u64::from(span_rows) * u64::from(span_columns);
                        let Some(channels) =
                            Self::staged(pool, weight_words, area, shape.outputs, locals, taps)
                        else {
                            continue;
                        };
                        let candidate = Self::new(
                            window,
                            rows,
                            columns,
                            channels,
                            locals,
                            thread_columns,
                            thread_locals,
                        );
                        let score = candidate.score(workgroup, wanted, shape);
                        if best.as_ref().is_none_or(|(kept, led)| {
                            score > *kept || (score == *kept && candidate.rows > led.rows)
                        }) {
                            best = Some((score, candidate));
                        }
                    }
                }
            }
        }
        best.map(|(_, tile)| tile)
    }

    fn staged(
        pool: u64,
        weight_words: u64,
        area: u64,
        outputs: u32,
        locals: u32,
        taps: u64,
    ) -> Option<u32> {
        for step in Self::CHANNEL_STEPS {
            let channels = step.min(outputs);
            let filter = u64::from(channels) * u64::from(locals) * taps;
            let words = u64::from(channels) * area + filter;
            if filter <= weight_words && words * neura_abi::WORD_BYTES <= pool {
                return Some(channels);
            }
        }
        None
    }

    fn score(self, workgroup: u32, wanted: u32, shape: ConvolutionInputShape) -> f64 {
        let columns = self.thread_columns.min(self.columns);
        let locals = self.thread_locals.min(self.locals);
        let fma = f64::from(columns * locals * self.window.reach_columns());
        let loads = f64::from(columns + self.window.reach_columns() - 1)
            + f64::from(locals * self.window.reach_columns());
        let inner = (fma + loads) / fma;
        let macs = f64::from(self.rows * self.columns * shape.locals * shape.outputs)
            * f64::from(self.window.taps());
        let staged = f64::from(self.span_rows() * self.span_columns() * shape.outputs)
            + f64::from(shape.outputs * shape.locals * self.window.taps());
        let throughput = macs / (inner * macs + staged);
        let tiles = self.tiles(shape);
        let fill = f64::from(tiles).min(f64::from(wanted)) / f64::from(wanted.max(1));
        let lanes =
            self.columns.div_ceil(self.thread_columns) * self.locals.div_ceil(self.thread_locals);
        let busy = f64::from(self.rows * lanes).min(f64::from(workgroup)) / f64::from(workgroup);
        throughput * (fill * busy).sqrt()
    }

    pub fn tiles(self, shape: ConvolutionInputShape) -> u32 {
        shape.batch
            * shape.groups
            * shape.locals.div_ceil(self.locals)
            * shape.rows.div_ceil(self.rows)
            * shape.columns.div_ceil(self.columns)
    }

    pub const fn window(self) -> ConvolutionWindow {
        self.window
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.columns
    }

    pub const fn channels(self) -> u32 {
        self.channels
    }

    pub const fn locals(self) -> u32 {
        self.locals
    }

    pub const fn thread_columns(self) -> u32 {
        self.thread_columns
    }

    pub const fn thread_locals(self) -> u32 {
        self.thread_locals
    }

    pub const fn words(self) -> [u32; 12] {
        let window = self.window.words();
        [
            window[0],
            window[1],
            window[2],
            window[3],
            window[4],
            window[5],
            self.rows,
            self.columns,
            self.channels,
            self.locals,
            self.thread_columns,
            self.thread_locals,
        ]
    }

    pub fn of_words(words: [u32; 12]) -> Option<Self> {
        let window = ConvolutionWindow::of_words(words[..6].try_into().ok()?)?;
        let [
            rows,
            columns,
            channels,
            locals,
            thread_columns,
            thread_locals,
        ] = words[6..].try_into().ok()?;
        if rows == 0
            || columns == 0
            || channels == 0
            || locals == 0
            || !Self::THREAD_STEPS.contains(&(thread_columns, thread_locals))
        {
            return None;
        }
        Some(Self {
            window,
            rows,
            columns,
            channels,
            locals,
            thread_columns,
            thread_locals,
        })
    }

    pub const fn span_rows(self) -> u32 {
        self.window.span_rows(self.rows)
    }

    pub const fn span_columns(self) -> u32 {
        self.window.span_columns(self.columns)
    }

    pub const fn scratch_words(self) -> u32 {
        self.channels * self.span_rows() * self.span_columns()
            + self.channels * self.locals * self.window.taps()
            + self.thread_columns
            - 1
    }

    pub const fn items(self) -> u32 {
        self.rows
            * self.columns.div_ceil(self.thread_columns)
            * self.locals.div_ceil(self.thread_locals)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct ConvolutionWeightTile {
    window: ConvolutionWindow,
    rows: u32,
    columns: u32,
    channels: u32,
    outputs: u32,
    thread_outputs: u32,
    thread_slices: u32,
}

impl ConvolutionWeightTile {
    pub const OUTPUT_STEPS: [u32; 6] = [32, 16, 8, 4, 2, 1];
    pub const THREAD_OUTPUTS: [u32; 4] = [1, 2, 4, 8];
    pub const THREAD_SLICES: [u32; 5] = [1, 2, 4, 8, 16];
    pub const ROW_STEPS: [u32; 4] = [8, 4, 2, 1];
    pub const COLUMN_STEPS: [u32; 6] = [32, 16, 8, 4, 2, 1];
    pub const REGISTER_CEILING: u32 = 32;

    pub fn new(
        window: ConvolutionWindow,
        rows: u32,
        columns: u32,
        channels: u32,
        outputs: u32,
        thread_outputs: u32,
        thread_slices: u32,
    ) -> Self {
        assert!(
            rows > 0 && columns > 0 && channels > 0 && outputs > 0,
            "a convolution weight tile weighs {rows} rows of {columns} positions across {channels} channels into {outputs} outputs, and a tile covers at least one of each",
        );
        assert!(
            thread_outputs > 0 && thread_outputs <= outputs,
            "a convolution weight tile hands {thread_outputs} of its {outputs} outputs to one thread",
        );
        assert!(
            Self::THREAD_SLICES.contains(&thread_slices),
            "a convolution weight tile hands {thread_slices} filter slices to one thread, where a thread holds one of {}",
            Self::THREAD_SLICES
                .iter()
                .map(u32::to_string)
                .collect::<Vec<String>>()
                .join("/"),
        );
        assert!(
            thread_outputs * thread_slices <= Self::REGISTER_CEILING,
            "a convolution weight tile hands {thread_outputs} by {thread_slices} filter elements to one thread, and those {} totals outrun the {} a thread holds here",
            thread_outputs * thread_slices,
            Self::REGISTER_CEILING,
        );
        Self {
            window,
            rows,
            columns,
            channels,
            outputs,
            thread_outputs,
            thread_slices,
        }
    }

    pub fn fit(
        pool: u64,
        workgroup: u32,
        wanted: u32,
        shape: ConvolutionShape,
        window: ConvolutionWindow,
    ) -> Option<Self> {
        if shape.batch == 0
            || shape.groups == 0
            || shape.rows == 0
            || shape.columns == 0
            || shape.outputs == 0
            || shape.channels == 0
        {
            return None;
        }
        let slices = u64::from(shape.channels) * u64::from(window.taps());
        let mut best: Option<(f64, ConvolutionWeightTile)> = None;
        for output_step in Self::OUTPUT_STEPS {
            let outputs = output_step.min(shape.outputs);
            for thread_outputs in Self::THREAD_OUTPUTS {
                if thread_outputs > outputs {
                    continue;
                }
                let output_groups = outputs.div_ceil(thread_outputs);
                for thread_slices in Self::THREAD_SLICES {
                    if u64::from(thread_slices) > slices
                        || thread_outputs * thread_slices > Self::REGISTER_CEILING
                    {
                        continue;
                    }
                    let slice_groups = u32::try_from(slices)
                        .expect("a filter slice fits one device word")
                        .div_ceil(thread_slices);
                    if output_groups * slice_groups > workgroup {
                        continue;
                    }
                    let staged_outputs = output_groups * thread_outputs;
                    for row_step in Self::ROW_STEPS {
                        let rows = row_step.min(shape.rows);
                        for column_step in Self::COLUMN_STEPS {
                            let columns = column_step.min(shape.columns);
                            let staged = u64::from(shape.channels)
                                * u64::from(window.span_rows(rows))
                                * u64::from(window.span_columns(columns))
                                + u64::from(staged_outputs) * u64::from(rows) * u64::from(columns);
                            if staged * WORD_BYTES > pool {
                                continue;
                            }
                            let candidate = Self::new(
                                window,
                                rows,
                                columns,
                                shape.channels,
                                outputs,
                                thread_outputs,
                                thread_slices,
                            );
                            let score = candidate.score(workgroup, wanted, shape);
                            if best.as_ref().is_none_or(|(kept, led)| {
                                score > *kept
                                    || (score == *kept
                                        && (candidate.rows, candidate.columns)
                                            > (led.rows, led.columns))
                            }) {
                                best = Some((score, candidate));
                            }
                        }
                    }
                }
            }
        }
        best.map(|(_, tile)| tile)
    }

    fn score(self, workgroup: u32, wanted: u32, shape: ConvolutionShape) -> f64 {
        let slices = f64::from(shape.channels) * f64::from(self.window.taps());
        let fma = f64::from(self.thread_outputs) * f64::from(self.thread_slices);
        let loads = f64::from(self.thread_outputs + self.thread_slices);
        let inner = (fma + loads) / fma;
        let macs =
            f64::from(self.outputs) * f64::from(self.rows) * f64::from(self.columns) * slices;
        let staged =
            f64::from(self.channels) * f64::from(self.span_rows()) * f64::from(self.span_columns())
                + f64::from(self.staged_outputs()) * f64::from(self.rows) * f64::from(self.columns);
        let throughput = macs / (inner * macs + staged);
        let chunks = self.chunks(shape, wanted);
        let tasks = self.blocks(shape).max(1) * chunks;
        let fill = f64::from(tasks).min(f64::from(wanted)) / f64::from(wanted.max(1));
        let busy = f64::from(self.lanes()).min(f64::from(workgroup)) / f64::from(workgroup);
        throughput * (fill * busy).sqrt()
    }

    pub fn blocks(self, shape: ConvolutionShape) -> u32 {
        shape.groups * shape.outputs.div_ceil(self.outputs)
    }

    pub fn chunks(self, shape: ConvolutionShape, wanted: u32) -> u32 {
        let blocks = self.blocks(shape).max(1);
        let rows_total = shape.batch * shape.rows;
        (wanted / blocks)
            .max(1)
            .min(rows_total.div_ceil(self.rows).max(1))
    }

    pub fn lanes(self) -> u32 {
        self.outputs.div_ceil(self.thread_outputs) * self.slice_groups()
    }

    pub const fn window(self) -> ConvolutionWindow {
        self.window
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.columns
    }

    pub const fn channels(self) -> u32 {
        self.channels
    }

    pub const fn outputs(self) -> u32 {
        self.outputs
    }

    pub const fn thread_outputs(self) -> u32 {
        self.thread_outputs
    }

    pub const fn thread_slices(self) -> u32 {
        self.thread_slices
    }

    pub const fn words(self) -> [u32; 12] {
        let window = self.window.words();
        [
            window[0],
            window[1],
            window[2],
            window[3],
            window[4],
            window[5],
            self.rows,
            self.columns,
            self.channels,
            self.outputs,
            self.thread_outputs,
            self.thread_slices,
        ]
    }

    pub fn of_words(words: [u32; 12]) -> Option<Self> {
        let window = ConvolutionWindow::of_words(words[..6].try_into().ok()?)?;
        let [
            rows,
            columns,
            channels,
            outputs,
            thread_outputs,
            thread_slices,
        ] = words[6..].try_into().ok()?;
        if rows == 0
            || columns == 0
            || channels == 0
            || outputs == 0
            || thread_outputs == 0
            || thread_outputs > outputs
            || !Self::THREAD_SLICES.contains(&thread_slices)
            || thread_outputs * thread_slices > Self::REGISTER_CEILING
        {
            return None;
        }
        Some(Self {
            window,
            rows,
            columns,
            channels,
            outputs,
            thread_outputs,
            thread_slices,
        })
    }

    pub const fn output_groups(self) -> u32 {
        self.outputs.div_ceil(self.thread_outputs)
    }

    pub const fn slices(self) -> u32 {
        self.channels * self.window.taps()
    }

    pub const fn slice_groups(self) -> u32 {
        self.slices().div_ceil(self.thread_slices)
    }

    pub const fn staged_outputs(self) -> u32 {
        self.output_groups() * self.thread_outputs
    }

    pub const fn span_rows(self) -> u32 {
        self.window.span_rows(self.rows)
    }

    pub const fn span_columns(self) -> u32 {
        self.window.span_columns(self.columns)
    }

    pub const fn scratch_words(self) -> u32 {
        self.channels * self.span_rows() * self.span_columns()
            + self.staged_outputs() * self.rows * self.columns
    }

    pub const fn items(self) -> u32 {
        self.output_groups() * self.slice_groups()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MatmulStrategy {
    Staged,
    Streamed,
    Cooperative,
    Folded,
}

pub const FOLDED_LANES: u32 = neura_abi::SUPER_SUB_BLOCK;
pub const STAGED_RUN: u32 = neura_abi::SUPER_SUB_BLOCK;
pub const BLOCK_RUN_WORDS: u32 = 9;
pub const RUN_CODES: u32 = neura_abi::QUANT_BLOCK / 4;
pub const RUN_SCALES: u32 = 2;
pub const RUN_OFFSET: u32 = 1;
pub const RUN_SLOT: u32 = RUN_CODES + RUN_SCALES + RUN_OFFSET;
pub const RUN_STAGE_BLOCKS: [u32; 4] = [1, 2, 4, 8];

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Run {
    Blocks,
    Codes,
}

pub const fn run_words(depth: u32, run: Run) -> u64 {
    let blocks = (depth / neura_abi::QUANT_BLOCK) as u64;
    match run {
        Run::Blocks => blocks * BLOCK_RUN_WORDS as u64,
        Run::Codes => blocks * RUN_SLOT as u64,
    }
}

const _: () = assert!(
    neura_abi::Element::Q8_0.stages_a_run() && neura_abi::Element::Q4K.packs_codes(),
    "a staged run stages the blocks of one element and the codes of another",
);
const _: () = assert!(
    RUN_CODES * neura_abi::WORD_BYTES as u32 == neura_abi::QUANT_BLOCK,
    "a run slot holds one code for every number of a block the element packs",
);
pub const FOLDED_STAGE: u32 = 1024;
pub const DELTA_WIDTH: u32 = neura_abi::DELTA_WIDTH;
pub const DELTA_COLUMNS: u32 = 32;
pub const DELTA_SPLITS: u32 = 8;
pub const DELTA_STATE_WORDS: u32 = DELTA_WIDTH * DELTA_COLUMNS;
pub const DELTA_KH: u32 = DELTA_STATE_WORDS;
pub const DELTA_QH: u32 = DELTA_KH + DELTA_COLUMNS;
pub const DELTA_DELTA: u32 = DELTA_QH + DELTA_COLUMNS;
pub const DELTA_Q: u32 = DELTA_DELTA + DELTA_COLUMNS;
pub const DELTA_K: u32 = DELTA_Q + DELTA_WIDTH;
pub const DELTA_V: u32 = DELTA_K + DELTA_WIDTH;
pub const DELTA_REDUCE: u32 = DELTA_V + DELTA_WIDTH;
pub const DELTA_PARTIALS: u32 = DELTA_REDUCE + 1;
pub const DELTA_SCRATCH_WORDS: u32 = DELTA_PARTIALS + 2 * DELTA_SPLITS * DELTA_COLUMNS;
const FOLDED_COLUMNS_FLOOR: u32 = 32;
pub const FOLDED_FOLDS: u32 = 8;
pub const FOLDED_RUN: u32 = match staged_block() {
    Some(block) => (FOLDED_LANES * block.bytes).div_ceil(neura_abi::WORD_BYTES as u32),
    None => panic!("a run stages the blocks of an element that packs none"),
};

const fn staged_block() -> Option<neura_abi::Block> {
    match neura_abi::Element::Q8_0.stages_a_run() {
        true => neura_abi::Element::Q8_0.block(),
        false => None,
    }
}

const _: () = assert!(
    FOLDED_RUN >= FOLDED_LANES,
    "a run stages a block of every lane of a warp",
);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct FoldedTile {
    rows: u32,
    warps: u32,
    folds: u32,
    depth: u32,
    lanes: u32,
    run: bool,
}

impl FoldedTile {
    pub fn new(rows: u32, warps: u32, folds: u32, depth: u32) -> Self {
        Self::within(rows, warps, folds, depth, false)
    }

    pub fn staged(rows: u32, warps: u32, folds: u32, depth: u32) -> Self {
        Self::within(rows, warps, folds, depth, true)
    }

    fn within(rows: u32, warps: u32, folds: u32, depth: u32, run: bool) -> Self {
        assert!(
            rows > 0 && warps > 0 && folds > 0 && depth > 0,
            "a folded tile spans no element",
        );
        assert_eq!(
            depth, FOLDED_LANES,
            "a folded tile walks {depth} numbers of the reduction per lane where its lanes walk {FOLDED_LANES}",
        );
        Self {
            rows,
            warps,
            folds,
            depth,
            lanes: FOLDED_LANES,
            run,
        }
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.warps * self.folds
    }

    pub const fn warps(self) -> u32 {
        self.warps
    }

    pub const fn folds(self) -> u32 {
        self.folds
    }

    pub const fn depth(self) -> u32 {
        self.depth
    }

    pub const fn lanes(self) -> u32 {
        self.lanes
    }

    pub const fn stages_a_run(self) -> bool {
        self.run
    }

    pub const fn run_words(self) -> u64 {
        match self.run {
            true => self.warps as u64 * FOLDED_RUN as u64,
            false => 0,
        }
    }
}

pub const fn folded_stage_words() -> u64 {
    FOLDED_STAGE as u64 + (FOLDED_STAGE / FOLDED_LANES) as u64
}

pub const fn folded_partials(columns: u32) -> u64 {
    FOLDED_LANES as u64 * (columns as u64 + 1)
}

pub const fn folded_splits(depth: u32) -> u32 {
    depth.div_ceil(FOLDED_STAGE)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct CooperativeMatrix {
    subgroup: u32,
    rows: u32,
    columns: u32,
    depth: u32,
}

impl CooperativeMatrix {
    pub fn new(subgroup: u32, rows: u32, columns: u32, depth: u32) -> Self {
        assert!(
            subgroup > 0 && rows > 0 && columns > 0 && depth > 0,
            "a cooperative matrix unit spans no element",
        );
        Self {
            subgroup,
            rows,
            columns,
            depth,
        }
    }

    pub const fn subgroup(self) -> u32 {
        self.subgroup
    }

    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.columns
    }

    pub const fn depth(self) -> u32 {
        self.depth
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct CooperativeTile {
    fragment: CooperativeMatrix,
    subgroups: (u32, u32),
    fragments: (u32, u32),
    stage: u32,
}

impl CooperativeTile {
    pub fn new(
        fragment: CooperativeMatrix,
        subgroups: (u32, u32),
        fragments: (u32, u32),
        stage: u32,
    ) -> Self {
        assert!(
            subgroups.0 > 0 && subgroups.1 > 0,
            "a cooperative tile hands its fragments to no subgroup",
        );
        assert!(
            fragments.0 > 0 && fragments.1 > 0,
            "a subgroup of a cooperative tile carries no fragment",
        );
        assert!(
            stage > 0 && stage.is_multiple_of(fragment.depth()),
            "a cooperative tile stages {stage} deep where a whole fragment of its device is {} deep",
            fragment.depth(),
        );
        Self {
            fragment,
            subgroups,
            fragments,
            stage,
        }
    }

    pub const fn fragment(self) -> CooperativeMatrix {
        self.fragment
    }

    pub const fn fragments(self) -> (u32, u32) {
        self.fragments
    }

    pub const fn stage(self) -> u32 {
        self.stage
    }

    pub const fn subgroup(self) -> u32 {
        self.fragment.subgroup()
    }

    pub const fn subgroup_rows(self) -> u32 {
        self.subgroups.0
    }

    pub const fn subgroup_columns(self) -> u32 {
        self.subgroups.1
    }

    pub const fn fragment_rows(self) -> u32 {
        self.fragments.0
    }

    pub const fn fragment_columns(self) -> u32 {
        self.fragments.1
    }

    pub const fn rows(self) -> u32 {
        self.subgroups.0 * self.fragments.0 * self.fragment.rows()
    }

    pub const fn columns(self) -> u32 {
        self.subgroups.1 * self.fragments.1 * self.fragment.columns()
    }

    pub const fn depth(self) -> u32 {
        self.stage
    }

    pub const fn threads(self) -> u32 {
        self.subgroups.0 * self.subgroups.1 * self.fragment.subgroup()
    }

    pub const fn accumulators(self) -> u32 {
        self.fragments.0 * self.fragments.1
    }

    pub const fn registers(self) -> u32 {
        let accumulator = self.accumulators() * self.fragment.rows() * self.fragment.columns();
        let operands = self.fragment.rows() * self.fragment.depth()
            + self.fragment.depth() * self.fragment.columns();
        (accumulator + operands) / self.fragment.subgroup()
    }

    pub const fn left_stride(self) -> u32 {
        self.stage
    }

    pub const fn left_stage(self) -> u64 {
        self.rows() as u64 * self.stage as u64
    }

    pub const fn right_stage(self) -> u64 {
        self.columns() as u64 * self.stage as u64
    }

    pub const fn half_panels(self) -> u64 {
        2 * (self.left_stage() + self.right_stage())
    }

    pub const fn half_bytes(self) -> u64 {
        self.half_panels() * 2
    }

    pub const fn copy(self) -> u64 {
        self.subgroups.0 as u64
            * self.subgroups.1 as u64
            * self.fragment.rows() as u64
            * self.fragment.columns() as u64
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct PlainTile {
    rows: u32,
    columns: u32,
    depth: u32,
    thread_rows: u32,
    thread_columns: u32,
    run: Option<Run>,
}

impl PlainTile {
    pub const fn rows(self) -> u32 {
        self.rows
    }

    pub const fn columns(self) -> u32 {
        self.columns
    }

    pub const fn depth(self) -> u32 {
        self.depth
    }

    pub const fn thread_rows(self) -> u32 {
        self.thread_rows
    }

    pub const fn thread_columns(self) -> u32 {
        self.thread_columns
    }

    pub const fn stages_a_run(self) -> bool {
        self.run.is_some()
    }

    pub const fn run(self) -> Option<Run> {
        self.run
    }

    pub const fn threads(self) -> u32 {
        self.thread_rows * self.thread_columns
    }

    pub const fn register_rows(self) -> u32 {
        self.rows / self.thread_rows
    }

    pub const fn register_columns(self) -> u32 {
        self.columns / self.thread_columns
    }

    pub const fn registers(self) -> u32 {
        self.register_rows() * self.register_columns()
    }

    pub const fn operands(self) -> u32 {
        self.register_rows() + self.register_columns()
    }

    pub const fn tile_work(self) -> u64 {
        self.rows as u64 * self.columns as u64 * self.depth as u64
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum MatmulTile {
    Staged(PlainTile),
    Streamed(PlainTile),
    Cooperative(CooperativeTile),
    Folded(FoldedTile),
}

impl MatmulTile {
    pub const fn new(
        strategy: MatmulStrategy,
        rows: u32,
        columns: u32,
        depth: u32,
        thread_rows: u32,
        thread_columns: u32,
    ) -> Self {
        Self::with(
            strategy,
            rows,
            columns,
            depth,
            thread_rows,
            thread_columns,
            None,
        )
    }

    pub const fn staged_run(
        run: Run,
        rows: u32,
        columns: u32,
        blocks: u32,
        thread_rows: u32,
        thread_columns: u32,
    ) -> Self {
        Self::with(
            MatmulStrategy::Staged,
            rows,
            columns,
            blocks * neura_abi::QUANT_BLOCK,
            thread_rows,
            thread_columns,
            Some(run),
        )
    }

    const fn with(
        strategy: MatmulStrategy,
        rows: u32,
        columns: u32,
        depth: u32,
        thread_rows: u32,
        thread_columns: u32,
        run: Option<Run>,
    ) -> Self {
        assert!(
            rows > 0 && columns > 0 && depth > 0,
            "a matmul tile spans no element"
        );
        assert!(
            thread_rows > 0 && thread_columns > 0,
            "a matmul tile is carried by no thread",
        );
        assert!(
            rows.is_multiple_of(thread_rows),
            "a matmul tile does not divide its rows by the threads along rows",
        );
        assert!(
            columns.is_multiple_of(thread_columns),
            "a matmul tile does not divide its columns by the threads along columns",
        );
        assert!(
            matches!(strategy, MatmulStrategy::Streamed) || depth.is_multiple_of(2),
            "a staged tile walks two of its rows at one stride into the same bank of the workgroup scratch",
        );
        assert!(
            run.is_none() || matches!(strategy, MatmulStrategy::Staged),
            "a run stages the blocks of an element beside the panels a staged tile walks",
        );
        assert!(
            run.is_none() || depth.is_multiple_of(neura_abi::QUANT_BLOCK),
            "a run stages a depth a block of its element divides",
        );
        let tile = PlainTile {
            rows,
            columns,
            depth,
            thread_rows,
            thread_columns,
            run,
        };
        match strategy {
            MatmulStrategy::Staged => Self::Staged(tile),
            MatmulStrategy::Streamed => Self::Streamed(tile),
            MatmulStrategy::Cooperative => panic!(
                "a cooperative tile divides the fragments of its device among subgroups instead of registers",
            ),
            MatmulStrategy::Folded => panic!(
                "a folded tile walks its reduction along the lanes of a warp instead of the panels of a workgroup",
            ),
        }
    }

    pub const fn cooperative(tile: CooperativeTile) -> Self {
        Self::Cooperative(tile)
    }

    pub const fn folded(tile: FoldedTile) -> Self {
        Self::Folded(tile)
    }

    pub const fn strategy(self) -> MatmulStrategy {
        match self {
            Self::Staged(_) => MatmulStrategy::Staged,
            Self::Streamed(_) => MatmulStrategy::Streamed,
            Self::Cooperative(_) => MatmulStrategy::Cooperative,
            Self::Folded(_) => MatmulStrategy::Folded,
        }
    }

    pub const fn rows(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.rows(),
            Self::Cooperative(tile) => tile.rows(),
            Self::Folded(tile) => tile.rows(),
        }
    }

    pub const fn columns(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.columns(),
            Self::Cooperative(tile) => tile.columns(),
            Self::Folded(tile) => tile.columns(),
        }
    }

    pub const fn depth(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.depth(),
            Self::Cooperative(tile) => tile.depth(),
            Self::Folded(tile) => tile.depth(),
        }
    }

    pub const fn thread_rows(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.thread_rows(),
            Self::Cooperative(tile) => tile.subgroup_rows(),
            Self::Folded(tile) => tile.lanes(),
        }
    }

    pub const fn thread_columns(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.thread_columns(),
            Self::Cooperative(tile) => tile.subgroup_columns(),
            Self::Folded(tile) => tile.warps(),
        }
    }

    pub const fn threads(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.threads(),
            Self::Cooperative(tile) => tile.threads(),
            Self::Folded(tile) => tile.warps() * tile.lanes(),
        }
    }

    pub const fn registers(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.registers(),
            Self::Cooperative(tile) => tile.registers(),
            Self::Folded(tile) => tile.rows(),
        }
    }

    pub const fn operands(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.operands(),
            Self::Cooperative(_) => 0,
            Self::Folded(tile) => tile.rows(),
        }
    }

    pub const fn register_rows(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.register_rows(),
            Self::Cooperative(_) => panic!("a cooperative tile accumulates across a subgroup"),
            Self::Folded(tile) => tile.rows(),
        }
    }

    pub const fn register_columns(self) -> u32 {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.register_columns(),
            Self::Cooperative(_) => panic!("a cooperative tile accumulates across a subgroup"),
            Self::Folded(_) => 1,
        }
    }

    pub const fn tile_work(self) -> u64 {
        self.rows() as u64 * self.columns() as u64 * self.depth() as u64
    }

    pub const fn stages_a_run(self) -> bool {
        match self {
            Self::Staged(tile) | Self::Streamed(tile) => tile.stages_a_run(),
            Self::Folded(tile) => tile.stages_a_run(),
            Self::Cooperative(_) => false,
        }
    }

    pub const fn run(self) -> Option<Run> {
        match self {
            Self::Staged(tile) => tile.run(),
            _ => None,
        }
    }

    pub const fn run_words(self) -> u64 {
        match (self.run(), self.depth()) {
            (Some(run), depth) => run_words(depth, run),
            _ => 0,
        }
    }

    pub const fn left_stride(self) -> u32 {
        match self {
            Self::Staged(tile) => tile.depth() + 1,
            Self::Streamed(_) => 0,
            Self::Cooperative(tile) => tile.left_stride(),
            Self::Folded(_) => 1,
        }
    }

    pub const fn left_stage(self) -> u64 {
        match self {
            Self::Staged(tile) => tile.rows() as u64 * (tile.depth() + 1) as u64,
            Self::Streamed(_) => 0,
            Self::Cooperative(tile) => tile.left_stage(),
            Self::Folded(_) => folded_stage_words(),
        }
    }

    pub const fn right_stage(self) -> u64 {
        match self {
            Self::Staged(tile) => match tile.run() {
                Some(run) => tile.columns() as u64 * run_words(tile.depth(), run),
                None => tile.columns() as u64 * (tile.depth() as u64 + 1),
            },
            Self::Streamed(_) => 0,
            Self::Cooperative(tile) => tile.right_stage(),
            Self::Folded(tile) => folded_partials(tile.columns()) + tile.run_words(),
        }
    }

    pub const fn half_panels(self) -> u64 {
        match self {
            Self::Cooperative(tile) => tile.half_panels(),
            _ => panic!("only a cooperative tile stages its panels on the half grid"),
        }
    }

    pub const fn shared_bytes(self) -> u64 {
        match self {
            Self::Cooperative(tile) => tile.half_bytes(),
            Self::Folded(tile) => {
                (folded_stage_words() + folded_partials(tile.columns())) * WORD_BYTES
            }
            _ => 2 * (self.left_stage() + self.right_stage()) * WORD_BYTES,
        }
    }

    pub const fn subgroup_rows(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.subgroup_rows(),
            _ => panic!("only a cooperative tile is carried by subgroups"),
        }
    }

    pub const fn subgroup_columns(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.subgroup_columns(),
            _ => panic!("only a cooperative tile is carried by subgroups"),
        }
    }

    pub const fn subgroup(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.subgroup(),
            _ => panic!("only a cooperative tile is carried by subgroups"),
        }
    }

    pub const fn fragment_rows(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.fragment().rows(),
            _ => panic!("only a cooperative tile holds the fragment of its device"),
        }
    }

    pub const fn fragment_columns(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.fragment().columns(),
            _ => panic!("only a cooperative tile holds the fragment of its device"),
        }
    }

    pub const fn fragment_depth(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.fragment().depth(),
            _ => panic!("only a cooperative tile holds the fragment of its device"),
        }
    }

    pub const fn fragment_grid(self) -> (u32, u32) {
        match self {
            Self::Cooperative(tile) => tile.fragments(),
            _ => panic!("only a cooperative tile holds the fragment of its device"),
        }
    }

    pub const fn accumulators(self) -> u32 {
        match self {
            Self::Cooperative(tile) => tile.accumulators(),
            _ => panic!("only a cooperative tile accumulates across a subgroup"),
        }
    }

    pub const fn copy(self) -> u64 {
        match self {
            Self::Cooperative(tile) => tile.copy(),
            _ => 0,
        }
    }

    pub const fn folds(self) -> u32 {
        match self {
            Self::Folded(tile) => tile.folds(),
            _ => panic!("only a folded tile hands every warp a run of columns"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Budget {
    threads: u32,
    shared_bytes: u64,
}

impl Budget {
    pub const BASELINE: Self = Self {
        threads: 256,
        shared_bytes: 16 << 10,
    };

    pub const BALANCED_THREADS: u32 = 256;

    pub const NOMINAL_RESIDENT_THREADS: u32 = 32_768;

    pub const fn of(threads: u32, shared_bytes: u64) -> Self {
        assert!(
            threads >= 32,
            "a device schedules workgroups of at least 32 threads",
        );
        assert!(
            shared_bytes >= 16 << 10,
            "a device offers its workgroups at least the baseline pool of shared memory",
        );
        Self {
            threads: 1 << (31 - threads.leading_zeros()),
            shared_bytes,
        }
    }

    pub const fn threads(self) -> u32 {
        self.threads
    }

    pub const fn shared_bytes(self) -> u64 {
        self.shared_bytes
    }
}

pub const MAX_TILES: usize = 64;
const PLAINEST_BLOCKING: (u32, u32) = (1, 1);
const WORKGROUP_SIZES: [u32; 5] = [64, 128, 256, 512, 1024];
const BLOCKINGS: [(u32, u32); 11] = [
    (8, 8),
    (8, 4),
    (4, 8),
    (4, 4),
    (2, 4),
    (4, 2),
    (2, 2),
    (1, 4),
    (4, 1),
    (1, 2),
    (2, 1),
];
const GRID_ASPECT: u32 = 4;
const STREAMED_ASPECT: u32 = 4;
const COOPERATIVE_ASPECT: u32 = 4;
const COOPERATIVE_FRAGMENTS: [(u32, u32); 4] = [(2, 2), (2, 1), (1, 2), (1, 1)];
const COOPERATIVE_STAGE_FRAGMENTS: u32 = 1;
const COOPERATIVE_MENU: usize = 6;
const POOL_BYTES: u64 = 21 << 10;
const RUN_POOL_BYTES: u64 = 32 << 10;
const BLOCKINGS_PER_GRID: usize = 2;
const DEPTH: u32 = 8;
pub const STREAMED_DEPTH: u32 = neura_abi::SUPER_SUB_BLOCK;
pub const CLAIM_WORDS: u32 = 8;
pub const CLAIM_BYTES: u64 = CLAIM_WORDS as u64 * WORD_BYTES;
const SCRATCH_WORDS_PER_THREAD: u64 = 2;
const RUN_ASPECT: u32 = 2;
const RUN_BLOCKINGS: [(u32, u32); 7] = [(1, 1), (1, 2), (1, 4), (2, 2), (2, 4), (4, 1), (8, 1)];
const REGISTER_FILE: u32 = 65_536;
const REGISTER_SLACK: u32 = 24;
const THREAD_REGISTER_CEILING: u32 = 255;
const MIN_REGISTER_BUDGET: u32 = 32;
const RESIDENT_LADDER: [u32; 3] = [1, 2, 4];

fn carries_registers(workgroup: u32, registers: u32) -> bool {
    registers <= loose_registers(workgroup)
}

const fn loose_registers(workgroup: u32) -> u32 {
    let pool = REGISTER_FILE / workgroup;
    let ceiling = if pool < THREAD_REGISTER_CEILING {
        pool
    } else {
        THREAD_REGISTER_CEILING
    };
    ceiling - REGISTER_SLACK
}

const fn resident_registers(workgroup: u32, resident: u32) -> u32 {
    (REGISTER_FILE / (workgroup * resident)).saturating_sub(REGISTER_SLACK)
}

const fn scratch_budget(workgroup: u32, staging: u64, copy: u64) -> u64 {
    scratch_of(workgroup, staging, 0, copy)
}

const fn scratch_of(workgroup: u32, staging: u64, folded: u64, copy: u64) -> u64 {
    let floor = SCRATCH_WORDS_PER_THREAD * workgroup as u64 * WORD_BYTES;
    let widest = if staging > floor { staging } else { floor };
    let widest = if folded > widest { folded } else { widest };
    if copy > widest {
        copy + CLAIM_BYTES
    } else {
        widest + CLAIM_BYTES
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Profile {
    workgroup: u32,
    registers: u32,
    shared_bytes: u64,
    tiles: [MatmulTile; MAX_TILES],
    count: u8,
}

impl std::fmt::Debug for Profile {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("Profile")
            .field("workgroup", &self.workgroup)
            .field("registers", &self.registers)
            .field("shared_bytes", &self.shared_bytes)
            .field("tiles", &self.tiles())
            .finish()
    }
}

impl Profile {
    pub fn of(tiles: &[MatmulTile]) -> Self {
        Self::within(tiles, loose_registers(tiles[0].threads()))
    }

    pub fn within(tiles: &[MatmulTile], registers: u32) -> Self {
        assert!(
            !tiles.is_empty() && tiles.len() <= MAX_TILES,
            "a profile offers between one and thirty-two matmul tiles",
        );
        let workgroup = tiles[0].threads();
        let mut entries = [tiles[0]; MAX_TILES];
        let mut fragment = None;
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        let mut partials = 0u64;
        let mut run = 0u64;
        let mut panels = 0u64;
        let mut copy = 0u64;
        let mut index = 0;
        while index < tiles.len() {
            let tile = tiles[index];
            assert!(
                tile.threads() == workgroup,
                "a profile hands one device program two workgroup sizes",
            );
            assert!(
                tile.registers() + tile.operands() <= registers,
                "a {workgroup} thread workgroup of {tile:?} spends {} registers a thread where its profile offers {registers}",
                tile.registers() + tile.operands(),
            );
            if tile.strategy() == MatmulStrategy::Streamed {
                assert_eq!(
                    tile.depth(),
                    STREAMED_DEPTH,
                    "a streamed tile walks a run of {STREAMED_DEPTH} elements of the reduction at a stride, and {tile:?} walks another run",
                );
            }
            if let MatmulTile::Cooperative(cooperative) = tile {
                assert!(
                    *fragment.get_or_insert(cooperative.fragment()) == cooperative.fragment(),
                    "a device program holds one cooperative fragment shape, and {tile:?} holds another",
                );
                let here = cooperative.half_bytes();
                if here > panels {
                    panels = here;
                }
                let here = cooperative.copy() * WORD_BYTES;
                if here > copy {
                    copy = here;
                }
            } else if let MatmulTile::Folded(held) = tile {
                partials = partials.max(folded_stage_words() + folded_partials(held.columns()));
                run = run.max(held.run_words());
            } else {
                if tile.left_stage() > left_stage {
                    left_stage = tile.left_stage();
                }
                if tile.right_stage() > right_stage {
                    right_stage = tile.right_stage();
                }
            }
            entries[index] = tile;
            index += 1;
        }
        let folded = (partials + run) * WORD_BYTES;
        let staging = 2 * (left_stage + right_stage) * WORD_BYTES;
        Self {
            workgroup,
            registers,
            shared_bytes: scratch_of(workgroup, staging, folded, copy) + panels,
            tiles: entries,
            count: tiles.len() as u8,
        }
    }

    pub fn derive(budget: Budget, cooperative: Option<CooperativeMatrix>) -> Vec<Self> {
        let mut profiles = Vec::new();
        let pool = budget.shared_bytes().min(POOL_BYTES);
        for workgroup in WORKGROUP_SIZES {
            if workgroup > budget.threads() {
                continue;
            }
            let run_pool = budget.shared_bytes().min(RUN_POOL_BYTES);
            let mut tiles = staged_tiles(workgroup, pool, run_pool);
            for (rows, columns) in streamed_grids(workgroup) {
                tiles.push(MatmulTile::new(
                    MatmulStrategy::Streamed,
                    rows,
                    columns,
                    STREAMED_DEPTH,
                    rows,
                    columns,
                ));
            }
            let mut left_stage = 0u64;
            let mut right_stage = 0u64;
            for tile in &tiles {
                left_stage = left_stage.max(tile.left_stage());
                right_stage = right_stage.max(tile.right_stage());
            }
            for tile in folded_tiles(workgroup, pool) {
                tiles.push(MatmulTile::folded(tile));
            }
            let mut partials = 0u64;
            let mut run = 0u64;
            for tile in &tiles {
                if let MatmulTile::Folded(held) = tile {
                    partials = partials.max(folded_stage_words() + folded_partials(held.columns()));
                    run = run.max(held.run_words());
                }
            }
            let folded = (partials + run) * WORD_BYTES;
            if let Some(fragment) = cooperative {
                tiles.extend(cooperative_tiles(
                    workgroup,
                    budget.shared_bytes(),
                    fragment,
                    left_stage,
                    right_stage,
                    folded,
                ));
            }
            if tiles.is_empty() {
                continue;
            }
            tiles.sort_by_key(|tile| (tile.rows() * tile.columns(), tile.depth()));
            tiles.dedup();
            assert!(
                tiles.len() <= MAX_TILES,
                "a device derives {} tiles for one workgroup of {workgroup} threads where a profile carries {MAX_TILES}",
                tiles.len(),
            );
            profiles.push(Self::of(&tiles));
        }
        profiles
    }

    pub fn ladder(budget: Budget, cooperative: Option<CooperativeMatrix>) -> Vec<Self> {
        let mut ladder: Vec<Self> = Vec::new();
        for profile in Self::derive(budget, cooperative) {
            let loose = profile.registers();
            for resident in RESIDENT_LADDER {
                let registers = resident_registers(profile.workgroup(), resident).min(loose);
                if registers < MIN_REGISTER_BUDGET {
                    continue;
                }
                let tiles = profile
                    .tiles()
                    .iter()
                    .copied()
                    .filter(|tile| tile.registers() + tile.operands() <= registers)
                    .collect::<Vec<_>>();
                if tiles.is_empty()
                    || ladder.last().is_some_and(|held| {
                        held.workgroup() == profile.workgroup() && held.registers() == registers
                    })
                {
                    continue;
                }
                ladder.push(Self::within(&tiles, registers));
            }
        }
        ladder
    }

    pub const fn workgroup(self) -> u32 {
        self.workgroup
    }

    pub const fn registers(self) -> u32 {
        self.registers
    }

    pub fn tiles(&self) -> &[MatmulTile] {
        &self.tiles[..self.count as usize]
    }

    pub const fn shared_bytes(self) -> u64 {
        self.shared_bytes
    }

    pub const fn scratch_bytes(self) -> u64 {
        self.shared_bytes - CLAIM_BYTES
    }

    pub const fn body_scratch_bytes(self) -> u64 {
        self.scratch_bytes() - self.cooperative_panels()
    }

    pub const fn workgroups(self) -> u32 {
        let count = Budget::NOMINAL_RESIDENT_THREADS / self.workgroup;
        if count == 0 { 1 } else { count }
    }

    pub const fn fits(self, threads: u32, shared_bytes: u64) -> bool {
        self.workgroup <= threads && self.shared_bytes <= shared_bytes
    }

    pub const fn staging_bytes(self) -> u64 {
        let mut left = 0u64;
        let mut right = 0u64;
        let mut index = 0;
        while index < self.count as usize {
            if matches!(
                self.tiles[index],
                MatmulTile::Staged(_) | MatmulTile::Streamed(_)
            ) {
                if self.tiles[index].left_stage() > left {
                    left = self.tiles[index].left_stage();
                }
                if self.tiles[index].right_stage() > right {
                    right = self.tiles[index].right_stage();
                }
            }
            index += 1;
        }
        2 * (left + right) * WORD_BYTES
    }

    pub const fn cooperative_panels(self) -> u64 {
        let mut panels = 0u64;
        let mut index = 0;
        while index < self.count as usize {
            let tile = self.tiles[index];
            if let MatmulTile::Cooperative(cooperative) = tile {
                let here = cooperative.half_bytes();
                if here > panels {
                    panels = here;
                }
            }
            index += 1;
        }
        panels
    }
}

fn folded_tiles(workgroup: u32, pool: u64) -> Vec<FoldedTile> {
    let warps = workgroup / FOLDED_LANES;
    if warps == 0 {
        return Vec::new();
    }
    let room = (pool.saturating_sub(CLAIM_BYTES) / WORD_BYTES).saturating_sub(folded_stage_words());
    let ceiling = u64::from(FOLDED_FOLDS).min(room / u64::from(warps * (FOLDED_LANES + 1)));
    let folds =
        u32::try_from(ceiling).expect("a folded tile folds fewer columns than a word holds");
    if folds == 0 {
        return Vec::new();
    }
    let mut tiles = Vec::new();
    let mut partials = folded_stage_words() + folded_partials(warps * folds);
    let mut staged = 1u64;
    while staged <= u64::from(folds) {
        let held =
            u32::try_from(staged).expect("a folded tile folds fewer columns than a word holds");
        let here = folded_stage_words() + folded_partials(warps * held);
        partials = partials.max(here);
        staged = match staged {
            1 => 2,
            _ => staged * 2,
        };
    }
    let run = partials + u64::from(warps * FOLDED_RUN);
    let stages_a_run = run * WORD_BYTES + CLAIM_BYTES <= pool;
    let mut held = 1u64;
    while held <= u64::from(folds) {
        let fold =
            u32::try_from(held).expect("a folded tile folds fewer columns than a word holds");
        if warps * fold >= FOLDED_COLUMNS_FLOOR {
            tiles.push(FoldedTile::new(1, warps, fold, FOLDED_LANES));
        }
        if stages_a_run {
            tiles.push(FoldedTile::staged(1, warps, fold, FOLDED_LANES));
        }
        held = match held {
            1 => 2,
            _ => held * 2,
        };
    }
    let last = FoldedTile::new(1, warps, folds, FOLDED_LANES);
    if !tiles.contains(&last) {
        tiles.push(last);
    }
    tiles
}

fn staged_tiles(workgroup: u32, budget: u64, run_budget: u64) -> Vec<MatmulTile> {
    let mut chosen = staged_family(workgroup, budget, DEPTH, &[]);
    chosen.extend(staged_family(workgroup, budget, STAGED_RUN, &chosen));
    let blocks = staged_run_family(workgroup, run_budget, &chosen, Run::Blocks, u64::MAX);
    let ceiling = blocks
        .iter()
        .map(|tile| tile.right_stage())
        .max()
        .unwrap_or(0);
    chosen.extend(blocks);
    chosen.extend(staged_run_family(
        workgroup,
        run_budget,
        &chosen,
        Run::Codes,
        ceiling,
    ));
    chosen
}

fn staged_run_family(
    workgroup: u32,
    budget: u64,
    held: &[MatmulTile],
    run: Run,
    ceiling: u64,
) -> Vec<MatmulTile> {
    let mut candidates = Vec::new();
    for (rows, columns) in run_grids(workgroup) {
        for (register_rows, register_columns) in RUN_BLOCKINGS {
            for blocks in RUN_STAGE_BLOCKS {
                let tile = MatmulTile::staged_run(
                    run,
                    rows * register_rows,
                    columns * register_columns,
                    blocks,
                    rows,
                    columns,
                );
                if !carries_registers(workgroup, tile.registers() + tile.operands()) {
                    continue;
                }
                if !carried(tile.left_stage(), tile.right_stage(), workgroup, budget) {
                    continue;
                }
                if tile.right_stage() > ceiling {
                    continue;
                }
                candidates.push(tile);
            }
        }
    }
    choose_family(candidates, held, workgroup, budget, true, Some(run))
}

fn run_grids(workgroup: u32) -> Vec<(u32, u32)> {
    let mut grids = Vec::new();
    let mut rows = 1;
    while rows <= RUN_ASPECT && rows <= workgroup {
        if workgroup.is_multiple_of(rows) {
            grids.push((rows, workgroup / rows));
        }
        rows *= 2;
    }
    grids
}

fn staged_family(workgroup: u32, budget: u64, depth: u32, held: &[MatmulTile]) -> Vec<MatmulTile> {
    let mut candidates = Vec::new();
    for (rows, columns) in grids(workgroup) {
        for (register_rows, register_columns) in BLOCKINGS.into_iter().chain([PLAINEST_BLOCKING]) {
            let tile = MatmulTile::new(
                MatmulStrategy::Staged,
                rows * register_rows,
                columns * register_columns,
                depth,
                rows,
                columns,
            );
            if !carries_registers(workgroup, tile.registers() + tile.operands()) {
                continue;
            }
            if !carried(tile.left_stage(), tile.right_stage(), workgroup, budget) {
                continue;
            }
            candidates.push(tile);
        }
    }
    choose_family(candidates, held, workgroup, budget, false, None)
}

fn choose_family(
    mut candidates: Vec<MatmulTile>,
    held: &[MatmulTile],
    workgroup: u32,
    budget: u64,
    deep: bool,
    run: Option<Run>,
) -> Vec<MatmulTile> {
    if deep {
        candidates.sort_by_key(|tile| {
            (
                std::cmp::Reverse(tile.depth()),
                std::cmp::Reverse(tile.rows() * tile.columns()),
                tile.left_stage() + tile.right_stage(),
            )
        });
    } else {
        candidates.sort_by_key(|tile| {
            (
                std::cmp::Reverse(tile.rows() * tile.columns()),
                tile.left_stage() + tile.right_stage(),
            )
        });
    }
    candidates.dedup();
    let mut chosen: Vec<MatmulTile> = Vec::new();
    let mut blocks = HashMap::<((u32, u32), Option<Run>), usize>::new();
    for tile in &candidates {
        let grid = ((tile.thread_rows(), tile.thread_columns()), run);
        let count = blocks.entry(grid).or_insert(0);
        if *count >= BLOCKINGS_PER_GRID {
            continue;
        }
        if !fits_the_stage(held, &chosen, *tile, workgroup, budget) {
            continue;
        }
        *count += 1;
        chosen.push(*tile);
    }
    for tile in &candidates {
        if tile.register_rows() * tile.register_columns() != 1 {
            continue;
        }
        if !fits_the_stage(held, &chosen, *tile, workgroup, budget) {
            continue;
        }
        chosen.push(*tile);
    }
    chosen
}

fn fits_the_stage(
    held: &[MatmulTile],
    chosen: &[MatmulTile],
    tile: MatmulTile,
    workgroup: u32,
    budget: u64,
) -> bool {
    let mut left = tile.left_stage();
    let mut right = tile.right_stage();
    for other in held.iter().chain(chosen) {
        left = left.max(other.left_stage());
        right = right.max(other.right_stage());
    }
    carried(left, right, workgroup, budget)
}

fn carried(left_stage: u64, right_stage: u64, workgroup: u32, shared_bytes: u64) -> bool {
    scratch_budget(workgroup, 2 * (left_stage + right_stage) * WORD_BYTES, 0) <= shared_bytes
}

fn cooperative_tiles(
    workgroup: u32,
    shared_bytes: u64,
    fragment: CooperativeMatrix,
    left_stage: u64,
    right_stage: u64,
    folded: u64,
) -> Vec<MatmulTile> {
    if !workgroup.is_multiple_of(fragment.subgroup()) {
        return Vec::new();
    }
    let mut tiles: Vec<CooperativeTile> = Vec::new();
    for (subgroup_rows, subgroup_columns) in cooperative_subgroups(workgroup / fragment.subgroup())
    {
        for (fragments_rows, fragments_columns) in COOPERATIVE_FRAGMENTS {
            let fragments = (fragments_rows, fragments_columns);
            let mut stage = fragment.depth();
            let mut staged = None;
            while stage <= COOPERATIVE_STAGE_FRAGMENTS * fragment.depth() {
                let tile = CooperativeTile::new(
                    fragment,
                    (subgroup_rows, subgroup_columns),
                    fragments,
                    stage,
                );
                if !carries_registers(workgroup, tile.registers())
                    || !cooperative_fits(
                        workgroup,
                        shared_bytes,
                        left_stage,
                        right_stage,
                        folded,
                        tile,
                    )
                {
                    break;
                }
                staged = Some(tile);
                stage += fragment.depth();
            }
            if let Some(tile) = staged {
                tiles.push(tile);
            }
        }
    }
    tiles.sort_by_key(|tile| std::cmp::Reverse((tile.rows() * tile.columns(), tile.stage())));
    tiles.dedup();
    tiles.truncate(COOPERATIVE_MENU);
    tiles.into_iter().map(MatmulTile::cooperative).collect()
}

fn cooperative_subgroups(subgroups: u32) -> Vec<(u32, u32)> {
    let mut grids = Vec::new();
    let mut rows = 1;
    while rows <= subgroups {
        if subgroups.is_multiple_of(rows) {
            let columns = subgroups / rows;
            if rows <= COOPERATIVE_ASPECT * columns {
                grids.push((rows, columns));
            }
        }
        rows *= 2;
    }
    grids
}

fn cooperative_fits(
    workgroup: u32,
    shared_bytes: u64,
    left_stage: u64,
    right_stage: u64,
    folded: u64,
    tile: CooperativeTile,
) -> bool {
    let staging = scratch_budget(
        workgroup,
        2 * (left_stage + right_stage) * WORD_BYTES,
        tile.copy() * WORD_BYTES,
    );
    let staging = if folded > staging { folded } else { staging };
    staging + tile.half_bytes() <= shared_bytes
}

fn grids(workgroup: u32) -> Vec<(u32, u32)> {
    let mut grids = Vec::new();
    let mut rows = 1;
    while rows <= workgroup {
        if workgroup.is_multiple_of(rows) {
            let columns = workgroup / rows;
            if rows <= GRID_ASPECT * columns && columns <= GRID_ASPECT * rows {
                grids.push((rows, columns));
            }
        }
        rows *= 2;
    }
    grids
}

fn streamed_grids(workgroup: u32) -> Vec<(u32, u32)> {
    let mut grids = Vec::new();
    let mut rows = 1;
    while rows <= STREAMED_ASPECT && rows < workgroup {
        let columns = workgroup / rows;
        assert!(
            columns * rows == workgroup,
            "a streamed tile hands its threads a grid that does not cover the workgroup",
        );
        grids.push((rows, columns));
        rows *= 2;
    }
    grids
}

#[derive(Clone, PartialEq, Eq, Debug, Hash)]
pub struct Geometry {
    workgroup: u32,
    budget: u64,
    body_scratch: u64,
    walked: Vec<(u32, MatmulTile)>,
    attention: Vec<AttentionTile>,
    convolution: Vec<ConvolutionTile>,
    convolution_weight: Vec<ConvolutionWeightTile>,
    convolution_input: Vec<ConvolutionInputTile>,
    left_stage: u64,
    right_stage: u64,
    folded: u64,
    partials: u64,
    half_stage: u64,
    copy: u64,
}

impl Profile {
    pub fn geometry(
        self,
        walked: &[(u32, MatmulTile)],
        attention: &[AttentionTile],
        convolution: &[ConvolutionTile],
        convolution_weight: &[ConvolutionWeightTile],
        convolution_input: &[ConvolutionInputTile],
    ) -> Geometry {
        Geometry::of(
            self,
            walked,
            attention,
            convolution,
            convolution_weight,
            convolution_input,
        )
    }
}

impl Geometry {
    fn of(
        profile: Profile,
        walked: &[(u32, MatmulTile)],
        attention: &[AttentionTile],
        convolution: &[ConvolutionTile],
        convolution_weight: &[ConvolutionWeightTile],
        convolution_input: &[ConvolutionInputTile],
    ) -> Self {
        let workgroup = profile.workgroup();
        assert!(
            walked.iter().all(|(_, tile)| tile.threads() == workgroup),
            "a device program carries a tile another workgroup stages",
        );
        assert!(
            walked.windows(2).all(|pair| pair[0].0 < pair[1].0),
            "a device program walks its tiles in the order its profile carries them",
        );
        let mut left_stage = 0u64;
        let mut right_stage = 0u64;
        let mut partials = 0u64;
        let mut run = 0u64;
        let mut half_stage = 0u64;
        let mut copy = 0u64;
        for (_, tile) in walked {
            if let MatmulTile::Cooperative(cooperative) = tile {
                half_stage = half_stage.max(cooperative.half_panels());
                copy = copy.max(cooperative.copy());
            } else if let MatmulTile::Folded(held) = tile {
                partials = partials.max(folded_stage_words() + folded_partials(held.columns()));
                run = run.max(held.run_words());
            } else {
                left_stage = left_stage.max(tile.left_stage());
                right_stage = right_stage.max(tile.right_stage());
            }
        }
        let folded = (partials + run) * WORD_BYTES;
        Self {
            workgroup,
            budget: profile.shared_bytes(),
            body_scratch: profile.body_scratch_bytes(),
            walked: walked.to_vec(),
            attention: attention.to_vec(),
            convolution: convolution.to_vec(),
            convolution_weight: convolution_weight.to_vec(),
            convolution_input: convolution_input.to_vec(),
            left_stage,
            right_stage,
            folded,
            partials,
            half_stage,
            copy,
        }
    }

    pub const fn workgroup(&self) -> u32 {
        self.workgroup
    }

    pub const fn budget(&self) -> u64 {
        self.budget
    }

    pub const fn staging_bytes(&self) -> u64 {
        2 * (self.left_stage + self.right_stage) * WORD_BYTES
    }

    pub fn scratch_bytes(&self, kinds: &[Kind]) -> u64 {
        u64::from(self.scratch_words(kinds)) * WORD_BYTES
    }

    pub const fn scratch_half_bytes(&self) -> u64 {
        self.half_stage * 2
    }

    pub fn workgroup_bytes(&self, kinds: &[Kind]) -> u64 {
        CLAIM_BYTES + self.scratch_bytes(kinds) + self.scratch_half_bytes()
    }

    fn scratch_words(&self, kinds: &[Kind]) -> u32 {
        DeviceModule::ALL
            .iter()
            .copied()
            .filter(|module| kinds.iter().any(|kind| kind.carries(*module)))
            .map(|module| self.module_scratch(module))
            .max()
            .unwrap_or(0)
    }

    fn module_scratch(&self, module: DeviceModule) -> u32 {
        match module {
            DeviceModule::MatmulTiles => (2 * (self.left_stage + self.right_stage))
                .max(self.folded / WORD_BYTES)
                .max(self.copy) as u32,
            DeviceModule::MatmulWeight => (self.left_stage + self.right_stage) as u32,
            DeviceModule::Attention => self.attention_scratch_words(),
            DeviceModule::Reduce => self.workgroup,
            DeviceModule::Choice => self.choice_layout().words(self.workgroup),
            DeviceModule::Conv => self
                .convolution
                .iter()
                .map(|tile| tile.scratch_words())
                .chain(
                    self.convolution_weight
                        .iter()
                        .map(|tile| tile.scratch_words()),
                )
                .chain(
                    self.convolution_input
                        .iter()
                        .map(|tile| tile.scratch_words()),
                )
                .max()
                .unwrap_or(0),
            DeviceModule::Scan => self.workgroup,
            DeviceModule::Delta => DELTA_SCRATCH_WORDS,
            _ => 0,
        }
    }

    pub fn cooperative(&self) -> Option<CooperativeTile> {
        self.walked.iter().find_map(|(_, tile)| match tile {
            MatmulTile::Cooperative(cooperative) => Some(*cooperative),
            _ => None,
        })
    }

    pub const fn half_panels(&self) -> u64 {
        self.half_stage
    }

    pub const fn matmul_right(&self) -> u32 {
        (2 * self.left_stage) as u32
    }

    pub const fn matmul_run(&self) -> u32 {
        self.partials as u32
    }

    pub const fn choice(&self) -> u32 {
        self.workgroup
    }

    pub fn choice_layout(&self) -> ChoiceLayout {
        ChoiceLayout::of(self.workgroup, self.body_scratch)
    }

    pub fn attention(&self) -> &[AttentionTile] {
        &self.attention
    }

    pub fn convolution(&self) -> &[ConvolutionTile] {
        &self.convolution
    }

    pub fn convolution_weight(&self) -> &[ConvolutionWeightTile] {
        &self.convolution_weight
    }

    pub fn convolution_input(&self) -> &[ConvolutionInputTile] {
        &self.convolution_input
    }

    pub fn attention_geometry(&self, tile: AttentionTile) -> u32 {
        self.attention
            .iter()
            .position(|candidate| *candidate == tile)
            .unwrap_or_else(|| panic!("a device program carries no {tile:?}"))
            .try_into()
            .expect("a device program carries fewer attention tiles than a word holds")
    }

    pub fn attention_tile(&self, geometry: u32) -> AttentionTile {
        self.attention
            .get(geometry as usize)
            .copied()
            .unwrap_or_else(|| {
                panic!(
                    "attention geometry {geometry} lies outside the {} a device program carries",
                    self.attention.len(),
                )
            })
    }

    pub fn attention_scratch_words(&self) -> u32 {
        self.attention
            .iter()
            .map(|tile| tile.scratch_words(self.workgroup))
            .max()
            .unwrap_or(0)
    }

    pub fn walked(&self) -> &[(u32, MatmulTile)] {
        &self.walked
    }
}
