use neura_compiler::{Compiler, ast};
use neura_profile::{ConvolutionTile, Geometry};

const COLUMNS: u32 = 4;
const SPAN: u32 = 4;
const WINDOW: u32 = COLUMNS + SPAN - 1;
const WIDE: u32 = 4;
const NARROW: u32 = 2;

pub(crate) fn install(compiler: &mut Compiler, geometry: &Geometry) {
    compiler.constant("CONV_COLUMNS", COLUMNS);
    compiler.constant("CONV_SPAN", SPAN);
    compiler.constant("CONV_WINDOW", WINDOW);
    compiler.constant("CONV_WIDE", WIDE);
    device::define(compiler);
    for channels in [NARROW, WIDE] {
        compiler.specialize(
            "template_conv2d_channels",
            &format!("conv2d_channels_{channels}"),
            &[("CONV_CHANNELS", channels)],
        );
    }
    for (index, tile) in geometry.convolution().iter().enumerate() {
        specialize(compiler, index as u32, *tile);
    }
}

fn specialize(compiler: &mut Compiler, index: u32, tile: ConvolutionTile) {
    let window = tile.window();
    let panel = tile.channels() * tile.span_rows() * tile.span_columns();
    let constants = [
        ("CONV_TILE_ROWS", tile.rows()),
        ("CONV_TILE_COLUMNS", tile.columns()),
        ("CONV_TILE_CHANNELS", tile.channels()),
        ("CONV_TILE_OUTPUTS", tile.outputs()),
        ("CONV_THREAD_COLUMNS", tile.thread_columns()),
        ("CONV_THREAD_OUTPUTS", tile.thread_outputs()),
        ("CONV_REACH_ROWS", window.reach_rows()),
        ("CONV_REACH_COLUMNS", window.reach_columns()),
        ("CONV_STRIDE_ROWS", window.stride_rows()),
        ("CONV_STRIDE_COLUMNS", window.stride_columns()),
        ("CONV_DILATION_ROWS", window.dilation_rows()),
        ("CONV_DILATION_COLUMNS", window.dilation_columns()),
        ("CONV_SPAN_ROWS", tile.span_rows()),
        ("CONV_SPAN_COLUMNS", tile.span_columns()),
        ("CONV_TAPS", window.taps()),
        (
            "CONV_TILE_WINDOW",
            tile.thread_columns() + window.reach_columns() - 1,
        ),
        (
            "CONV_COLUMN_GROUPS",
            tile.columns().div_ceil(tile.thread_columns()),
        ),
        (
            "CONV_OUTPUT_GROUPS",
            tile.outputs().div_ceil(tile.thread_outputs()),
        ),
        (
            "CONV_REGISTERS",
            tile.thread_columns() * tile.thread_outputs(),
        ),
        ("CONV_PANEL", 0),
        ("CONV_FILTER", panel),
    ];
    compiler.specialize(
        "template_conv2d_tile_panels",
        &format!("conv2d_tile_panels_{index}"),
        &constants,
    );
    let body = if window.slides_columns() {
        "template_run_conv2d_tile_slide"
    } else {
        "template_run_conv2d_tile_walk"
    };
    compiler.specialize(body, &format!("run_conv2d_tile_{index}"), &constants);
    compiler.insert_case(
        "run_conv2d_tile",
        ast::Arm {
            pattern: ast::Pattern::Integer(index),
            body: vec![ast::Statement::Expression(ast::Expression::call(
                format!("run_conv2d_tile_{index}"),
                vec![ast::Expression::name("task"), ast::Expression::name("lid")],
            ))],
        },
    );
}

#[neura_compiler::module]
mod device {
    fn run_conv2d(task: Task, lid: u32) {
        let input = values[task.a];
        let taps = values[task.b];
        let output = values[task.out];
        let channels = taps.dims.y;
        let out_channels = taps.dims.x / (input.dims.y / channels);
        let input_rows = i32(input.dims.z);
        let input_columns = i32(input.dims.w);
        let wide = channels % CONV_WIDE == 0u32;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            if wide {
                conv2d_channels_4(
                    task,
                    index,
                    input,
                    taps,
                    output,
                    channels,
                    out_channels,
                    input_rows,
                    input_columns,
                );
            } else {
                conv2d_channels_2(
                    task,
                    index,
                    input,
                    taps,
                    output,
                    channels,
                    out_channels,
                    input_rows,
                    input_columns,
                );
            }
        }
    }

    fn template_conv2d_channels(
        task: Task,
        index: u32,
        input: Value,
        taps: Value,
        output: Value,
        channels: u32,
        out_channels: u32,
        input_rows: i32,
        input_columns: i32,
    ) {
        let at = coordinates(index, output.dims);
        let first_channel = (at.y / out_channels) * channels;
        let mut totals = scalar_array(0.0, CONV_CHANNELS);
        let blocks = channels / CONV_CHANNELS;
        for block in stride(0u32, blocks, 1u32) {
            let channel = block * CONV_CHANNELS;
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = i32(at.z) * i32(task.stride_rows)
                    + i32(reach_row) * i32(task.dilation_rows)
                    - i32(task.pad_rows);
                if row < 0i32 || row >= input_rows {
                    continue;
                }
                let mut weights = scalar_array(0u32, CONV_CHANNELS);
                let mut sources = scalar_array(0u32, CONV_CHANNELS);
                for unit in unroll(0u32, CONV_CHANNELS, 1u32) {
                    weights[unit] =
                        read_address(uvec4(at.y, channel + unit, reach_row, 0u32), taps.strides);
                    sources[unit] = read_address(
                        uvec4(at.x, first_channel + channel + unit, u32(row), 0u32),
                        input.strides,
                    );
                }
                for reach_column in stride(0u32, task.reach_columns, 1u32) {
                    let column = i32(at.w) * i32(task.stride_columns)
                        + i32(reach_column) * i32(task.dilation_columns)
                        - i32(task.pad_columns);
                    if column >= 0i32 && column < input_columns {
                        let offset = u32(column) * input.strides.w;
                        for unit in unroll(0u32, CONV_CHANNELS, 1u32) {
                            totals[unit] = totals[unit]
                                + fetch(input, sources[unit] + offset) * fetch(taps, weights[unit]);
                        }
                    }
                    for unit in unroll(0u32, CONV_CHANNELS, 1u32) {
                        weights[unit] = weights[unit] + taps.strides.w;
                    }
                }
            }
        }
        let mut total = 0.0;
        for unit in unroll(0u32, CONV_CHANNELS, 1u32) {
            total = total + totals[unit];
        }
        for channel in stride(blocks * CONV_CHANNELS, channels, 1u32) {
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = i32(at.z) * i32(task.stride_rows)
                    + i32(reach_row) * i32(task.dilation_rows)
                    - i32(task.pad_rows);
                if row < 0i32 || row >= input_rows {
                    continue;
                }
                let mut weight = read_address(uvec4(at.y, channel, reach_row, 0u32), taps.strides);
                let source = read_address(
                    uvec4(at.x, first_channel + channel, u32(row), 0u32),
                    input.strides,
                );
                for reach_column in stride(0u32, task.reach_columns, 1u32) {
                    let column = i32(at.w) * i32(task.stride_columns)
                        + i32(reach_column) * i32(task.dilation_columns)
                        - i32(task.pad_columns);
                    if column >= 0i32 && column < input_columns {
                        total = total
                            + fetch(input, source + u32(column) * input.strides.w)
                                * fetch(taps, weight);
                    }
                    weight = weight + taps.strides.w;
                }
            }
        }
        publish(output, index, chained(task, at, total));
    }

    fn run_conv2d_tile(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::CONV2D_TILE, refusal::GEOMETRY, task.geometry),
        }
    }

    fn template_conv2d_tile_panels(
        task: Task,
        lid: u32,
        input: Value,
        taps: Value,
        block: u32,
        local: u32,
        group: u32,
        batch: u32,
        row0: u32,
        column0: u32,
        out0: u32,
    ) {
        let input_dims = input.dims;
        for unit in stride(
            lid,
            CONV_TILE_CHANNELS * CONV_SPAN_ROWS * CONV_SPAN_COLUMNS,
            WORKGROUP_SIZE,
        ) {
            let column = unit % CONV_SPAN_COLUMNS;
            let row = unit / CONV_SPAN_COLUMNS % CONV_SPAN_ROWS;
            let channel = unit / (CONV_SPAN_COLUMNS * CONV_SPAN_ROWS);
            let at_channel = block + channel;
            let input_row = i32(row0 * CONV_STRIDE_ROWS) - i32(task.pad_rows) + i32(row);
            let input_column =
                i32(column0 * CONV_STRIDE_COLUMNS) - i32(task.pad_columns) + i32(column);
            let inside = at_channel < local
                && input_row >= 0i32
                && input_row < i32(input_dims.z)
                && input_column >= 0i32
                && input_column < i32(input_dims.w);
            let address = select(
                0u32,
                read_address(
                    uvec4(
                        batch,
                        group * local + at_channel,
                        u32(input_row),
                        u32(input_column),
                    ),
                    input.strides,
                ),
                inside,
            );
            scratch[CONV_PANEL + unit] = select(0.0, fetch(input, address), inside);
        }
        for unit in stride(
            lid,
            CONV_TILE_OUTPUTS * CONV_TILE_CHANNELS * CONV_TAPS,
            WORKGROUP_SIZE,
        ) {
            let tap = unit % CONV_TAPS;
            let channel = unit / CONV_TAPS % CONV_TILE_CHANNELS;
            let output = unit / (CONV_TAPS * CONV_TILE_CHANNELS);
            let at_channel = block + channel;
            let at_output = out0 + output;
            let inside = at_channel < local && at_output < taps.dims.x;
            let address = select(
                0u32,
                read_address(
                    uvec4(
                        at_output,
                        at_channel,
                        tap / CONV_REACH_COLUMNS,
                        tap % CONV_REACH_COLUMNS,
                    ),
                    taps.strides,
                ),
                inside,
            );
            scratch[CONV_FILTER + unit] = select(0.0, fetch(taps, address), inside);
        }
    }

    fn template_run_conv2d_tile_slide(task: Task, lid: u32) {
        let input = values[task.a];
        let taps = values[task.b];
        let output = values[task.out];
        let dims = output.dims;
        let filter = taps.dims;
        let input_dims = input.dims;
        let local = filter.y;
        let groups = input_dims.y / local;
        let out_per_group = filter.x / groups;
        let out_blocks = ceil_div(out_per_group, CONV_TILE_OUTPUTS);
        let row_blocks = ceil_div(dims.z, CONV_TILE_ROWS);
        let column_blocks = ceil_div(dims.w, CONV_TILE_COLUMNS);
        let per_batch = groups * out_blocks * row_blocks * column_blocks;
        let items = CONV_TILE_ROWS * CONV_COLUMN_GROUPS * CONV_OUTPUT_GROUPS;
        let mut acc = scalar_array(0.0, CONV_REGISTERS);
        for tile in stride(task.first, task.first + task.count, 1u32) {
            let batch = tile / per_batch;
            if batch >= dims.x {
                continue;
            }
            let within = tile % per_batch;
            let column_block = within % column_blocks;
            let row_block = within / column_blocks % row_blocks;
            let out_block = within / (column_blocks * row_blocks) % out_blocks;
            let group = within / (column_blocks * row_blocks * out_blocks);
            let row0 = row_block * CONV_TILE_ROWS;
            let column0 = column_block * CONV_TILE_COLUMNS;
            let out0 = group * out_per_group + out_block * CONV_TILE_OUTPUTS;
            for register in unroll(0u32, CONV_REGISTERS, 1u32) {
                acc[register] = 0.0;
            }
            for block in stride(0u32, local, CONV_TILE_CHANNELS) {
                template_conv2d_tile_panels(
                    task, lid, input, taps, block, local, group, batch, row0, column0, out0,
                );
                workgroup_barrier();
                for item in stride(lid, items, WORKGROUP_SIZE) {
                    let row = item / (CONV_OUTPUT_GROUPS * CONV_COLUMN_GROUPS);
                    let rem = item % (CONV_OUTPUT_GROUPS * CONV_COLUMN_GROUPS);
                    let output_group = rem / CONV_COLUMN_GROUPS;
                    let column_group = rem % CONV_COLUMN_GROUPS;
                    let output_at = output_group * CONV_THREAD_OUTPUTS;
                    let column_at = column_group * CONV_THREAD_COLUMNS;
                    for channel in unroll(0u32, CONV_TILE_CHANNELS, 1u32) {
                        if block + channel < local {
                            for reach_row in unroll(0u32, CONV_REACH_ROWS, 1u32) {
                                let panel_row =
                                    row * CONV_STRIDE_ROWS + reach_row * CONV_DILATION_ROWS;
                                let mut window = scalar_array(0.0, CONV_TILE_WINDOW);
                                for slot in unroll(0u32, CONV_TILE_WINDOW, 1u32) {
                                    window[slot] = scratch[CONV_PANEL
                                        + (channel * CONV_SPAN_ROWS + panel_row)
                                            * CONV_SPAN_COLUMNS
                                        + column_at
                                        + slot];
                                }
                                for reach_column in unroll(0u32, CONV_REACH_COLUMNS, 1u32) {
                                    for output_unit in unroll(0u32, CONV_THREAD_OUTPUTS, 1u32) {
                                        if output_at + output_unit < CONV_TILE_OUTPUTS {
                                            let weight = scratch[CONV_FILTER
                                                + ((output_at + output_unit) * CONV_TILE_CHANNELS
                                                    + channel)
                                                    * CONV_TAPS
                                                + reach_row * CONV_REACH_COLUMNS
                                                + reach_column];
                                            for column_unit in
                                                unroll(0u32, CONV_THREAD_COLUMNS, 1u32)
                                            {
                                                if column_at + column_unit < CONV_TILE_COLUMNS {
                                                    let register = output_unit
                                                        * CONV_THREAD_COLUMNS
                                                        + column_unit;
                                                    acc[register] = acc[register]
                                                        + window[column_unit + reach_column]
                                                            * weight;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                workgroup_barrier();
            }
            for item in stride(lid, items, WORKGROUP_SIZE) {
                let row = item / (CONV_OUTPUT_GROUPS * CONV_COLUMN_GROUPS);
                let rem = item % (CONV_OUTPUT_GROUPS * CONV_COLUMN_GROUPS);
                let output_group = rem / CONV_COLUMN_GROUPS;
                let column_group = rem % CONV_COLUMN_GROUPS;
                let output_at = output_group * CONV_THREAD_OUTPUTS;
                let column_at = column_group * CONV_THREAD_COLUMNS;
                let row_at = row0 + row;
                if row_at < dims.z {
                    for output_unit in unroll(0u32, CONV_THREAD_OUTPUTS, 1u32) {
                        if output_at + output_unit < CONV_TILE_OUTPUTS {
                            let output_index = out0 + output_at + output_unit;
                            if output_index < dims.y {
                                for column_unit in unroll(0u32, CONV_THREAD_COLUMNS, 1u32) {
                                    if column_at + column_unit < CONV_TILE_COLUMNS {
                                        let column_index = column0 + column_at + column_unit;
                                        if column_index < dims.w {
                                            let register =
                                                output_unit * CONV_THREAD_COLUMNS + column_unit;
                                            let index = ((batch * dims.y + output_index) * dims.z
                                                + row_at)
                                                * dims.w
                                                + column_index;
                                            publish(
                                                output,
                                                index,
                                                chained(
                                                    task,
                                                    uvec4(
                                                        batch,
                                                        output_index,
                                                        row_at,
                                                        column_index,
                                                    ),
                                                    acc[register],
                                                ),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn template_run_conv2d_tile_walk(task: Task, lid: u32) {
        let input = values[task.a];
        let taps = values[task.b];
        let output = values[task.out];
        let dims = output.dims;
        let filter = taps.dims;
        let input_dims = input.dims;
        let local = filter.y;
        let groups = input_dims.y / local;
        let out_per_group = filter.x / groups;
        let out_blocks = ceil_div(out_per_group, CONV_TILE_OUTPUTS);
        let row_blocks = ceil_div(dims.z, CONV_TILE_ROWS);
        let column_blocks = ceil_div(dims.w, CONV_TILE_COLUMNS);
        let per_batch = groups * out_blocks * row_blocks * column_blocks;
        let items = CONV_TILE_ROWS * CONV_COLUMN_GROUPS * CONV_OUTPUT_GROUPS;
        let mut acc = scalar_array(0.0, CONV_REGISTERS);
        for tile in stride(task.first, task.first + task.count, 1u32) {
            let batch = tile / per_batch;
            if batch >= dims.x {
                continue;
            }
            let within = tile % per_batch;
            let column_block = within % column_blocks;
            let row_block = within / column_blocks % row_blocks;
            let out_block = within / (column_blocks * row_blocks) % out_blocks;
            let group = within / (column_blocks * row_blocks * out_blocks);
            let row0 = row_block * CONV_TILE_ROWS;
            let column0 = column_block * CONV_TILE_COLUMNS;
            let out0 = group * out_per_group + out_block * CONV_TILE_OUTPUTS;
            for register in unroll(0u32, CONV_REGISTERS, 1u32) {
                acc[register] = 0.0;
            }
            for block in stride(0u32, local, CONV_TILE_CHANNELS) {
                template_conv2d_tile_panels(
                    task, lid, input, taps, block, local, group, batch, row0, column0, out0,
                );
                workgroup_barrier();
                for item in stride(lid, items, WORKGROUP_SIZE) {
                    let row = item / (CONV_OUTPUT_GROUPS * CONV_COLUMN_GROUPS);
                    let rem = item % (CONV_OUTPUT_GROUPS * CONV_COLUMN_GROUPS);
                    let output_group = rem / CONV_COLUMN_GROUPS;
                    let column_group = rem % CONV_COLUMN_GROUPS;
                    let output_at = output_group * CONV_THREAD_OUTPUTS;
                    let column_at = column_group * CONV_THREAD_COLUMNS;
                    for channel in unroll(0u32, CONV_TILE_CHANNELS, 1u32) {
                        if block + channel < local {
                            for reach_row in unroll(0u32, CONV_REACH_ROWS, 1u32) {
                                let panel_row =
                                    row * CONV_STRIDE_ROWS + reach_row * CONV_DILATION_ROWS;
                                for reach_column in unroll(0u32, CONV_REACH_COLUMNS, 1u32) {
                                    for output_unit in unroll(0u32, CONV_THREAD_OUTPUTS, 1u32) {
                                        if output_at + output_unit < CONV_TILE_OUTPUTS {
                                            let weight = scratch[CONV_FILTER
                                                + ((output_at + output_unit) * CONV_TILE_CHANNELS
                                                    + channel)
                                                    * CONV_TAPS
                                                + reach_row * CONV_REACH_COLUMNS
                                                + reach_column];
                                            for column_unit in
                                                unroll(0u32, CONV_THREAD_COLUMNS, 1u32)
                                            {
                                                if column_at + column_unit < CONV_TILE_COLUMNS {
                                                    let panel_column = column_at
                                                        * CONV_STRIDE_COLUMNS
                                                        + reach_column * CONV_DILATION_COLUMNS
                                                        + column_unit * CONV_STRIDE_COLUMNS;
                                                    let held = scratch[CONV_PANEL
                                                        + (channel * CONV_SPAN_ROWS + panel_row)
                                                            * CONV_SPAN_COLUMNS
                                                        + panel_column];
                                                    let register = output_unit
                                                        * CONV_THREAD_COLUMNS
                                                        + column_unit;
                                                    acc[register] = acc[register] + held * weight;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                workgroup_barrier();
            }
            for item in stride(lid, items, WORKGROUP_SIZE) {
                let row = item / (CONV_OUTPUT_GROUPS * CONV_COLUMN_GROUPS);
                let rem = item % (CONV_OUTPUT_GROUPS * CONV_COLUMN_GROUPS);
                let output_group = rem / CONV_COLUMN_GROUPS;
                let column_group = rem % CONV_COLUMN_GROUPS;
                let output_at = output_group * CONV_THREAD_OUTPUTS;
                let column_at = column_group * CONV_THREAD_COLUMNS;
                let row_at = row0 + row;
                if row_at < dims.z {
                    for output_unit in unroll(0u32, CONV_THREAD_OUTPUTS, 1u32) {
                        if output_at + output_unit < CONV_TILE_OUTPUTS {
                            let output_index = out0 + output_at + output_unit;
                            if output_index < dims.y {
                                for column_unit in unroll(0u32, CONV_THREAD_COLUMNS, 1u32) {
                                    if column_at + column_unit < CONV_TILE_COLUMNS {
                                        let column_index = column0 + column_at + column_unit;
                                        if column_index < dims.w {
                                            let register =
                                                output_unit * CONV_THREAD_COLUMNS + column_unit;
                                            let index = ((batch * dims.y + output_index) * dims.z
                                                + row_at)
                                                * dims.w
                                                + column_index;
                                            publish(
                                                output,
                                                index,
                                                chained(
                                                    task,
                                                    uvec4(
                                                        batch,
                                                        output_index,
                                                        row_at,
                                                        column_index,
                                                    ),
                                                    acc[register],
                                                ),
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn run_conv2d_input_grad(task: Task, lid: u32) {
        match task.geometry {
            strategy::INPUT_FOLD => run_conv2d_input_fold(task, lid),
            strategy::INPUT_CHUNK => run_conv2d_input_chunk(task, lid),
            _ => refuse(kind::CONV2D_INPUT_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }

    fn window_at(shifted: i32, stride: u32, bound: i32) -> i32 {
        if shifted < 0i32 {
            return -1i32;
        }
        if stride == 1u32 {
            if shifted >= bound {
                return -1i32;
            }
            return shifted;
        }
        if shifted % i32(stride) != 0i32 {
            return -1i32;
        }
        let mapped = shifted / i32(stride);
        if mapped >= bound {
            return -1i32;
        }
        return mapped;
    }

    fn run_conv2d_transpose(task: Task, lid: u32) {
        match task.geometry {
            strategy::INPUT_FOLD => run_conv2d_input_fold(task, lid),
            strategy::INPUT_CHUNK => run_conv2d_input_chunk(task, lid),
            _ => refuse(kind::CONV2D_TRANSPOSE, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_conv2d_input_chunk(task: Task, lid: u32) {
        let filter = values[task.a];
        let gradient = values[task.b];
        let source = values[task.out];
        let channels = filter.dims.y;
        let out_channels = filter.dims.x / (source.dims.y / channels);
        let gradient_rows = i32(gradient.dims.z);
        let gradient_columns = i32(gradient.dims.w);
        let per_chunk = ceil_div(out_channels, task.splits);
        let first_block = task.slot * per_chunk;
        let last_block = min(first_block + per_chunk, out_channels);
        if task.stride_rows == 1u32
            && task.stride_columns == 1u32
            && task.dilation_columns == 1u32
            && task.reach_columns >= 2u32
            && task.reach_columns <= CONV_SPAN
        {
            for group in stride(
                task.first + lid * CONV_COLUMNS,
                task.first + task.count,
                WORKGROUP_SIZE * CONV_COLUMNS,
            ) {
                let at = coordinates(group, source.dims);
                if at.w + CONV_COLUMNS <= source.dims.w
                    && group + CONV_COLUMNS <= task.first + task.count
                {
                    conv2d_input_columns(
                        task,
                        at,
                        group,
                        filter,
                        gradient,
                        source,
                        channels,
                        out_channels,
                        first_block,
                        last_block,
                        gradient_rows,
                        gradient_columns,
                    );
                } else {
                    for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
                        let index = group + offset;
                        if index < task.first + task.count {
                            conv2d_input_element(
                                task,
                                index,
                                filter,
                                gradient,
                                source,
                                channels,
                                out_channels,
                                first_block,
                                last_block,
                                gradient_rows,
                                gradient_columns,
                            );
                        }
                    }
                }
            }
            return;
        }
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            conv2d_input_element(
                task,
                index,
                filter,
                gradient,
                source,
                channels,
                out_channels,
                first_block,
                last_block,
                gradient_rows,
                gradient_columns,
            );
        }
    }

    fn conv2d_input_columns(
        task: Task,
        at: uvec4,
        group: u32,
        filter: Value,
        gradient: Value,
        source: Value,
        channels: u32,
        out_channels: u32,
        first_block: u32,
        last_block: u32,
        gradient_rows: i32,
        gradient_columns: i32,
    ) {
        let first_channel = (at.y / channels) * out_channels;
        let local_channel = at.y % channels;
        let plane = at.x * gradient.strides.x;
        let left = i32(at.w) + i32(task.pad_columns) - i32(CONV_SPAN) + 1i32;
        let mut totals = scalar_array(0.0, CONV_COLUMNS);
        for channel in stride(
            first_channel + first_block,
            first_channel + last_block,
            1u32,
        ) {
            let planes = plane + channel * gradient.strides.y;
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = window_at(
                    i32(at.z) + i32(task.pad_rows) - i32(reach_row) * i32(task.dilation_rows),
                    task.stride_rows,
                    gradient_rows,
                );
                if row < 0i32 {
                    continue;
                }
                let mut window = scalar_array(0.0, CONV_WINDOW);
                for slot in unroll(0u32, CONV_WINDOW, 1u32) {
                    let column = left + i32(slot);
                    let inside = column >= 0i32 && column < gradient_columns;
                    let address = select(
                        0u32,
                        planes + u32(row) * gradient.strides.z + u32(column) * gradient.strides.w,
                        inside,
                    );
                    window[slot] = select(0.0, fetch(gradient, address), inside);
                }
                let mut weight = read_address(
                    uvec4(channel, local_channel, reach_row, 0u32),
                    filter.strides,
                );
                for reach_column in unroll(0u32, CONV_SPAN, 1u32) {
                    if reach_column < task.reach_columns {
                        let weighed = fetch(filter, weight);
                        for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
                            totals[offset] = totals[offset]
                                + window[offset + CONV_SPAN - 1u32 - reach_column] * weighed;
                        }
                        weight = weight + filter.strides.w;
                    }
                }
            }
        }
        for offset in unroll(0u32, CONV_COLUMNS, 1u32) {
            conv2d_input_publish(
                task,
                group + offset,
                uvec4(at.x, at.y, at.z, at.w + offset),
                totals[offset],
                source,
            );
        }
    }

    fn conv2d_input_publish(task: Task, index: u32, at: uvec4, total: f32, source: Value) {
        if task.extra == NO_VALUE {
            publish(source, index, chained(task, at, total));
        } else {
            let partials = values[task.extra];
            publish(
                partials,
                read_address(uvec4(0u32, 0u32, task.slot, index), partials.strides),
                total,
            );
        }
    }

    fn conv2d_input_element(
        task: Task,
        index: u32,
        filter: Value,
        gradient: Value,
        source: Value,
        channels: u32,
        out_channels: u32,
        first_block: u32,
        last_block: u32,
        gradient_rows: i32,
        gradient_columns: i32,
    ) {
        let at = coordinates(index, source.dims);
        let first_channel = (at.y / channels) * out_channels;
        let local_channel = at.y % channels;
        let mut total = 0.0;
        for channel in stride(
            first_channel + first_block,
            first_channel + last_block,
            1u32,
        ) {
            for reach_row in stride(0u32, task.reach_rows, 1u32) {
                let row = window_at(
                    i32(at.z) + i32(task.pad_rows) - i32(reach_row) * i32(task.dilation_rows),
                    task.stride_rows,
                    gradient_rows,
                );
                if row < 0i32 {
                    continue;
                }
                let mut weight = read_address(
                    uvec4(channel, local_channel, reach_row, 0u32),
                    filter.strides,
                );
                let mut source_at =
                    read_address(uvec4(at.x, channel, u32(row), 0u32), gradient.strides);
                for reach_column in stride(0u32, task.reach_columns, 1u32) {
                    let column = window_at(
                        i32(at.w) + i32(task.pad_columns)
                            - i32(reach_column) * i32(task.dilation_columns),
                        task.stride_columns,
                        gradient_columns,
                    );
                    if column >= 0i32 {
                        total = total
                            + fetch(gradient, source_at + u32(column) * gradient.strides.w)
                                * fetch(filter, weight);
                    }
                    weight = weight + filter.strides.w;
                }
            }
        }
        conv2d_input_publish(task, index, at, total, source);
    }

    fn run_conv2d_input_fold(task: Task, lid: u32) {
        let partials = values[task.a];
        let output = values[task.out];
        let chunks = partials.dims.z;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let mut total = 0.0;
            for chunk in stride(0u32, chunks, 1u32) {
                total = total
                    + fetch(
                        partials,
                        read_address(uvec4(0u32, 0u32, chunk, index), partials.strides),
                    );
            }
            publish(
                output,
                index,
                chained(task, coordinates(index, output.dims), total),
            );
        }
    }

    fn run_conv2d_weight_chunk(task: Task, lid: u32) {
        let input = values[task.a];
        let gradient = values[task.b];
        let taps = values[task.c];
        let output = values[task.out];
        let chunks = output.dims.z;
        let channels = taps.dims.y;
        let out_channels = taps.dims.x / (input.dims.y / channels);
        let input_rows = i32(input.dims.z);
        let input_columns = i32(input.dims.w);
        let gradient_rows = gradient.dims.z;
        let gradient_columns = gradient.dims.w;
        let reach_rows = taps.dims.z;
        let reach_columns = taps.dims.w;
        let positions = input.dims.x * gradient_rows * gradient_columns;
        let plane = gradient_rows * gradient_columns;
        let limit = (positions + chunks - 1u32) / chunks;
        let first = task.slot * limit;
        let last = min(first + limit, positions);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let reach_row = index / reach_columns % reach_rows;
            let reach_column = index % reach_columns;
            let channel = index / (reach_columns * reach_rows) % channels;
            let out_channel = index / (reach_columns * reach_rows * channels);
            let first_channel = (out_channel / out_channels) * channels;
            let mut total = 0.0;
            let mut batch = first / plane;
            let mut within = first % plane;
            let mut row = within / gradient_columns;
            let mut column = within % gradient_columns;
            for position in stride(first, last, 1u32) {
                let used_row = i32(row) * i32(task.stride_rows)
                    + i32(reach_row) * i32(task.dilation_rows)
                    - i32(task.pad_rows);
                let used_column = i32(column) * i32(task.stride_columns)
                    + i32(reach_column) * i32(task.dilation_columns)
                    - i32(task.pad_columns);
                if used_row >= 0i32
                    && used_row < input_rows
                    && used_column >= 0i32
                    && used_column < input_columns
                {
                    let source =
                        read_address(uvec4(batch, out_channel, row, column), gradient.strides);
                    let weight = read_address(
                        uvec4(
                            batch,
                            first_channel + channel,
                            u32(used_row),
                            u32(used_column),
                        ),
                        input.strides,
                    );
                    total = total + fetch(gradient, source) * fetch(input, weight);
                }
                column = column + 1u32;
                if column >= gradient_columns {
                    column = 0u32;
                    row = row + 1u32;
                    if row >= gradient_rows {
                        row = 0u32;
                        batch = batch + 1u32;
                    }
                }
            }
            publish(
                output,
                read_address(uvec4(0u32, 0u32, task.slot, index), output.strides),
                total,
            );
        }
    }

    fn run_conv2d_weight_fold(task: Task, lid: u32) {
        let partials = values[task.a];
        let output = values[task.out];
        let chunks = partials.dims.z;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let mut total = 0.0;
            for chunk in stride(0u32, chunks, 1u32) {
                total = total
                    + fetch(
                        partials,
                        read_address(uvec4(0u32, 0u32, chunk, index), partials.strides),
                    );
            }
            publish(
                output,
                index,
                chained(task, coordinates(index, output.dims), total),
            );
        }
    }

    fn run_conv2d_weight_grad(task: Task, lid: u32) {
        match task.geometry {
            strategy::WEIGHT_CHUNK => run_conv2d_weight_chunk(task, lid),
            strategy::WEIGHT_FOLD => run_conv2d_weight_fold(task, lid),
            _ => refuse(kind::CONV2D_WEIGHT_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }
}
