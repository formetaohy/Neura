use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    compiler.constant("NORM_RMS", neura_abi::NORM_RMS);
    compiler.constant("NORM_L2", neura_abi::NORM_L2);
    compiler.constant("NORM_LAYER", neura_abi::NORM_LAYER);
    compiler.constant("NORM_GROUP", neura_abi::NORM_GROUP);
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn norm_spans(task: Task, source: Value) -> uvec4 {
        if task.op == NORM_GROUP {
            let groups = max(task.slot, 1u32);
            let per = source.dims.y / groups;
            let span = source.dims.z * source.dims.w;
            return uvec4(per * span, groups, per, span);
        }
        return uvec4(source.dims.w, 1u32, 0u32, 0u32);
    }

    fn norm_channel(spans: uvec4, row: u32, column: u32) -> u32 {
        if spans.z == 0u32 {
            return column;
        }
        return (row % spans.y) * spans.z + column / spans.w;
    }

    fn norm_at(spans: uvec4, source: Value, row: u32, column: u32) -> uvec4 {
        if spans.z != 0u32 {
            return coordinates(row * spans.x + column, source.dims);
        }
        return coordinates(row * spans.x, source.dims) + uvec4(0u32, 0u32, 0u32, column);
    }

    fn norm_read(task: Task, source: Value, spans: uvec4, row: u32, column: u32) -> f32 {
        return read_frame(task, source, norm_at(spans, source, row, column));
    }

    fn norm_staged(
        task: Task,
        source: Value,
        output: Value,
        spans: uvec4,
        row: u32,
        column: u32,
    ) -> f32 {
        if task.prelude_steps == 0u32 {
            return norm_read(task, source, spans, row, column);
        }
        return fetch(output, row * spans.x + column);
    }

    fn template_norm_row(task: Task, lid: u32, source: Value, output: Value, row: u32) {
        let spans = norm_spans(task, source);
        let columns = spans.x;
        let group = task.op == NORM_GROUP;
        let layer = task.op == NORM_LAYER || group;
        let rms = task.op == NORM_RMS;
        let staged = task.prelude_steps > 0u32;
        let walked = staged || group;
        let row_at = coordinates(row * columns, source.dims);
        let row_base = read_address(row_at, source.strides);
        let mut floor = 0.0;
        if task.c != NO_VALUE {
            floor = fetch(values[task.c], 0u32);
        }
        let mut total = 0.0;
        let mut square = 0.0;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let mut value = 0.0;
            if walked {
                value = norm_read(task, source, spans, row, column);
            } else {
                value = fetch(source, row_base + column * source.strides.w);
            }
            total = total + value;
            square = square + value * value;
            if staged {
                publish(output, row * columns + column, value);
            }
        }
        let mut mean = 0.0;
        let mut scale = 0.0;
        if layer {
            mean = workgroup_sum(lid, total) / f32(columns);
            let mut spread = 0.0;
            for column in stride(lid, columns, WORKGROUP_SIZE) {
                let mut value = 0.0;
                if walked {
                    value = norm_staged(task, source, output, spans, row, column);
                } else {
                    value = fetch(source, row_base + column * source.strides.w);
                }
                let centered = value - mean;
                spread = spread + centered * centered;
            }
            let variance = workgroup_sum(lid, spread) / f32(columns);
            scale = 1.0 / sqrt(max(variance, 0.0) + floor);
        } else {
            let held = workgroup_sum(lid, square);
            let mean_square = select(held, held / f32(columns), rms);
            scale = 1.0 / sqrt(mean_square + floor);
        }
        if lid == 0u32 && task.extra != NO_VALUE {
            publish(values[task.extra], row * 2u32, mean);
            publish(values[task.extra], row * 2u32 + 1u32, scale);
        }
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let mut value = 0.0;
            if walked {
                value = norm_staged(task, source, output, spans, row, column);
            } else {
                value = fetch(source, row_base + column * source.strides.w);
            }
            let mut result = select(value * scale, (value - mean) * scale, layer);
            if task.b != NO_VALUE {
                result = result * fetch(values[task.b], norm_channel(spans, row, column));
            }
            if task.d != NO_VALUE {
                result = result + fetch(values[task.d], norm_channel(spans, row, column));
            }
            let mut at = row_at + uvec4(0u32, 0u32, 0u32, column);
            if group {
                at = norm_at(spans, source, row, column);
            }
            publish(output, row * columns + column, chained(task, at, result));
        }
    }

    fn template_norm_grad_row(
        task: Task,
        lid: u32,
        source: Value,
        gradient: Value,
        output: Value,
        row: u32,
    ) {
        let spans = norm_spans(task, source);
        let columns = spans.x;
        let stats = values[task.c];
        let mean = fetch(stats, row * 2u32);
        let scale = fetch(stats, row * 2u32 + 1u32);
        let group = task.op == NORM_GROUP;
        let layer = task.op == NORM_LAYER || group;
        let staged = task.prelude_steps > 0u32;
        let walked = staged || group;
        let row_at = coordinates(row * columns, source.dims);
        let row_base = read_address(row_at, source.strides);
        let mut dot = 0.0;
        let mut first = 0.0;
        let mut second = 0.0;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let mut value = 0.0;
            if walked {
                value = norm_read(task, source, spans, row, column);
            } else {
                value = fetch(source, row_base + column * source.strides.w);
            }
            if staged {
                publish(output, row * columns + column, value);
            }
            let held = fetch(gradient, row * columns + column);
            let mut weight = 1.0;
            if task.b != NO_VALUE {
                weight = fetch(values[task.b], norm_channel(spans, row, column));
            }
            if layer {
                first = first + held * weight;
                second = second + held * weight * (value - mean);
            } else {
                dot = dot + held * weight * value;
            }
            if task.extra != NO_VALUE {
                publish(
                    values[task.extra],
                    row * columns + column,
                    (value - mean) * scale,
                );
            }
        }
        let mut held_dot = 0.0;
        let mut held_first = 0.0;
        let mut held_second = 0.0;
        if layer {
            held_first = workgroup_sum(lid, first) / f32(columns);
            held_second = workgroup_sum(lid, second) / f32(columns);
        } else {
            held_dot = workgroup_sum(lid, dot);
        }
        let cubed = scale * scale * scale;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let mut value = 0.0;
            if walked {
                value = norm_staged(task, source, output, spans, row, column);
            } else {
                value = fetch(source, row_base + column * source.strides.w);
            }
            let held = fetch(gradient, row * columns + column);
            let mut weight = 1.0;
            if task.b != NO_VALUE {
                weight = fetch(values[task.b], norm_channel(spans, row, column));
            }
            let mut result = 0.0;
            if layer {
                result = scale * weight * held
                    - scale * held_first
                    - cubed * (value - mean) * held_second;
            } else {
                let factor = select(
                    cubed * held_dot,
                    cubed * held_dot / f32(columns),
                    task.op == NORM_RMS,
                );
                result = scale * weight * held - factor * value;
            }
            let mut at = row_at + uvec4(0u32, 0u32, 0u32, column);
            if group {
                at = norm_at(spans, source, row, column);
            }
            publish(output, row * columns + column, chained(task, at, result));
        }
    }

    fn norm_band(columns: u32, sections: u32, section: u32) -> uvec4 {
        let shared = columns / sections;
        let rest = columns % sections;
        let start = section * shared + min(section, rest);
        let end = (section + 1u32) * shared + min(section + 1u32, rest);
        return uvec4(start, end - start, 0u32, 0u32);
    }

    fn run_norm_section(task: Task, lid: u32) {
        let source = values[task.a];
        let partials = values[task.out];
        let spans = norm_spans(task, source);
        let columns = spans.x;
        let band = norm_band(columns, task.splits, task.section);
        let layer = task.op == NORM_LAYER || task.op == NORM_GROUP;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let mut total = 0.0;
            let mut square = 0.0;
            for column in stride(lid, band.y, WORKGROUP_SIZE) {
                let value = norm_read(task, source, spans, row, band.x + column);
                total = total + value;
                square = square + value * value;
            }
            let mut mean = 0.0;
            let mut spread = 0.0;
            if layer {
                mean = workgroup_sum(lid, total) / f32(band.y);
                for column in stride(lid, band.y, WORKGROUP_SIZE) {
                    let value = norm_read(task, source, spans, row, band.x + column);
                    let centered = value - mean;
                    spread = spread + centered * centered;
                }
                spread = workgroup_sum(lid, spread);
            } else {
                spread = workgroup_sum(lid, square);
            }
            if lid == 0u32 {
                publish(partials, (row * task.splits + task.section) * 2u32, mean);
                publish(
                    partials,
                    (row * task.splits + task.section) * 2u32 + 1u32,
                    spread,
                );
            }
            workgroup_barrier();
        }
    }

    fn run_norm_fold(task: Task, lid: u32) {
        let partials = values[task.a];
        let stats = values[task.out];
        let sections = task.splits;
        let columns = task.axis;
        let layer = task.op == NORM_LAYER || task.op == NORM_GROUP;
        let mut floor = 0.0;
        if task.c != NO_VALUE {
            floor = fetch(values[task.c], 0u32);
        }
        for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let mut mean = 0.0;
            let mut scale = 0.0;
            if layer {
                let mut count = 0u32;
                let mut m2 = 0.0;
                for section in stride(0u32, sections, 1u32) {
                    let band = norm_band(columns, sections, section);
                    let section_mean = fetch(partials, (row * sections + section) * 2u32);
                    let section_m2 = fetch(partials, (row * sections + section) * 2u32 + 1u32);
                    let combined = count + band.y;
                    let delta = section_mean - mean;
                    mean = mean + delta * f32(band.y) / f32(combined);
                    m2 = m2 + section_m2 + delta * delta * f32(count) * f32(band.y) / f32(combined);
                    count = combined;
                }
                scale = 1.0 / sqrt(m2 / f32(columns) + floor);
            } else {
                let mut total = 0.0;
                for section in stride(0u32, sections, 1u32) {
                    total = total + fetch(partials, (row * sections + section) * 2u32 + 1u32);
                }
                let mean_square = select(total, total / f32(columns), task.op == NORM_RMS);
                scale = 1.0 / sqrt(mean_square + floor);
            }
            publish(stats, row * 2u32, mean);
            publish(stats, row * 2u32 + 1u32, scale);
        }
    }

    fn run_norm_apply(task: Task, lid: u32) {
        let source = values[task.a];
        let stats = values[task.c];
        let output = values[task.out];
        let spans = norm_spans(task, source);
        let columns = spans.x;
        let band = norm_band(columns, task.splits, task.section);
        let layer = task.op == NORM_LAYER || task.op == NORM_GROUP;
        if task.geometry != strategy::WORKGROUP_ROW {
            refuse(kind::NORM_APPLY, refusal::GEOMETRY, task.geometry);
            return;
        }
        for row in stride(task.first, task.first + task.count, 1u32) {
            let mean = fetch(stats, row * 2u32);
            let scale = fetch(stats, row * 2u32 + 1u32);
            for column in stride(lid, band.y, WORKGROUP_SIZE) {
                let value = norm_read(task, source, spans, row, band.x + column);
                let mut result = select(value * scale, (value - mean) * scale, layer);
                if task.b != NO_VALUE {
                    result =
                        result * fetch(values[task.b], norm_channel(spans, row, band.x + column));
                }
                if task.d != NO_VALUE {
                    result =
                        result + fetch(values[task.d], norm_channel(spans, row, band.x + column));
                }
                publish(
                    output,
                    row * columns + band.x + column,
                    chained(task, norm_at(spans, source, row, band.x + column), result),
                );
            }
            workgroup_barrier();
        }
    }

    fn run_norm(task: Task, lid: u32) {
        if task.splits > 1u32 {
            run_norm_section(task, lid);
            return;
        }
        let source = values[task.a];
        let output = values[task.out];
        let spans = norm_spans(task, source);
        if spans.x == 0u32 {
            return;
        }
        if task.geometry != strategy::WORKGROUP_ROW {
            refuse(kind::NORM, refusal::GEOMETRY, task.geometry);
            return;
        }
        for row in stride(task.first, task.first + task.count, 1u32) {
            template_norm_row(task, lid, source, output, row);
            workgroup_barrier();
        }
    }

    fn run_norm_grad(task: Task, lid: u32) {
        let source = values[task.a];
        let gradient = values[task.d];
        let output = values[task.out];
        let spans = norm_spans(task, source);
        if spans.x == 0u32 {
            return;
        }
        if task.geometry != strategy::WORKGROUP_ROW {
            refuse(kind::NORM_GRAD, refusal::GEOMETRY, task.geometry);
            return;
        }
        for row in stride(task.first, task.first + task.count, 1u32) {
            template_norm_grad_row(task, lid, source, gradient, output, row);
            workgroup_barrier();
        }
    }
}
