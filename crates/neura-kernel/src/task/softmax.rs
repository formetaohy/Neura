use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    compiler.constant("SOFTMAX_STREAM", neura_abi::SOFTMAX_STREAM);
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn section_first(columns: u32, sections: u32, section: u32) -> u32 {
        return columns * section / sections;
    }

    fn section_last(columns: u32, sections: u32, section: u32) -> u32 {
        return columns * (section + 1u32) / sections;
    }

    fn section_stats(
        task: Task,
        lid: u32,
        source: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
    ) {
        let stats = values[task.out];
        let mut local_max = max_identity();
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
            local_max = max(local_max, fetch(source, row * columns + column));
        }
        let held = workgroup_max(lid, local_max);
        let mut local_sum = 0.0;
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
            local_sum = local_sum + softmax_exp(fetch(source, row * columns + column) - held);
        }
        let mass = workgroup_sum(lid, local_sum);
        if lid == 0u32 {
            let base = row * 2u32 * task.splits + 2u32 * task.section;
            publish(stats, base, held);
            publish(stats, base + 1u32, mass);
        }
    }

    fn section_dot(
        task: Task,
        lid: u32,
        probability: Value,
        gradient: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
    ) {
        let stats = values[task.out];
        let mut local = 0.0;
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
            local = local
                + fetch(gradient, row * columns + column)
                    * fetch(probability, row * columns + column);
        }
        let total = workgroup_sum(lid, local);
        if lid == 0u32 {
            publish(stats, row * task.splits + task.section, total);
        }
    }

    fn section_gradient_total(
        task: Task,
        lid: u32,
        gradient: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
    ) {
        let stats = values[task.out];
        let mut local = 0.0;
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
            local = local + fetch(gradient, row * columns + column);
        }
        let total = workgroup_sum(lid, local);
        if lid == 0u32 {
            publish(stats, row * task.splits + task.section, total);
        }
    }

    fn template_softmax_apply(
        task: Task,
        lid: u32,
        source: Value,
        output: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
    ) {
        let stats = values[task.c];
        let row_max = fetch(stats, row * 2u32);
        let row_sum = fetch(stats, row * 2u32 + 1u32);
        let row_at = coordinates(row * columns, output.dims);
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
            publish(
                output,
                row * columns + column,
                chained(
                    task,
                    row_at + uvec4(0u32, 0u32, 0u32, column),
                    softmax_exp(fetch(source, row * columns + column) - row_max) / row_sum,
                ),
            );
        }
    }

    fn template_log_softmax_apply(
        task: Task,
        lid: u32,
        source: Value,
        output: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
    ) {
        let stats = values[task.c];
        let normalizer = fetch(stats, row * 2u32) + positive_log(fetch(stats, row * 2u32 + 1u32));
        let row_at = coordinates(row * columns, output.dims);
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
            publish(
                output,
                row * columns + column,
                chained(
                    task,
                    row_at + uvec4(0u32, 0u32, 0u32, column),
                    fetch(source, row * columns + column) - normalizer,
                ),
            );
        }
    }

    fn template_softmax_grad_apply(
        task: Task,
        lid: u32,
        probability: Value,
        gradient: Value,
        output: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
    ) {
        let total = fetch(values[task.c], row);
        let row_at = coordinates(row * columns, output.dims);
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
            let at = row * columns + column;
            let held = fetch(probability, at);
            publish(
                output,
                at,
                chained(
                    task,
                    row_at + uvec4(0u32, 0u32, 0u32, column),
                    held * (fetch(gradient, at) - total),
                ),
            );
        }
    }

    fn template_log_softmax_grad_apply(
        task: Task,
        lid: u32,
        probability: Value,
        gradient: Value,
        output: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
    ) {
        let total = fetch(values[task.c], row);
        let row_at = coordinates(row * columns, output.dims);
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
            let at = row * columns + column;
            publish(
                output,
                at,
                chained(
                    task,
                    row_at + uvec4(0u32, 0u32, 0u32, column),
                    fetch(gradient, at) - softmax_exp(fetch(probability, at)) * total,
                ),
            );
        }
    }

    fn row_max_by_thread(source: Value, row: u32, columns: u32) -> f32 {
        let streamed = columns / SOFTMAX_STREAM * SOFTMAX_STREAM;
        let mut maximum = max_identity();
        for base in stride(0u32, streamed, SOFTMAX_STREAM) {
            let mut held = scalar_array(max_identity(), SOFTMAX_STREAM);
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                held[at] = fetch(source, row * columns + base + at);
            }
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                maximum = max(maximum, held[at]);
            }
        }
        for column in stride(streamed, columns, 1u32) {
            maximum = max(maximum, fetch(source, row * columns + column));
        }
        return maximum;
    }

    fn row_mass_by_thread(source: Value, row: u32, columns: u32, maximum: f32) -> f32 {
        let streamed = columns / SOFTMAX_STREAM * SOFTMAX_STREAM;
        let mut mass = 0.0;
        for base in stride(0u32, streamed, SOFTMAX_STREAM) {
            let mut held = scalar_array(0.0, SOFTMAX_STREAM);
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                held[at] = softmax_exp(fetch(source, row * columns + base + at) - maximum);
            }
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                mass = mass + held[at];
            }
        }
        for column in stride(streamed, columns, 1u32) {
            mass = mass + softmax_exp(fetch(source, row * columns + column) - maximum);
        }
        return mass;
    }

    fn row_dot_by_thread(probability: Value, gradient: Value, row: u32, columns: u32) -> f32 {
        let streamed = columns / SOFTMAX_STREAM * SOFTMAX_STREAM;
        let mut dot = 0.0;
        for base in stride(0u32, streamed, SOFTMAX_STREAM) {
            let mut held = scalar_array(0.0, SOFTMAX_STREAM);
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                held[at] = fetch(probability, row * columns + base + at)
                    * fetch(gradient, row * columns + base + at);
            }
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                dot = dot + held[at];
            }
        }
        for column in stride(streamed, columns, 1u32) {
            dot = dot
                + fetch(probability, row * columns + column)
                    * fetch(gradient, row * columns + column);
        }
        return dot;
    }

    fn row_total_by_thread(gradient: Value, row: u32, columns: u32) -> f32 {
        let streamed = columns / SOFTMAX_STREAM * SOFTMAX_STREAM;
        let mut total = 0.0;
        for base in stride(0u32, streamed, SOFTMAX_STREAM) {
            let mut held = scalar_array(0.0, SOFTMAX_STREAM);
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                held[at] = fetch(gradient, row * columns + base + at);
            }
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                total = total + held[at];
            }
        }
        for column in stride(streamed, columns, 1u32) {
            total = total + fetch(gradient, row * columns + column);
        }
        return total;
    }

    fn softmax_row_by_thread(task: Task, source: Value, output: Value, row: u32, columns: u32) {
        let row_at = coordinates(row * columns, output.dims);
        let maximum = row_max_by_thread(source, row, columns);
        let mass = row_mass_by_thread(source, row, columns, maximum);
        let streamed = columns / SOFTMAX_STREAM * SOFTMAX_STREAM;
        for base in stride(0u32, streamed, SOFTMAX_STREAM) {
            let mut held = scalar_array(0.0, SOFTMAX_STREAM);
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                held[at] = softmax_exp(fetch(source, row * columns + base + at) - maximum) / mass;
            }
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                let column = base + at;
                publish(
                    output,
                    row * output.dims.w + column,
                    chained(task, row_at + uvec4(0u32, 0u32, 0u32, column), held[at]),
                );
            }
        }
        for column in stride(streamed, columns, 1u32) {
            publish(
                output,
                row * output.dims.w + column,
                chained(
                    task,
                    row_at + uvec4(0u32, 0u32, 0u32, column),
                    softmax_exp(fetch(source, row * columns + column) - maximum) / mass,
                ),
            );
        }
    }

    fn log_softmax_row_by_thread(task: Task, source: Value, output: Value, row: u32, columns: u32) {
        let row_at = coordinates(row * columns, output.dims);
        let maximum = row_max_by_thread(source, row, columns);
        let normalizer = positive_log(row_mass_by_thread(source, row, columns, maximum));
        let streamed = columns / SOFTMAX_STREAM * SOFTMAX_STREAM;
        for base in stride(0u32, streamed, SOFTMAX_STREAM) {
            let mut held = scalar_array(0.0, SOFTMAX_STREAM);
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                held[at] = fetch(source, row * columns + base + at) - maximum - normalizer;
            }
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                let column = base + at;
                publish(
                    output,
                    row * output.dims.w + column,
                    chained(task, row_at + uvec4(0u32, 0u32, 0u32, column), held[at]),
                );
            }
        }
        for column in stride(streamed, columns, 1u32) {
            publish(
                output,
                row * output.dims.w + column,
                chained(
                    task,
                    row_at + uvec4(0u32, 0u32, 0u32, column),
                    fetch(source, row * columns + column) - maximum - normalizer,
                ),
            );
        }
    }

    fn softmax_grad_row_by_thread(
        task: Task,
        probability: Value,
        gradient: Value,
        output: Value,
        row: u32,
        columns: u32,
    ) {
        let row_at = coordinates(row * columns, output.dims);
        let dot = row_dot_by_thread(probability, gradient, row, columns);
        let streamed = columns / SOFTMAX_STREAM * SOFTMAX_STREAM;
        for base in stride(0u32, streamed, SOFTMAX_STREAM) {
            let mut held = scalar_array(0.0, SOFTMAX_STREAM);
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                held[at] = fetch(probability, row * columns + base + at)
                    * (fetch(gradient, row * columns + base + at) - dot);
            }
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                let column = base + at;
                publish(
                    output,
                    row * output.dims.w + column,
                    chained(task, row_at + uvec4(0u32, 0u32, 0u32, column), held[at]),
                );
            }
        }
        for column in stride(streamed, columns, 1u32) {
            publish(
                output,
                row * output.dims.w + column,
                chained(
                    task,
                    row_at + uvec4(0u32, 0u32, 0u32, column),
                    fetch(probability, row * columns + column)
                        * (fetch(gradient, row * columns + column) - dot),
                ),
            );
        }
    }

    fn log_softmax_grad_row_by_thread(
        task: Task,
        probability: Value,
        gradient: Value,
        output: Value,
        row: u32,
        columns: u32,
    ) {
        let row_at = coordinates(row * columns, output.dims);
        let total = row_total_by_thread(gradient, row, columns);
        let streamed = columns / SOFTMAX_STREAM * SOFTMAX_STREAM;
        for base in stride(0u32, streamed, SOFTMAX_STREAM) {
            let mut held = scalar_array(0.0, SOFTMAX_STREAM);
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                held[at] = fetch(gradient, row * columns + base + at)
                    - softmax_exp(fetch(probability, row * columns + base + at)) * total;
            }
            for at in unroll(0u32, SOFTMAX_STREAM, 1u32) {
                let column = base + at;
                publish(
                    output,
                    row * output.dims.w + column,
                    chained(task, row_at + uvec4(0u32, 0u32, 0u32, column), held[at]),
                );
            }
        }
        for column in stride(streamed, columns, 1u32) {
            publish(
                output,
                row * output.dims.w + column,
                chained(
                    task,
                    row_at + uvec4(0u32, 0u32, 0u32, column),
                    fetch(gradient, row * columns + column)
                        - softmax_exp(fetch(probability, row * columns + column)) * total,
                ),
            );
        }
    }

    fn chosen_class(task: Task, classes: Value, row: u32, columns: u32, subject: u32) -> u32 {
        return whole_index(fetch(classes, row), columns, subject, refusal::INDEX);
    }

    fn class_log_softmax_row_by_thread(
        task: Task,
        source: Value,
        classes: Value,
        output: Value,
        statistic: Value,
        row: u32,
        columns: u32,
    ) {
        let maximum = row_max_by_thread(source, row, columns);
        let normalizer = maximum + positive_log(row_mass_by_thread(source, row, columns, maximum));
        let chosen = fetch(
            source,
            row * columns + chosen_class(task, classes, row, columns, kind::CLASS_LOG_SOFTMAX),
        );
        publish(output, row, chosen - normalizer);
        publish(statistic, row, normalizer);
    }

    fn class_log_softmax_grad_span(
        task: Task,
        source: Value,
        classes: Value,
        gradient: Value,
        statistic: Value,
        output: Value,
        row: u32,
        columns: u32,
        opening: u32,
        closing: u32,
        step: u32,
    ) {
        let normalizer = fetch(statistic, row);
        let held = fetch(gradient, row);
        let chosen = chosen_class(task, classes, row, columns, kind::CLASS_LOG_SOFTMAX_GRAD);
        let row_at = coordinates(row * output.dims.w, output.dims);
        for column in stride(opening, closing, step) {
            let at = row_at + uvec4(0u32, 0u32, 0u32, column);
            let probability = softmax_exp(fetch(source, row * columns + column) - normalizer);
            publish(
                output,
                row * output.dims.w + column,
                chained(
                    task,
                    at,
                    held * (select(0.0, 1.0, column == chosen) - probability),
                ),
            );
        }
    }

    fn run_class_log_softmax(task: Task, lid: u32) {
        let source = values[task.a];
        let classes = values[task.b];
        let output = values[task.out];
        let columns = source.dims.w;
        if columns == 0u32 {
            return;
        }
        let sections = task.splits;
        let first = section_first(columns, sections, task.section);
        let last = section_last(columns, sections, task.section);
        if sections > 1u32 {
            for row in stride(task.first, task.first + task.count, 1u32) {
                section_stats(task, lid, source, row, columns, first, last);
                workgroup_barrier();
            }
            return;
        }
        let statistic = values[task.extra];
        match task.geometry {
            strategy::THREAD_ROW => {
                for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
                    class_log_softmax_row_by_thread(
                        task, source, classes, output, statistic, row, columns,
                    );
                }
            }
            strategy::WORKGROUP_ROW => {
                for row in stride(task.first, task.first + task.count, 1u32) {
                    let mut local_max = max_identity();
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        local_max = max(local_max, fetch(source, row * columns + column));
                    }
                    let row_max = workgroup_max(lid, local_max);
                    let mut local_sum = 0.0;
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        local_sum = local_sum
                            + softmax_exp(fetch(source, row * columns + column) - row_max);
                    }
                    let row_sum = workgroup_sum(lid, local_sum);
                    if lid == 0u32 {
                        let normalizer = row_max + positive_log(row_sum);
                        let chosen = fetch(
                            source,
                            row * columns
                                + chosen_class(
                                    task,
                                    classes,
                                    row,
                                    columns,
                                    kind::CLASS_LOG_SOFTMAX,
                                ),
                        );
                        publish(output, row, chosen - normalizer);
                        publish(statistic, row, normalizer);
                    }
                    workgroup_barrier();
                }
            }
            _ => refuse(kind::CLASS_LOG_SOFTMAX, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_class_log_softmax_fold(task: Task, lid: u32) {
        let stats = values[task.a];
        let source = values[task.b];
        let classes = values[task.c];
        let output = values[task.out];
        let statistic = values[task.extra];
        let columns = source.dims.w;
        let sections = task.splits;
        let step = (sections + WORKGROUP_SIZE - 1u32) / WORKGROUP_SIZE;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let base = row * 2u32 * sections;
            let mut local_max = max_identity();
            for at in stride(0u32, step, 1u32) {
                let other = lid * step + at;
                if other < sections {
                    local_max = max(local_max, fetch(stats, base + 2u32 * other));
                }
            }
            let row_max = workgroup_max(lid, local_max);
            let mut local_sum = 0.0;
            for at in stride(0u32, step, 1u32) {
                let other = lid * step + at;
                if other < sections {
                    let held = fetch(stats, base + 2u32 * other);
                    local_sum = local_sum
                        + fetch(stats, base + 2u32 * other + 1u32) * softmax_exp(held - row_max);
                }
            }
            let row_sum = workgroup_sum(lid, local_sum);
            if lid == 0u32 {
                let normalizer = row_max + positive_log(row_sum);
                let chosen = fetch(
                    source,
                    row * columns
                        + chosen_class(task, classes, row, columns, kind::CLASS_LOG_SOFTMAX_FOLD),
                );
                publish(output, row, chosen - normalizer);
                publish(statistic, row, normalizer);
            }
            workgroup_barrier();
        }
    }

    fn run_class_log_softmax_grad(task: Task, lid: u32) {
        let source = values[task.a];
        let classes = values[task.b];
        let gradient = values[task.c];
        let statistic = values[task.d];
        let output = values[task.out];
        let columns = source.dims.w;
        if columns == 0u32 {
            return;
        }
        let sections = task.splits;
        let first = section_first(columns, sections, task.section);
        let last = section_last(columns, sections, task.section);
        if sections > 1u32 {
            for row in stride(task.first, task.first + task.count, 1u32) {
                class_log_softmax_grad_span(
                    task,
                    source,
                    classes,
                    gradient,
                    statistic,
                    output,
                    row,
                    columns,
                    first + lid,
                    last,
                    WORKGROUP_SIZE,
                );
            }
            return;
        }
        match task.geometry {
            strategy::THREAD_ROW => {
                for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
                    class_log_softmax_grad_span(
                        task, source, classes, gradient, statistic, output, row, columns, 0u32,
                        columns, 1u32,
                    );
                }
            }
            strategy::WORKGROUP_ROW => {
                for row in stride(task.first, task.first + task.count, 1u32) {
                    class_log_softmax_grad_span(
                        task,
                        source,
                        classes,
                        gradient,
                        statistic,
                        output,
                        row,
                        columns,
                        lid,
                        columns,
                        WORKGROUP_SIZE,
                    );
                }
            }
            _ => refuse(
                kind::CLASS_LOG_SOFTMAX_GRAD,
                refusal::GEOMETRY,
                task.geometry,
            ),
        }
    }

    fn run_softmax_fold(task: Task, lid: u32) {
        let stats = values[task.a];
        let folded = values[task.out];
        let sections = task.splits;
        let step = (sections + WORKGROUP_SIZE - 1u32) / WORKGROUP_SIZE;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let base = row * 2u32 * sections;
            let mut local_max = max_identity();
            for at in stride(0u32, step, 1u32) {
                let other = lid * step + at;
                if other < sections {
                    local_max = max(local_max, fetch(stats, base + 2u32 * other));
                }
            }
            let row_max = workgroup_max(lid, local_max);
            let mut local_sum = 0.0;
            for at in stride(0u32, step, 1u32) {
                let other = lid * step + at;
                if other < sections {
                    let held = fetch(stats, base + 2u32 * other);
                    local_sum = local_sum
                        + fetch(stats, base + 2u32 * other + 1u32) * softmax_exp(held - row_max);
                }
            }
            let row_sum = workgroup_sum(lid, local_sum);
            if lid == 0u32 {
                publish(folded, row * 2u32, row_max);
                publish(folded, row * 2u32 + 1u32, row_sum);
            }
            workgroup_barrier();
        }
    }

    fn run_softmax_grad_fold(task: Task, lid: u32) {
        let stats = values[task.a];
        let folded = values[task.out];
        let sections = task.splits;
        let step = (sections + WORKGROUP_SIZE - 1u32) / WORKGROUP_SIZE;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let mut local = 0.0;
            for at in stride(0u32, step, 1u32) {
                let other = lid * step + at;
                if other < sections {
                    local = local + fetch(stats, row * sections + other);
                }
            }
            let total = workgroup_sum(lid, local);
            if lid == 0u32 {
                publish(folded, row, total);
            }
            workgroup_barrier();
        }
    }

    fn run_softmax(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let columns = source.dims.w;
        if columns == 0u32 {
            return;
        }
        let sections = task.splits;
        let first = section_first(columns, sections, task.section);
        let last = section_last(columns, sections, task.section);
        if task.c != NO_VALUE {
            for row in stride(task.first, task.first + task.count, 1u32) {
                template_softmax_apply(task, lid, source, output, row, columns, first, last);
            }
            return;
        }
        if sections > 1u32 {
            for row in stride(task.first, task.first + task.count, 1u32) {
                section_stats(task, lid, source, row, columns, first, last);
                workgroup_barrier();
            }
            return;
        }
        match task.geometry {
            strategy::THREAD_ROW => {
                for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
                    softmax_row_by_thread(task, source, output, row, columns);
                }
            }
            strategy::WORKGROUP_ROW => {
                for row in stride(task.first, task.first + task.count, 1u32) {
                    let row_at = coordinates(row * columns, output.dims);
                    let mut local_max = max_identity();
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        local_max = max(local_max, fetch(source, row * columns + column));
                    }
                    let row_max = workgroup_max(lid, local_max);
                    let mut local_sum = 0.0;
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        local_sum = local_sum
                            + softmax_exp(fetch(source, row * columns + column) - row_max);
                    }
                    let row_sum = workgroup_sum(lid, local_sum);
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        let index = row * output.dims.w + column;
                        publish(
                            output,
                            index,
                            chained(
                                task,
                                row_at + uvec4(0u32, 0u32, 0u32, column),
                                softmax_exp(fetch(source, row * columns + column) - row_max)
                                    / row_sum,
                            ),
                        );
                    }
                    workgroup_barrier();
                }
            }
            _ => refuse(kind::SOFTMAX, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_log_softmax(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let columns = source.dims.w;
        if columns == 0u32 {
            return;
        }
        let sections = task.splits;
        let first = section_first(columns, sections, task.section);
        let last = section_last(columns, sections, task.section);
        if task.c != NO_VALUE {
            for row in stride(task.first, task.first + task.count, 1u32) {
                template_log_softmax_apply(task, lid, source, output, row, columns, first, last);
            }
            return;
        }
        if sections > 1u32 {
            for row in stride(task.first, task.first + task.count, 1u32) {
                section_stats(task, lid, source, row, columns, first, last);
                workgroup_barrier();
            }
            return;
        }
        match task.geometry {
            strategy::THREAD_ROW => {
                for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
                    log_softmax_row_by_thread(task, source, output, row, columns);
                }
            }
            strategy::WORKGROUP_ROW => {
                for row in stride(task.first, task.first + task.count, 1u32) {
                    let row_at = coordinates(row * columns, output.dims);
                    let mut local_max = max_identity();
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        local_max = max(local_max, fetch(source, row * columns + column));
                    }
                    let row_max = workgroup_max(lid, local_max);
                    let mut local_sum = 0.0;
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        local_sum = local_sum
                            + softmax_exp(fetch(source, row * columns + column) - row_max);
                    }
                    let row_sum = workgroup_sum(lid, local_sum);
                    let normalizer = positive_log(row_sum);
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        let index = row * output.dims.w + column;
                        publish(
                            output,
                            index,
                            chained(
                                task,
                                row_at + uvec4(0u32, 0u32, 0u32, column),
                                fetch(source, row * columns + column) - row_max - normalizer,
                            ),
                        );
                    }
                    workgroup_barrier();
                }
            }
            _ => refuse(kind::LOG_SOFTMAX, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_softmax_grad(task: Task, lid: u32) {
        let probability = values[task.a];
        let gradient = values[task.b];
        let output = values[task.out];
        let columns = probability.dims.w;
        if columns == 0u32 {
            return;
        }
        let sections = task.splits;
        let first = section_first(columns, sections, task.section);
        let last = section_last(columns, sections, task.section);
        if task.c != NO_VALUE {
            for row in stride(task.first, task.first + task.count, 1u32) {
                template_softmax_grad_apply(
                    task,
                    lid,
                    probability,
                    gradient,
                    output,
                    row,
                    columns,
                    first,
                    last,
                );
            }
            return;
        }
        if sections > 1u32 {
            for row in stride(task.first, task.first + task.count, 1u32) {
                section_dot(task, lid, probability, gradient, row, columns, first, last);
                workgroup_barrier();
            }
            return;
        }
        match task.geometry {
            strategy::THREAD_ROW => {
                for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
                    softmax_grad_row_by_thread(task, probability, gradient, output, row, columns);
                }
            }
            strategy::WORKGROUP_ROW => {
                for row in stride(task.first, task.first + task.count, 1u32) {
                    let row_at = coordinates(row * columns, output.dims);
                    let mut local = 0.0;
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        local = local
                            + fetch(gradient, row * columns + column)
                                * fetch(probability, row * columns + column);
                    }
                    let row_dot = workgroup_sum(lid, local);
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        let index = row * output.dims.w + column;
                        let y = fetch(probability, row * columns + column);
                        let g = fetch(gradient, row * columns + column);
                        publish(
                            output,
                            index,
                            chained(
                                task,
                                row_at + uvec4(0u32, 0u32, 0u32, column),
                                y * (g - row_dot),
                            ),
                        );
                    }
                    workgroup_barrier();
                }
            }
            _ => refuse(kind::SOFTMAX_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_log_softmax_grad(task: Task, lid: u32) {
        let probability = values[task.a];
        let gradient = values[task.b];
        let output = values[task.out];
        let columns = probability.dims.w;
        if columns == 0u32 {
            return;
        }
        let sections = task.splits;
        let first = section_first(columns, sections, task.section);
        let last = section_last(columns, sections, task.section);
        if task.c != NO_VALUE {
            for row in stride(task.first, task.first + task.count, 1u32) {
                template_log_softmax_grad_apply(
                    task,
                    lid,
                    probability,
                    gradient,
                    output,
                    row,
                    columns,
                    first,
                    last,
                );
            }
            return;
        }
        if sections > 1u32 {
            for row in stride(task.first, task.first + task.count, 1u32) {
                section_gradient_total(task, lid, gradient, row, columns, first, last);
                workgroup_barrier();
            }
            return;
        }
        match task.geometry {
            strategy::THREAD_ROW => {
                for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
                    log_softmax_grad_row_by_thread(
                        task,
                        probability,
                        gradient,
                        output,
                        row,
                        columns,
                    );
                }
            }
            strategy::WORKGROUP_ROW => {
                for row in stride(task.first, task.first + task.count, 1u32) {
                    let row_at = coordinates(row * columns, output.dims);
                    let mut local = 0.0;
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        local = local + fetch(gradient, row * columns + column);
                    }
                    let row_total = workgroup_sum(lid, local);
                    for column in stride(lid, columns, WORKGROUP_SIZE) {
                        let index = row * output.dims.w + column;
                        let y = fetch(probability, row * columns + column);
                        let g = fetch(gradient, row * columns + column);
                        publish(
                            output,
                            index,
                            chained(
                                task,
                                row_at + uvec4(0u32, 0u32, 0u32, column),
                                g - softmax_exp(y) * row_total,
                            ),
                        );
                    }
                    workgroup_barrier();
                }
            }
            _ => refuse(kind::LOG_SOFTMAX_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }
}
