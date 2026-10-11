use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
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
        for row in stride(task.first, task.first + task.count, 1u32) {
            if task.c != NO_VALUE {
                template_softmax_apply(task, lid, source, output, row, columns, first, last);
            } else if sections > 1u32 {
                section_stats(task, lid, source, row, columns, first, last);
                workgroup_barrier();
            } else {
                let row_at = coordinates(row * columns, output.dims);
                let mut local_max = max_identity();
                for column in stride(lid, columns, WORKGROUP_SIZE) {
                    local_max = max(local_max, fetch(source, row * columns + column));
                }
                let row_max = workgroup_max(lid, local_max);
                let mut local_sum = 0.0;
                for column in stride(lid, columns, WORKGROUP_SIZE) {
                    local_sum =
                        local_sum + softmax_exp(fetch(source, row * columns + column) - row_max);
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
                            softmax_exp(fetch(source, row * columns + column) - row_max) / row_sum,
                        ),
                    );
                }
                workgroup_barrier();
            }
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
        for row in stride(task.first, task.first + task.count, 1u32) {
            if task.c != NO_VALUE {
                template_log_softmax_apply(task, lid, source, output, row, columns, first, last);
            } else if sections > 1u32 {
                section_stats(task, lid, source, row, columns, first, last);
                workgroup_barrier();
            } else {
                let row_at = coordinates(row * columns, output.dims);
                let mut local_max = max_identity();
                for column in stride(lid, columns, WORKGROUP_SIZE) {
                    local_max = max(local_max, fetch(source, row * columns + column));
                }
                let row_max = workgroup_max(lid, local_max);
                let mut local_sum = 0.0;
                for column in stride(lid, columns, WORKGROUP_SIZE) {
                    local_sum =
                        local_sum + softmax_exp(fetch(source, row * columns + column) - row_max);
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
        for row in stride(task.first, task.first + task.count, 1u32) {
            if task.c != NO_VALUE {
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
            } else if sections > 1u32 {
                section_dot(task, lid, probability, gradient, row, columns, first, last);
                workgroup_barrier();
            } else {
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
        for row in stride(task.first, task.first + task.count, 1u32) {
            if task.c != NO_VALUE {
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
            } else if sections > 1u32 {
                section_gradient_total(task, lid, gradient, row, columns, first, last);
                workgroup_barrier();
            } else {
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
    }
}
