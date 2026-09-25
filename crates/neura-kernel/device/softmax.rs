#[neura_compiler::module]
mod source {
    fn run_softmax(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let columns = source.dims.w;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let row_at = coordinates(row * columns, output.dims);
            let mut local_max = -3.4028235e38;
            for column in stride(lid, columns, WORKGROUP_SIZE) {
                local_max = max(local_max, fetch(source, row * columns + column));
            }
            let row_max = workgroup_max(lid, local_max);
            let mut local_sum = 0.0;
            for column in stride(lid, columns, WORKGROUP_SIZE) {
                local_sum = local_sum + exp(fetch(source, row * columns + column) - row_max);
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
                        exp(fetch(source, row * columns + column) - row_max) / row_sum,
                    ),
                );
            }
            workgroup_barrier();
        }
    }

    fn run_softmax_grad(task: Task, lid: u32) {
        let probability = values[task.a];
        let gradient = values[task.b];
        let output = values[task.out];
        let columns = probability.dims.w;
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

    fn run_log_softmax(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let columns = source.dims.w;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let row_at = coordinates(row * columns, output.dims);
            let mut local_max = -3.4028235e38;
            for column in stride(lid, columns, WORKGROUP_SIZE) {
                local_max = max(local_max, fetch(source, row * columns + column));
            }
            let row_max = workgroup_max(lid, local_max);
            let mut local_sum = 0.0;
            for column in stride(lid, columns, WORKGROUP_SIZE) {
                local_sum = local_sum + exp(fetch(source, row * columns + column) - row_max);
            }
            let row_sum = workgroup_sum(lid, local_sum);
            let normalizer = log(row_sum);
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

    fn run_log_softmax_grad(task: Task, lid: u32) {
        let probability = values[task.a];
        let gradient = values[task.b];
        let output = values[task.out];
        let columns = probability.dims.w;
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
                        g - exp(y) * row_total,
                    ),
                );
            }
            workgroup_barrier();
        }
    }
}
