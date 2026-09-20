fn run_softmax(task: Task, lid: u32) {
    let source = values[task.a];
    let output = values[task.out];
    let columns = source.dims.w;
    for (var row = task.first; row < task.first + task.count; row = row + 1u) {
        let source_base = source.base + row * columns;
        let output_base = output.base + row * output.dims.w;
        var local_max = -3.4028235e38;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local_max = max(local_max, arena[source_base + column]);
        }
        let row_max = softmax_row_max(lid, local_max);
        var local_sum = 0.0;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local_sum = local_sum + exp(arena[source_base + column] - row_max);
        }
        let row_sum = softmax_row_sum(lid, local_sum);
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            arena[output_base + column] = chained(task, row * output.dims.w + column, exp(arena[source_base + column] - row_max) / row_sum);
        }
        workgroupBarrier();
    }
}

fn run_softmax_grad(task: Task, lid: u32) {
    let probability = values[task.a];
    let gradient = values[task.b];
    let output = values[task.out];
    let columns = probability.dims.w;
    for (var row = task.first; row < task.first + task.count; row = row + 1u) {
        let probability_base = probability.base + row * columns;
        let gradient_base = gradient.base + row * columns;
        let output_base = output.base + row * output.dims.w;
        var local = 0.0;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local = local + arena[gradient_base + column] * arena[probability_base + column];
        }
        let row_dot = softmax_row_sum(lid, local);
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            let y = arena[probability_base + column];
            let g = arena[gradient_base + column];
            arena[output_base + column] = chained(task, row * output.dims.w + column, y * (g - row_dot));
        }
        workgroupBarrier();
    }
}
