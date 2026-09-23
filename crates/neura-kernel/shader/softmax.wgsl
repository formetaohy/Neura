fn run_softmax(task: Task, lid: u32) {
    let source = values[task.a];
    let output = values[task.out];
    let columns = source.dims.w;
    for (var row = task.first; row < task.first + task.count; row = row + 1u) {
        let row_at = coordinates(row * columns, output.dims);
        var local_max = -3.4028235e38;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local_max = max(local_max, fetch(source, row_origin(source, row) + column * source.strides.w));
        }
        let row_max = workgroup_max(lid, local_max);
        var local_sum = 0.0;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local_sum = local_sum + exp(fetch(source, row_origin(source, row) + column * source.strides.w) - row_max);
        }
        let row_sum = workgroup_sum(lid, local_sum);
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            let index = row * output.dims.w + column;
            publish(output, index, chained(task, row_at + vec4<u32>(0u, 0u, 0u, column), exp(fetch(source, row_origin(source, row) + column * source.strides.w) - row_max) / row_sum));
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
        let row_at = coordinates(row * columns, output.dims);
        var local = 0.0;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local = local + fetch(gradient, row * columns + column) * fetch(probability, row * columns + column);
        }
        let row_dot = workgroup_sum(lid, local);
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            let index = row * output.dims.w + column;
            let y = fetch(probability, row * columns + column);
            let g = fetch(gradient, row * columns + column);
            publish(output, index, chained(task, row_at + vec4<u32>(0u, 0u, 0u, column), y * (g - row_dot)));
        }
        workgroupBarrier();
    }
}

fn run_log_softmax(task: Task, lid: u32) {
    let source = values[task.a];
    let output = values[task.out];
    let columns = source.dims.w;
    for (var row = task.first; row < task.first + task.count; row = row + 1u) {
        let row_at = coordinates(row * columns, output.dims);
        var local_max = -3.4028235e38;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local_max = max(local_max, fetch(source, row_origin(source, row) + column * source.strides.w));
        }
        let row_max = workgroup_max(lid, local_max);
        var local_sum = 0.0;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local_sum = local_sum + exp(fetch(source, row_origin(source, row) + column * source.strides.w) - row_max);
        }
        let row_sum = workgroup_sum(lid, local_sum);
        let normalizer = log(row_sum);
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            let index = row * output.dims.w + column;
            publish(output, index, chained(task, row_at + vec4<u32>(0u, 0u, 0u, column), fetch(source, row_origin(source, row) + column * source.strides.w) - row_max - normalizer));
        }
        workgroupBarrier();
    }
}

fn run_log_softmax_grad(task: Task, lid: u32) {
    let probability = values[task.a];
    let gradient = values[task.b];
    let output = values[task.out];
    let columns = probability.dims.w;
    for (var row = task.first; row < task.first + task.count; row = row + 1u) {
        let row_at = coordinates(row * columns, output.dims);
        var local = 0.0;
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            local = local + fetch(gradient, row * columns + column);
        }
        let row_total = workgroup_sum(lid, local);
        for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
            let index = row * output.dims.w + column;
            let y = fetch(probability, row * columns + column);
            let g = fetch(gradient, row * columns + column);
            publish(output, index, chained(task, row_at + vec4<u32>(0u, 0u, 0u, column), g - exp(y) * row_total));
        }
        workgroupBarrier();
    }
}
