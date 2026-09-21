fn chain_operand(step: Step, at: vec4<u32>) -> f32 {
    if (step.operand == NO_VALUE) {
        return 0.0;
    }
    let source = values[step.operand];
    return fetch(source, read_address(at, source.strides));
}

fn chained(task: Task, at: vec4<u32>, carried: f32) -> f32 {
    var result = carried;
    for (var step = 0u; step < task.steps; step = step + 1u) {
        let record = steps[task.chain + step];
        let operand = chain_operand(record, at);
        let swapped = record.swapped == 1u;
        result = op_apply(
            task.kind,
            record.op,
            select(result, operand, swapped),
            select(operand, result, swapped),
        );
    }
    return result;
}

fn run_binary(task: Task, lid: u32) {
    let left = values[task.a];
    let right = values[task.b];
    let output = values[task.out];
    let dims = output.dims;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, dims);
        let a = fetch(left, read_address(at, left.strides));
        let b = fetch(right, read_address(at, right.strides));
        publish(output, index, chained(task, at, op_apply(task.kind, task.op, a, b)));
    }
}

fn run_unary(task: Task, lid: u32) {
    let source = values[task.a];
    let output = values[task.out];
    let dims = output.dims;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, dims);
        let a = fetch(source, read_address(at, source.strides));
        publish(output, index, chained(task, at, op_apply(task.kind, task.op, a, 0.0)));
    }
}

fn run_fill(task: Task, lid: u32) {
    let output = values[task.out];
    let dims = output.dims;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        publish(output, index, chained(task, coordinates(index, dims), task.param));
    }
}

fn run_broadcast(task: Task, lid: u32) {
    let output = values[task.out];
    let source = values[task.a];
    let dims = output.dims;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, dims);
        publish(output, index, chained(task, at, fetch(source, read_address(at, source.strides))));
    }
}
