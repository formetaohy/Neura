fn value_offset(flat: u32, dims: vec4<u32>, strides: vec4<u32>) -> u32 {
    let w = flat % dims.w;
    let z = (flat / dims.w) % dims.z;
    let y = (flat / (dims.w * dims.z)) % dims.y;
    let x = flat / (dims.w * dims.z * dims.y);
    return x * strides.x + y * strides.y + z * strides.z + w * strides.w;
}

fn chain_operand(step: Step, index: u32, dims: vec4<u32>) -> f32 {
    if (step.operand == NO_VALUE) {
        return 0.0;
    }
    let source = values[step.operand];
    return arena[source.base + value_offset(index, dims, source.strides)];
}

fn chained(task: Task, index: u32, carried: f32) -> f32 {
    var result = carried;
    let dims = values[task.out].dims;
    for (var step = 0u; step < task.steps; step = step + 1u) {
        let record = steps[task.chain + step];
        let operand = chain_operand(record, index, dims);
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
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let a = arena[left.base + value_offset(index, output.dims, left.strides)];
        let b = arena[right.base + value_offset(index, output.dims, right.strides)];
        arena[output.base + index] = chained(task, index, op_apply(task.kind, task.op, a, b));
    }
}

fn run_unary(task: Task, lid: u32) {
    let source = values[task.a];
    let output = values[task.out];
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let a = arena[source.base + value_offset(index, output.dims, source.strides)];
        arena[output.base + index] = chained(task, index, op_apply(task.kind, task.op, a, 0.0));
    }
}

fn run_fill(task: Task, lid: u32) {
    let output = values[task.out];
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        arena[output.base + index] = chained(task, index, task.param);
    }
}

fn run_broadcast(task: Task, lid: u32) {
    let output = values[task.out];
    let scalar = arena[values[task.a].base + task.slot];
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        arena[output.base + index] = chained(task, index, scalar);
    }
}
