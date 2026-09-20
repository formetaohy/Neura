fn run_binary(task: Task, lid: u32) {
    let left = values[task.a];
    let right = values[task.b];
    let output = values[task.out];
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let a = arena[left.base + value_offset(index, output.dims, left.strides)];
        let b = arena[right.base + value_offset(index, output.dims, right.strides)];
        var result = 0.0;
        switch (task.flags) {
            case BINARY_ADD: { result = a + b; }
            case BINARY_MUL: { result = a * b; }
            default: { refuse(task.kind, task.flags); }
        }
        arena[output.base + index] = chained(task, index, result);
    }
}

fn run_unary(task: Task, lid: u32) {
    let source = values[task.a];
    let output = values[task.out];
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let x = arena[source.base + value_offset(index, output.dims, source.strides)];
        var result = 0.0;
        switch (task.flags) {
            case UNARY_RELU: { result = max(x, 0.0); }
            case UNARY_SQRT: { result = sqrt(x); }
            case UNARY_RECIP: { result = 1.0 / x; }
            default: { refuse(task.kind, task.flags); }
        }
        arena[output.base + index] = chained(task, index, result);
    }
}

fn run_unary_grad(task: Task, lid: u32) {
    let forward = values[task.a];
    let gradient = values[task.b];
    let output = values[task.out];
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let y = arena[forward.base + value_offset(index, output.dims, forward.strides)];
        let g = arena[gradient.base + value_offset(index, output.dims, gradient.strides)];
        var result = 0.0;
        switch (task.flags) {
            case UNARY_RELU: { result = select(0.0, g, y > 0.0); }
            case UNARY_SQRT: { result = g * 0.5 / y; }
            case UNARY_RECIP: { result = -g * y * y; }
            default: { refuse(task.kind, task.flags); }
        }
        arena[output.base + index] = chained(task, index, result);
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
