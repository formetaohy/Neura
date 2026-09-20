fn chain_operand(step: Step, index: u32, dims: vec4<u32>) -> f32 {
    if (step.operand == NO_VALUE) {
        return 0.0;
    }
    let source = values[step.operand];
    return arena[source.base + value_offset(index, dims, source.strides)];
}

fn chain_apply(op: u32, carried: f32, operand: f32) -> f32 {
    switch (op) {
        case CHAIN_ADD: { return carried + operand; }
        case CHAIN_MUL: { return carried * operand; }
        case CHAIN_RELU: { return max(carried, 0.0); }
        case CHAIN_SQRT: { return sqrt(carried); }
        case CHAIN_RECIP: { return 1.0 / carried; }
        default: { refuse(REFUSED_CHAIN, op); return 0.0; }
    }
}

fn chained(task: Task, index: u32, carried: f32) -> f32 {
    var result = carried;
    let dims = values[task.out].dims;
    for (var step = 0u; step < task.steps; step = step + 1u) {
        let record = steps[task.chain + step];
        result = chain_apply(record.op, result, chain_operand(record, index, dims));
    }
    return result;
}
