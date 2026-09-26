#[neura_compiler::module]
mod source {
    fn sigmoid(x: f32) -> f32 {
        return 1.0 / (1.0 + exp(-x));
    }

    fn chain_operand(step: Step, at: uvec4) -> f32 {
        if step.operand == NO_VALUE {
            return 0.0;
        }
        let source = values[step.operand];
        return fetch(source, read_address(at, source.strides));
    }

    fn chained(task: Task, at: uvec4, carried: f32) -> f32 {
        let mut result = carried;
        for step in stride(0u32, task.steps, 1u32) {
            let record = steps[task.chain + step];
            let operand = chain_operand(record, at);
            let swapped = record.swapped == 1u32;
            result = op_apply(
                task.kind,
                record.op,
                select(result, operand, swapped),
                select(operand, result, swapped),
            );
        }
        return result;
    }

    fn opened(task: Task, at: uvec4, carried: f32) -> f32 {
        let mut result = carried;
        for step in stride(0u32, task.prelude_steps, 1u32) {
            let record = steps[task.prelude + step];
            let operand = chain_operand(record, at);
            let swapped = record.swapped == 1u32;
            result = op_apply(
                task.kind,
                record.op,
                select(result, operand, swapped),
                select(operand, result, swapped),
            );
        }
        return result;
    }

    fn read_frame(task: Task, source: Value, at: uvec4) -> f32 {
        let carried = fetch(source, read_address(at, source.strides));
        if task.prelude_steps == 0u32 {
            return carried;
        }
        return opened(task, at, carried);
    }

    fn read_flat(task: Task, source: Value, index: u32) -> f32 {
        if task.prelude_steps == 0u32 {
            return fetch(source, index);
        }
        let at = coordinates(index, source.dims);
        return opened(task, at, fetch(source, read_address(at, source.strides)));
    }

    fn run_binary(task: Task, lid: u32) {
        let left = values[task.a];
        let right = values[task.b];
        let output = values[task.out];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, dims);
            let a = fetch(left, read_address(at, left.strides));
            let b = fetch(right, read_address(at, right.strides));
            publish(
                output,
                index,
                chained(task, at, op_apply(task.kind, task.op, a, b)),
            );
        }
    }

    fn run_unary(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, dims);
            let a = fetch(source, read_address(at, source.strides));
            publish(
                output,
                index,
                chained(task, at, op_apply(task.kind, task.op, a, 0.0)),
            );
        }
    }

    fn run_fill(task: Task, lid: u32) {
        let output = values[task.out];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            publish(
                output,
                index,
                chained(task, coordinates(index, dims), task.param),
            );
        }
    }

    fn run_broadcast(task: Task, lid: u32) {
        let output = values[task.out];
        let source = values[task.a];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, dims);
            publish(
                output,
                index,
                chained(task, at, fetch(source, read_address(at, source.strides))),
            );
        }
    }
}
