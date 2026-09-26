#[neura_compiler::module]
mod source {
    fn op_apply(kind: u32, op: u32, a: f32, b: f32) -> f32 {
        match op {
            _ => {
                refuse(kind, refusal::OP, op);
                return 0.0;
            }
        }
    }

    fn run_partial(task: Task, lid: u32) {
        let primary = values[select(task.a, task.out, task.a == NO_VALUE)];
        let other = values[select(task.b, task.out, task.b == NO_VALUE)];
        let gradient = values[task.c];
        let output = values[task.out];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, dims);
            let g = fetch(gradient, read_address(at, gradient.strides));
            let mut result = 0.0;
            match task.op * 2u32 + task.slot {
                _ => refuse(task.kind, refusal::PARTIAL, task.op * 2u32 + task.slot),
            }
            publish(output, index, chained(task, at, result));
        }
    }
}
