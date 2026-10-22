use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    compiler.constant("NORM_RMS", neura_abi::NORM_RMS);
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn template_norm_row(
        task: Task,
        lid: u32,
        source: Value,
        output: Value,
        row: u32,
        columns: u32,
    ) {
        let rms = task.op == NORM_RMS;
        let row_at = coordinates(row * columns, source.dims);
        let row_base = read_address(row_at, source.strides);
        let mut floor = 0.0;
        if task.c != NO_VALUE {
            floor = fetch(values[task.c], 0u32);
        }
        let mut local = 0.0;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let value = fetch(source, row_base + column * source.strides.w);
            local = local + value * value;
        }
        let total = workgroup_sum(lid, local);
        let mean = select(total, total / f32(columns), rms);
        let scale = 1.0 / sqrt(mean + floor);
        if lid == 0u32 {
            if task.extra != NO_VALUE {
                publish(values[task.extra], row, scale);
            }
        }
        let at = coordinates(row * columns, output.dims);
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let mut weight = 1.0;
            if task.b != NO_VALUE {
                weight = fetch(values[task.b], column);
            }
            publish(
                output,
                row * columns + column,
                chained(
                    task,
                    at + uvec4(0u32, 0u32, 0u32, column),
                    fetch(source, row_base + column * source.strides.w) * scale * weight,
                ),
            );
        }
    }

    fn template_norm_grad_row(
        task: Task,
        lid: u32,
        source: Value,
        gradient: Value,
        output: Value,
        row: u32,
        columns: u32,
    ) {
        let rms = task.op == NORM_RMS;
        let scale = fetch(values[task.c], row);
        let row_at = coordinates(row * columns, source.dims);
        let row_base = read_address(row_at, source.strides);
        let mut local = 0.0;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let mut weight = 1.0;
            if task.b != NO_VALUE {
                weight = fetch(values[task.b], column);
            }
            local = local
                + fetch(gradient, row * columns + column)
                    * fetch(source, row_base + column * source.strides.w)
                    * weight;
        }
        let dot = workgroup_sum(lid, local);
        let cubed = scale * scale * scale;
        let factor = select(cubed * dot, cubed * dot / f32(columns), rms);
        let at = coordinates(row * columns, output.dims);
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let mut weight = 1.0;
            if task.b != NO_VALUE {
                weight = fetch(values[task.b], column);
            }
            let value = scale * weight * fetch(gradient, row * columns + column)
                - factor * fetch(source, row_base + column * source.strides.w);
            publish(
                output,
                row * columns + column,
                chained(task, at + uvec4(0u32, 0u32, 0u32, column), value),
            );
        }
    }

    fn run_norm(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let columns = source.dims.w;
        if columns == 0u32 {
            return;
        }
        if task.geometry != strategy::WORKGROUP_ROW {
            refuse(kind::NORM, refusal::GEOMETRY, task.geometry);
            return;
        }
        for row in stride(task.first, task.first + task.count, 1u32) {
            template_norm_row(task, lid, source, output, row, columns);
            workgroup_barrier();
        }
    }

    fn run_norm_grad(task: Task, lid: u32) {
        let source = values[task.a];
        let gradient = values[task.d];
        let output = values[task.out];
        let columns = source.dims.w;
        if columns == 0u32 {
            return;
        }
        if task.geometry != strategy::WORKGROUP_ROW {
            refuse(kind::NORM_GRAD, refusal::GEOMETRY, task.geometry);
            return;
        }
        for row in stride(task.first, task.first + task.count, 1u32) {
            template_norm_grad_row(task, lid, source, gradient, output, row, columns);
            workgroup_barrier();
        }
    }
}
