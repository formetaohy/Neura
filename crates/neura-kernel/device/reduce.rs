#[neura_compiler::module]
mod source {
    fn workgroup_sum(lid: u32, start: f32) -> f32 {
        reduction_scratch[lid] = start;
        workgroup_barrier();
        let mut stride = WORKGROUP_SIZE / 2u32;
        loop {
            if stride == 0u32 {
                break;
            }
            if lid < stride {
                reduction_scratch[lid] = reduction_scratch[lid] + reduction_scratch[lid + stride];
            }
            workgroup_barrier();
            stride = stride / 2u32;
        }
        let total = reduction_scratch[0u32];
        workgroup_barrier();
        return total;
    }

    fn workgroup_max(lid: u32, start: f32) -> f32 {
        reduction_scratch[lid] = start;
        workgroup_barrier();
        let mut stride = WORKGROUP_SIZE / 2u32;
        loop {
            if stride == 0u32 {
                break;
            }
            if lid < stride {
                reduction_scratch[lid] =
                    max(reduction_scratch[lid], reduction_scratch[lid + stride]);
            }
            workgroup_barrier();
            stride = stride / 2u32;
        }
        let total = reduction_scratch[0u32];
        workgroup_barrier();
        return total;
    }

    fn run_sum_chunk(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let mut local = 0.0;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            local = local + read_flat(task, source, index);
        }
        let total = workgroup_sum(lid, local);
        if lid == 0u32 {
            publish(output, task.slot, total);
        }
    }

    fn sum_row_with_workgroup(task: Task, lid: u32, source: Value, row: u32, columns: u32) -> f32 {
        let mut local = 0.0;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            local = local + read_flat(task, source, row * columns + column);
        }
        return workgroup_sum(lid, local);
    }

    fn sum_row_with_thread(task: Task, source: Value, row: u32, columns: u32) -> f32 {
        let mut local = 0.0;
        for column in stride(0u32, columns, 1u32) {
            local = local + read_flat(task, source, row * columns + column);
        }
        return local;
    }

    fn sum_rows_with_workgroup(task: Task, lid: u32, source: Value, columns: u32) {
        let output = values[task.out];
        for row in stride(task.first, task.first + task.count, 1u32) {
            let total = sum_row_with_workgroup(task, lid, source, row, columns);
            if lid == 0u32 {
                publish(
                    output,
                    row,
                    chained(task, coordinates(row, output.dims), total),
                );
            }
            workgroup_barrier();
        }
    }

    fn sum_rows_with_thread(task: Task, lid: u32, source: Value, columns: u32) {
        let output = values[task.out];
        for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            publish(
                output,
                row,
                chained(
                    task,
                    coordinates(row, output.dims),
                    sum_row_with_thread(task, source, row, columns),
                ),
            );
        }
    }

    fn sum_axis_element(task: Task, lid: u32, source: Value, output: Value, axis: u32) {
        let step = uvec4(
            select(0u32, 1u32, axis == 0u32),
            select(0u32, 1u32, axis == 1u32),
            select(0u32, 1u32, axis == 2u32),
            select(0u32, 1u32, axis == 3u32),
        );
        let folds = component(source.dims, axis);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            let mut total = 0.0;
            for fold in stride(0u32, folds, 1u32) {
                total = total + read_frame(task, source, at + step * fold);
            }
            publish(output, index, chained(task, at, total));
        }
    }

    fn run_sum_axis(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        match task.geometry {
            strategy::THREAD_ELEMENT => sum_axis_element(task, lid, source, output, task.slot),
            strategy::THREAD_ROW => {
                if task.slot == 3u32 {
                    sum_rows_with_thread(task, lid, source, source.dims.w);
                } else {
                    refuse(kind::SUM_AXIS, task.geometry);
                }
            }
            strategy::WORKGROUP_ROW => {
                if task.slot == 3u32 {
                    sum_rows_with_workgroup(task, lid, source, source.dims.w);
                } else {
                    refuse(kind::SUM_AXIS, task.geometry);
                }
            }
            _ => refuse(kind::SUM_AXIS, task.geometry),
        }
    }
}
