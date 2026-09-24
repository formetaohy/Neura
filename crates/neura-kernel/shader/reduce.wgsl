fn run_sum_chunk(task: Task, lid: u32, slot: u32) {
    let source = values[task.a];
    let output = values[task.out];
    var local = 0.0;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        local = local + fetch(source, index);
    }
    let total = workgroup_sum(lid, local);
    if (lid == 0u) {
        publish(output, task.slot, total);
    }
}

fn sum_row_with_workgroup(lid: u32, source: Value, row: u32, columns: u32) -> f32 {
    var local = 0.0;
    for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
        local = local + fetch(source, row_origin(source, row) + column * source.strides.w);
    }
    return workgroup_sum(lid, local);
}

fn sum_row_with_thread(source: Value, row: u32, columns: u32) -> f32 {
    var local = 0.0;
    for (var column = 0u; column < columns; column = column + 1u) {
        local = local + fetch(source, row_origin(source, row) + column * source.strides.w);
    }
    return local;
}

fn sum_rows_with_workgroup(task: Task, lid: u32, slot: u32, source: Value, columns: u32) {
    let output = values[task.out];
    for (var row = task.first; row < task.first + task.count; row = row + 1u) {
        let total = sum_row_with_workgroup(lid, source, row, columns);
        if (lid == 0u) {
            publish(output, row, chained(task, coordinates(row, output.dims), total, slot));
        }
        workgroupBarrier();
    }
}

fn sum_rows_with_thread(task: Task, lid: u32, slot: u32, source: Value, columns: u32) {
    let output = values[task.out];
    for (var row = task.first + lid; row < task.first + task.count; row = row + WORKGROUP_SIZE) {
        publish(output, row, chained(task, coordinates(row, output.dims), sum_row_with_thread(source, row, columns), slot));
    }
}

fn sum_axis_element(task: Task, lid: u32, slot: u32, source: Value, output: Value, axis: u32) {
    let step = vec4<u32>(
        select(0u, 1u, axis == 0u),
        select(0u, 1u, axis == 1u),
        select(0u, 1u, axis == 2u),
        select(0u, 1u, axis == 3u),
    );
    let folds = component(source.dims, axis);
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, output.dims);
        var total = 0.0;
        for (var fold = 0u; fold < folds; fold = fold + 1u) {
            total = total + fetch(source, read_address(at + step * fold, source.strides));
        }
        publish(output, index, chained(task, at, total, slot));
    }
}

fn run_sum_axis(task: Task, lid: u32, slot: u32) {
    let source = values[task.a];
    let output = values[task.out];
    switch (task.geometry) {
        case THREAD_ELEMENT: { sum_axis_element(task, lid, slot, source, output, task.slot); }
        case THREAD_ROW: {
            if (task.slot == 3u) {
                sum_rows_with_thread(task, lid, slot, source, source.dims.w);
            } else {
                refuse(slot, SumAxis, task.geometry);
            }
        }
        case WORKGROUP_ROW: {
            if (task.slot == 3u) {
                sum_rows_with_workgroup(task, lid, slot, source, source.dims.w);
            } else {
                refuse(slot, SumAxis, task.geometry);
            }
        }
        default: { refuse(slot, SumAxis, task.geometry); }
    }
}
