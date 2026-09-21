var<workgroup> choice_index: array<u32, WORKGROUP_SIZE>;

fn choice_precedes(value: f32, index: u32, other_value: f32, other_index: u32) -> bool {
    if (other_value > value) {
        return true;
    }
    return other_value == value && other_index < index;
}

fn workgroup_choice(lid: u32, value: f32, index: u32) -> u32 {
    reduction_scratch[lid] = value;
    choice_index[lid] = index;
    workgroupBarrier();
    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride && choice_precedes(reduction_scratch[lid], choice_index[lid], reduction_scratch[lid + stride], choice_index[lid + stride])) {
            reduction_scratch[lid] = reduction_scratch[lid + stride];
            choice_index[lid] = choice_index[lid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    let chosen = choice_index[0];
    workgroupBarrier();
    return chosen;
}

fn gumbel_noise(seed: u32, index: u32) -> f32 {
    var hash = seed ^ (index * 0x9e3779b9u);
    hash = hash ^ (hash >> 16u);
    hash = hash * 0x7feb352du;
    hash = hash ^ (hash >> 15u);
    hash = hash * 0x846ca68bu;
    hash = hash ^ (hash >> 16u);
    return -log(-log(f32(hash >> 8u) * (1.0 / 16777216.0)));
}

fn choice_weight(value: f32, seed: u32, index: u32, noised: bool) -> f32 {
    if (!noised) {
        return value;
    }
    return value + gumbel_noise(seed, index);
}

fn fold_row_by_workgroup(lid: u32, source: Value, row: u32, columns: u32, seed: u32, noised: bool) -> u32 {
    var local = -3.4028235e38;
    var local_index = 0u;
    for (var column = lid; column < columns; column = column + WORKGROUP_SIZE) {
        let weight = choice_weight(fetch(source.base, row * columns + column), seed, row * columns + column, noised);
        if (weight > local) {
            local = weight;
            local_index = column;
        }
    }
    return workgroup_choice(lid, local, local_index);
}

fn fold_row_by_thread(source: Value, row: u32, columns: u32, seed: u32, noised: bool) -> u32 {
    var local = -3.4028235e38;
    var local_index = 0u;
    for (var column = 0u; column < columns; column = column + 1u) {
        let weight = choice_weight(fetch(source.base, row * columns + column), seed, row * columns + column, noised);
        if (weight > local) {
            local = weight;
            local_index = column;
        }
    }
    return local_index;
}

fn fold_rows_by_workgroup(task: Task, lid: u32, source: Value, seed: u32, noised: bool) {
    let output = values[task.out];
    let columns = source.dims.w;
    for (var row = task.first; row < task.first + task.count; row = row + 1u) {
        let chosen = fold_row_by_workgroup(lid, source, row, columns, seed, noised);
        if (lid == 0u) {
            publish(output.base, row, chained(task, coordinates(row, output.dims), f32(chosen)));
        }
        workgroupBarrier();
    }
}

fn fold_rows_by_thread(task: Task, lid: u32, source: Value, seed: u32, noised: bool) {
    let output = values[task.out];
    let columns = source.dims.w;
    for (var row = task.first + lid; row < task.first + task.count; row = row + WORKGROUP_SIZE) {
        publish(output.base, row, chained(task, coordinates(row, output.dims), f32(fold_row_by_thread(source, row, columns, seed, noised))));
    }
}

fn run_argmax(task: Task, lid: u32) {
    let source = values[task.a];
    switch (task.geometry) {
        case THREAD_ROW: { fold_rows_by_thread(task, lid, source, 0u, false); }
        case WORKGROUP_ROW: { fold_rows_by_workgroup(task, lid, source, 0u, false); }
        default: { refuse(Argmax, task.geometry); }
    }
}

fn run_categorical(task: Task, lid: u32) {
    let source = values[task.a];
    let seed = bitcast<u32>(fetch(values[task.b].base, 0));
    switch (task.geometry) {
        case THREAD_ROW: { fold_rows_by_thread(task, lid, source, seed, true); }
        case WORKGROUP_ROW: { fold_rows_by_workgroup(task, lid, source, seed, true); }
        default: { refuse(Categorical, task.geometry); }
    }
}
