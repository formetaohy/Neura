#[neura_compiler::module]
mod source {
    fn choice_precedes(value: f32, index: u32, other_value: f32, other_index: u32) -> bool {
        if other_value > value {
            return true;
        }
        return other_value == value && other_index < index;
    }

    fn workgroup_choice(lid: u32, value: f32, index: u32) -> u32 {
        reduction_scratch[lid] = value;
        choice_index[lid] = index;
        workgroup_barrier();
        let mut stride = WORKGROUP_SIZE / 2u32;
        loop {
            if stride == 0u32 {
                break;
            }
            if lid < stride
                && choice_precedes(
                    reduction_scratch[lid],
                    choice_index[lid],
                    reduction_scratch[lid + stride],
                    choice_index[lid + stride],
                )
            {
                reduction_scratch[lid] = reduction_scratch[lid + stride];
                choice_index[lid] = choice_index[lid + stride];
            }
            workgroup_barrier();
            stride = stride / 2u32;
        }
        let chosen = choice_index[0u32];
        workgroup_barrier();
        return chosen;
    }

    fn gumbel_noise(seed: u32, index: u32) -> f32 {
        let mut hash = seed ^ (index * 0x9e3779b9u32);
        hash = hash ^ (hash >> 16u32);
        hash = hash * 0x7feb352du32;
        hash = hash ^ (hash >> 15u32);
        hash = hash * 0x846ca68bu32;
        hash = hash ^ (hash >> 16u32);
        return -log(-log(f32(hash >> 8u32) * (1.0 / 16777216.0)));
    }

    fn choice_weight(value: f32, seed: u32, index: u32, noised: bool) -> f32 {
        if !noised {
            return value;
        }
        return value + gumbel_noise(seed, index);
    }

    fn fold_row_by_workgroup(
        task: Task,
        lid: u32,
        source: Value,
        row: u32,
        columns: u32,
        seed: u32,
        noised: bool,
    ) -> u32 {
        let mut local = -3.4028235e38;
        let mut local_index = 0u32;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let weight = choice_weight(
                read_flat(task, source, row * columns + column),
                seed,
                row * columns + column,
                noised,
            );
            if weight > local {
                local = weight;
                local_index = column;
            }
        }
        return workgroup_choice(lid, local, local_index);
    }

    fn fold_row_by_thread(
        task: Task,
        source: Value,
        row: u32,
        columns: u32,
        seed: u32,
        noised: bool,
    ) -> u32 {
        let mut local = -3.4028235e38;
        let mut local_index = 0u32;
        for column in stride(0u32, columns, 1u32) {
            let weight = choice_weight(
                read_flat(task, source, row * columns + column),
                seed,
                row * columns + column,
                noised,
            );
            if weight > local {
                local = weight;
                local_index = column;
            }
        }
        return local_index;
    }

    fn fold_rows_by_workgroup(task: Task, lid: u32, source: Value, seed: u32, noised: bool) {
        let output = values[task.out];
        let columns = source.dims.w;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let chosen = fold_row_by_workgroup(task, lid, source, row, columns, seed, noised);
            if lid == 0u32 {
                publish(
                    output,
                    row,
                    chained(task, coordinates(row, output.dims), f32(chosen)),
                );
            }
            workgroup_barrier();
        }
    }

    fn fold_rows_by_thread(task: Task, lid: u32, source: Value, seed: u32, noised: bool) {
        let output = values[task.out];
        let columns = source.dims.w;
        for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            publish(
                output,
                row,
                chained(
                    task,
                    coordinates(row, output.dims),
                    f32(fold_row_by_thread(task, source, row, columns, seed, noised)),
                ),
            );
        }
    }

    fn run_argmax(task: Task, lid: u32) {
        let source = values[task.a];
        match task.geometry {
            strategy::THREAD_ROW => fold_rows_by_thread(task, lid, source, 0u32, false),
            strategy::WORKGROUP_ROW => fold_rows_by_workgroup(task, lid, source, 0u32, false),
            _ => refuse(kind::ARGMAX, task.geometry),
        }
    }

    fn run_categorical(task: Task, lid: u32) {
        let source = values[task.a];
        let seed = bitcast_u32(fetch(values[task.b], 0u32));
        match task.geometry {
            strategy::THREAD_ROW => fold_rows_by_thread(task, lid, source, seed, true),
            strategy::WORKGROUP_ROW => fold_rows_by_workgroup(task, lid, source, seed, true),
            _ => refuse(kind::CATEGORICAL, task.geometry),
        }
    }
}
