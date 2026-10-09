use neura_compiler::Compiler;
use neura_profile::Geometry;

pub(crate) fn install(compiler: &mut Compiler, geometry: &Geometry) {
    compiler.constant("SCRATCH_CHOICE", geometry.choice());
    compiler.constant("SCRATCH_SAMPLE", 2u32 * geometry.workgroup());
    compiler.constant("SAMPLE_CANDIDATES", neura_profile::SAMPLE_CANDIDATES);
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn choice_precedes(value: f32, index: u32, other_value: f32, other_index: u32) -> bool {
        if other_value > value {
            return true;
        }
        return other_value == value && other_index < index;
    }

    fn workgroup_choice(lid: u32, value: f32, index: u32) -> u32 {
        scratch[lid] = value;
        scratch[SCRATCH_CHOICE + lid] = bitcast_f32(index);
        workgroup_barrier();
        let mut stride = WORKGROUP_SIZE / 2u32;
        loop {
            if stride == 0u32 {
                break;
            }
            if lid < stride
                && choice_precedes(
                    scratch[lid],
                    bitcast_u32(scratch[SCRATCH_CHOICE + lid]),
                    scratch[lid + stride],
                    bitcast_u32(scratch[SCRATCH_CHOICE + lid + stride]),
                )
            {
                scratch[lid] = scratch[lid + stride];
                scratch[SCRATCH_CHOICE + lid] = scratch[SCRATCH_CHOICE + lid + stride];
            }
            workgroup_barrier();
            stride = stride / 2u32;
        }
        let chosen = bitcast_u32(scratch[SCRATCH_CHOICE]);
        workgroup_barrier();
        return chosen;
    }

    fn element_coordinates(source: Value, row: u32, column: u32) -> uvec4 {
        return coordinates(row * source.dims.w + column, source.dims);
    }

    fn gumbel_noise(seed: u32, at: uvec4) -> f32 {
        let mut hash = seed ^ (element_key(at) * 0x9e3779b9u32);
        hash = hash ^ (hash >> 16u32);
        hash = hash * 0x7feb352du32;
        hash = hash ^ (hash >> 15u32);
        hash = hash * 0x846ca68bu32;
        hash = hash ^ (hash >> 16u32);
        return -log(-log(f32(hash >> 8u32) * (1.0 / 16777216.0)));
    }

    fn choice_weight(
        value: f32,
        seed: u32,
        noised: bool,
        source: Value,
        row: u32,
        column: u32,
    ) -> f32 {
        if !noised {
            return value;
        }
        return value + gumbel_noise(seed, element_coordinates(source, row, column));
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
        let mut local = max_identity();
        let mut local_index = 0u32;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let weight = choice_weight(
                read_flat(task, source, row * columns + column),
                seed,
                noised,
                source,
                row,
                column,
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
        let mut local = max_identity();
        let mut local_index = 0u32;
        for column in stride(0u32, columns, 1u32) {
            let weight = choice_weight(
                read_flat(task, source, row * columns + column),
                seed,
                noised,
                source,
                row,
                column,
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
            _ => refuse(kind::ARGMAX, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_categorical(task: Task, lid: u32) {
        let source = values[task.a];
        let seed = bitcast_u32(fetch(values[task.b], 0u32));
        match task.geometry {
            strategy::THREAD_ROW => fold_rows_by_thread(task, lid, source, seed, true),
            strategy::WORKGROUP_ROW => fold_rows_by_workgroup(task, lid, source, seed, true),
            _ => refuse(kind::CATEGORICAL, refusal::GEOMETRY, task.geometry),
        }
    }

    fn sample_after(value: f32, column: u32, previous_value: f32, previous_index: u32) -> bool {
        if previous_index == NO_VALUE {
            return true;
        }
        if value < previous_value {
            return true;
        }
        return value == previous_value && column > previous_index;
    }

    fn sample_mass(lid: u32, max_value: f32, total: f32) -> f32 {
        scratch[lid] = max_value;
        scratch[SCRATCH_CHOICE + lid] = total;
        workgroup_barrier();
        let mut stride = WORKGROUP_SIZE / 2u32;
        loop {
            if stride == 0u32 {
                break;
            }
            if lid < stride {
                let left_max = scratch[lid];
                let left_total = scratch[SCRATCH_CHOICE + lid];
                let right_max = scratch[lid + stride];
                let right_total = scratch[SCRATCH_CHOICE + lid + stride];
                let merged_max = max(left_max, right_max);
                let mut merged_total = 0.0;
                if left_total > 0.0 || right_total > 0.0 {
                    merged_total = left_total * softmax_exp(left_max - merged_max)
                        + right_total * softmax_exp(right_max - merged_max);
                }
                scratch[lid] = merged_max;
                scratch[SCRATCH_CHOICE + lid] = merged_total;
            }
            workgroup_barrier();
            stride = stride / 2u32;
        }
        return scratch[SCRATCH_CHOICE];
    }

    fn sample_row(
        task: Task,
        lid: u32,
        source: Value,
        row: u32,
        columns: u32,
        keep: u32,
        cumulative: f32,
        seed: u32,
    ) -> u32 {
        let mut local_max = max_identity();
        let mut local_total = 0.0;
        for column in stride(lid, columns, WORKGROUP_SIZE) {
            let value = read_flat(task, source, row * columns + column);
            if local_max == max_identity() {
                local_max = value;
                local_total = select(0.0, 1.0, value > max_identity());
            } else {
                if value > local_max {
                    local_total = local_total * softmax_exp(local_max - value) + 1.0;
                    local_max = value;
                } else {
                    local_total = local_total + softmax_exp(value - local_max);
                }
            }
        }
        let total = sample_mass(lid, local_max, local_total);
        let row_max = scratch[0u32];
        workgroup_barrier();
        let mut count = 0u32;
        let mut previous_value = max_identity();
        let mut previous_index = NO_VALUE;
        loop {
            if count >= keep {
                break;
            }
            let mut local_value = max_identity();
            let mut local_index = NO_VALUE;
            for column in stride(lid, columns, WORKGROUP_SIZE) {
                let value = read_flat(task, source, row * columns + column);
                if sample_after(value, column, previous_value, previous_index)
                    && (local_index == NO_VALUE
                        || choice_precedes(local_value, local_index, value, column))
                {
                    local_value = value;
                    local_index = column;
                }
            }
            let chosen_index = workgroup_choice(lid, local_value, local_index);
            let chosen_value = scratch[0u32];
            if chosen_index != NO_VALUE {
                previous_value = chosen_value;
                previous_index = chosen_index;
            }
            if lid == 0u32 {
                scratch[SCRATCH_SAMPLE + count] = chosen_value;
                scratch[SCRATCH_SAMPLE + SAMPLE_CANDIDATES + count] = bitcast_f32(chosen_index);
            }
            count = count + 1u32;
            workgroup_barrier();
        }
        let mut chosen = NO_VALUE;
        if lid == 0u32 {
            let mut mass = 0.0;
            for candidate in stride(0u32, count, 1u32) {
                if bitcast_u32(scratch[SCRATCH_SAMPLE + SAMPLE_CANDIDATES + candidate]) != NO_VALUE
                {
                    mass = mass + softmax_exp(scratch[SCRATCH_SAMPLE + candidate] - row_max);
                }
            }
            let tail = total - mass;
            let threshold = cumulative * total;
            let mut boundary = count;
            let mut running = 0.0;
            for candidate in stride(0u32, count, 1u32) {
                if bitcast_u32(scratch[SCRATCH_SAMPLE + SAMPLE_CANDIDATES + candidate]) != NO_VALUE
                {
                    running = running + softmax_exp(scratch[SCRATCH_SAMPLE + candidate] - row_max);
                    if running + tail >= threshold && candidate + 1u32 < boundary {
                        boundary = candidate + 1u32;
                    }
                }
            }
            let mut best = max_identity();
            for candidate in stride(0u32, boundary, 1u32) {
                let value = scratch[SCRATCH_SAMPLE + candidate];
                let column = bitcast_u32(scratch[SCRATCH_SAMPLE + SAMPLE_CANDIDATES + candidate]);
                let weight =
                    value + gumbel_noise(seed, coordinates(row * columns + column, source.dims));
                if chosen == NO_VALUE || choice_precedes(best, chosen, weight, column) {
                    best = weight;
                    chosen = column;
                }
            }
        }
        if chosen == NO_VALUE {
            chosen = 0u32;
        }
        return chosen;
    }

    fn sample_rows(task: Task, lid: u32, source: Value, seed: u32, keep: u32, cumulative: f32) {
        let output = values[task.out];
        let columns = source.dims.w;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let chosen = sample_row(task, lid, source, row, columns, keep, cumulative, seed);
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

    fn run_sample(task: Task, lid: u32) {
        let source = values[task.a];
        let seed = bitcast_u32(fetch(values[task.b], 0u32));
        if lid == 0u32 {
            scratch[SCRATCH_SAMPLE] = fetch(values[task.c], 0u32);
            scratch[SCRATCH_SAMPLE + 1u32] = fetch(values[task.d], 0u32);
        }
        let declared = workgroup_uniform_load(&scratch[SCRATCH_SAMPLE]);
        let cumulative = workgroup_uniform_load(&scratch[SCRATCH_SAMPLE + 1u32]);
        let whole = declared > 0.0
            && declared < f32(SAMPLE_CANDIDATES + 1u32)
            && trunc(declared) == declared;
        if !whole {
            refuse(kind::SAMPLE, refusal::SAMPLE, 0u32);
            return;
        }
        if !(cumulative > 0.0) || !(cumulative <= 1.0) {
            refuse(kind::SAMPLE, refusal::SAMPLE, 1u32);
            return;
        }
        let keep = u32(declared);
        match task.geometry {
            strategy::WORKGROUP_ROW => sample_rows(task, lid, source, seed, keep, cumulative),
            _ => refuse(kind::SAMPLE, refusal::GEOMETRY, task.geometry),
        }
    }
}
