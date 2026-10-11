use neura_compiler::Compiler;
use neura_profile::Geometry;

pub(crate) fn install(compiler: &mut Compiler, geometry: &Geometry) {
    compiler.constant("SCRATCH_CHOICE", geometry.choice());
    compiler.constant("SCRATCH_SAMPLE", 2u32 * geometry.workgroup());
    compiler.constant(
        "SCRATCH_UNION",
        2u32 * geometry.workgroup() + 2u32 * neura_abi::CANDIDATES,
    );
    let layout = geometry.choice_layout();
    compiler.constant("CHOICE_LOCAL", layout.local());
    compiler.constant("CHOICE_OWNERS", layout.owners());
    compiler.constant("CHOICE_ENTRIES", layout.entries());
    compiler.constant("CHOICE_LEVELS", layout.levels());
    compiler.constant("CANDIDATES", neura_abi::CANDIDATES);
    compiler.constant("CHOICE_SECTIONS", neura_abi::CHOICE_SECTIONS);
    let merge =
        2u32 * geometry.workgroup() + 2u32 * neura_abi::CANDIDATES + 2u32 * layout.entries();
    compiler.constant("SCRATCH_MERGE_VALUE", merge);
    compiler.constant("SCRATCH_MERGE_INDEX", merge + neura_abi::CHOICE_SECTIONS);
    compiler.constant(
        "SCRATCH_MERGE_WEIGHT",
        merge + 2u32 * neura_abi::CHOICE_SECTIONS,
    );
    compiler.constant(
        "SCRATCH_MERGE_CURSOR",
        merge + 3u32 * neura_abi::CHOICE_SECTIONS,
    );
    compiler.constant(
        "SCRATCH_MERGE_COUNT",
        merge + 4u32 * neura_abi::CHOICE_SECTIONS,
    );
    compiler.constant(
        "SCRATCH_MERGE_OUT",
        merge + 5u32 * neura_abi::CHOICE_SECTIONS,
    );
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
        first: u32,
        last: u32,
        seed: u32,
        noised: bool,
    ) -> u32 {
        let mut local = max_identity();
        let mut local_index = 0u32;
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
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
        first: u32,
        last: u32,
        seed: u32,
        noised: bool,
    ) -> u32 {
        let mut local = max_identity();
        let mut local_index = 0u32;
        for column in stride(first, last, 1u32) {
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
            let chosen =
                fold_row_by_workgroup(task, lid, source, row, columns, 0u32, columns, seed, noised);
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
                    f32(fold_row_by_thread(
                        task, source, row, columns, 0u32, columns, seed, noised,
                    )),
                ),
            );
        }
    }

    fn fold_section_by_workgroup(
        task: Task,
        lid: u32,
        source: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
        seed: u32,
        noised: bool,
    ) {
        let carried = values[task.out];
        let rows = carried.dims.z;
        let mut local = max_identity();
        let mut local_index = 0u32;
        for column in stride(first + lid, last, WORKGROUP_SIZE) {
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
        let chosen = workgroup_choice(lid, local, local_index);
        if lid == 0u32 {
            let base = (task.section * rows + row) * 2u32;
            publish(carried, base, scratch[0u32]);
            publish(carried, base + 1u32, f32(chosen));
        }
        workgroup_barrier();
    }

    fn fold_sections_by_workgroup(task: Task, lid: u32, source: Value, seed: u32, noised: bool) {
        let columns = source.dims.w;
        let first = columns * task.section / task.splits;
        let last = columns * (task.section + 1u32) / task.splits;
        for row in stride(task.first, task.first + task.count, 1u32) {
            fold_section_by_workgroup(task, lid, source, row, columns, first, last, seed, noised);
        }
    }

    fn run_choice_fold(task: Task, lid: u32) {
        let carried = values[task.a];
        let output = values[task.out];
        let sections = task.splits;
        let rows = carried.dims.z;
        for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let mut best = max_identity();
            let mut chosen = NO_VALUE;
            for section in stride(0u32, sections, 1u32) {
                let value = fetch(carried, (section * rows + row) * 2u32);
                let index = u32(fetch(carried, (section * rows + row) * 2u32 + 1u32));
                if better(value, index, best, chosen) {
                    best = value;
                    chosen = index;
                }
            }
            publish(
                output,
                row,
                chained(task, coordinates(row, output.dims), f32(chosen)),
            );
        }
    }

    fn run_argmax(task: Task, lid: u32) {
        let source = values[task.a];
        match task.geometry {
            strategy::THREAD_ROW => fold_rows_by_thread(task, lid, source, 0u32, false),
            strategy::WORKGROUP_ROW => {
                if task.splits > 1u32 {
                    fold_sections_by_workgroup(task, lid, source, 0u32, false);
                } else {
                    fold_rows_by_workgroup(task, lid, source, 0u32, false);
                }
            }
            _ => refuse(kind::ARGMAX, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_categorical(task: Task, lid: u32) {
        let source = values[task.a];
        let seed = bitcast_u32(fetch(values[task.b], 0u32));
        match task.geometry {
            strategy::THREAD_ROW => fold_rows_by_thread(task, lid, source, seed, true),
            strategy::WORKGROUP_ROW => {
                if task.splits > 1u32 {
                    fold_sections_by_workgroup(task, lid, source, seed, true);
                } else {
                    fold_rows_by_workgroup(task, lid, source, seed, true);
                }
            }
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

    fn better(value: f32, index: u32, other_value: f32, other_index: u32) -> bool {
        if other_index == NO_VALUE {
            return true;
        }
        if value > other_value {
            return true;
        }
        return value == other_value && index < other_index;
    }

    fn rank_row(
        task: Task,
        lid: u32,
        source: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
        keep: u32,
    ) -> u32 {
        let mut count = 0u32;
        let mut previous_value = max_identity();
        let mut previous_index = NO_VALUE;
        let span = last - first;
        loop {
            if count >= keep {
                break;
            }
            let mut local_value = scalar_array(max_identity(), CHOICE_LOCAL);
            let mut local_index = scalar_array(NO_VALUE, CHOICE_LOCAL);
            if lid < CHOICE_OWNERS {
                for base in stride(lid, span, CHOICE_OWNERS * 4u32) {
                    let mut held = scalar_array(max_identity(), 4u32);
                    for at in unroll(0u32, 4u32, 1u32) {
                        let column = first + base + at * CHOICE_OWNERS;
                        held[at] = select(
                            max_identity(),
                            read_flat(task, source, row * columns + min(column, columns - 1u32)),
                            column < last,
                        );
                    }
                    for at in unroll(0u32, 4u32, 1u32) {
                        let column = first + base + at * CHOICE_OWNERS;
                        let value = held[at];
                        if column < last
                            && sample_after(value, column, previous_value, previous_index)
                            && better(
                                value,
                                column,
                                local_value[CHOICE_LOCAL - 1u32],
                                local_index[CHOICE_LOCAL - 1u32],
                            )
                        {
                            let mut held_value = value;
                            let mut held_index = column;
                            for seat in unroll(0u32, CHOICE_LOCAL, 1u32) {
                                if better(
                                    held_value,
                                    held_index,
                                    local_value[seat],
                                    local_index[seat],
                                ) {
                                    let swap_value = local_value[seat];
                                    let swap_index = local_index[seat];
                                    local_value[seat] = held_value;
                                    local_index[seat] = held_index;
                                    held_value = swap_value;
                                    held_index = swap_index;
                                }
                            }
                        }
                    }
                }
                for seat in unroll(0u32, CHOICE_LOCAL, 1u32) {
                    scratch[SCRATCH_UNION + lid * CHOICE_LOCAL + seat] = local_value[seat];
                    scratch[SCRATCH_UNION + CHOICE_ENTRIES + lid * CHOICE_LOCAL + seat] =
                        bitcast_f32(local_index[seat]);
                }
            }
            workgroup_barrier();
            let mut distance = 1u32;
            for level in unroll(0u32, CHOICE_LEVELS, 1u32) {
                let slot = lid * 2u32 * distance;
                if slot + 2u32 * distance <= CHOICE_OWNERS {
                    let mut carried = scalar_array(max_identity(), CHOICE_LOCAL);
                    let mut carried_index = scalar_array(NO_VALUE, CHOICE_LOCAL);
                    for half in unroll(0u32, 2u32, 1u32) {
                        for at in unroll(0u32, CHOICE_LOCAL, 1u32) {
                            let seat = slot + half * distance;
                            let mut held_value = scratch[SCRATCH_UNION + seat * CHOICE_LOCAL + at];
                            let mut held_index = bitcast_u32(
                                scratch[SCRATCH_UNION + CHOICE_ENTRIES + seat * CHOICE_LOCAL + at],
                            );
                            for place in unroll(0u32, CHOICE_LOCAL, 1u32) {
                                if better(
                                    held_value,
                                    held_index,
                                    carried[place],
                                    carried_index[place],
                                ) {
                                    let swap_value = carried[place];
                                    let swap_index = carried_index[place];
                                    carried[place] = held_value;
                                    carried_index[place] = held_index;
                                    held_value = swap_value;
                                    held_index = swap_index;
                                }
                            }
                        }
                    }
                    for at in unroll(0u32, CHOICE_LOCAL, 1u32) {
                        scratch[SCRATCH_UNION + slot * CHOICE_LOCAL + at] = carried[at];
                        scratch[SCRATCH_UNION + CHOICE_ENTRIES + slot * CHOICE_LOCAL + at] =
                            bitcast_f32(carried_index[at]);
                    }
                }
                workgroup_barrier();
                distance = distance * 2u32;
            }
            let taken = min(keep - count, CHOICE_LOCAL);
            let mut last_value = previous_value;
            let mut last_index = previous_index;
            for at in unroll(0u32, CHOICE_LOCAL, 1u32) {
                if at < taken {
                    let index = bitcast_u32(scratch[SCRATCH_UNION + CHOICE_ENTRIES + at]);
                    if index != NO_VALUE {
                        last_value = scratch[SCRATCH_UNION + at];
                        last_index = index;
                    }
                }
            }
            if lid == 0u32 {
                for at in unroll(0u32, CHOICE_LOCAL, 1u32) {
                    if at < taken {
                        scratch[SCRATCH_SAMPLE + count + at] = scratch[SCRATCH_UNION + at];
                        scratch[SCRATCH_SAMPLE + CANDIDATES + count + at] =
                            scratch[SCRATCH_UNION + CHOICE_ENTRIES + at];
                    }
                }
            }
            previous_value = last_value;
            previous_index = last_index;
            count = count + taken;
            workgroup_barrier();
        }
        return count;
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

    fn row_mass(
        task: Task,
        lid: u32,
        source: Value,
        row: u32,
        columns: u32,
        first: u32,
        last: u32,
    ) -> f32 {
        let mut local_max = max_identity();
        let mut local_total = 0.0;
        for base in stride(lid, last - first, WORKGROUP_SIZE * 4u32) {
            let mut held = scalar_array(max_identity(), 4u32);
            for at in unroll(0u32, 4u32, 1u32) {
                let column = first + base + at * WORKGROUP_SIZE;
                held[at] = select(
                    max_identity(),
                    read_flat(task, source, row * columns + min(column, columns - 1u32)),
                    column < last,
                );
            }
            for at in unroll(0u32, 4u32, 1u32) {
                let column = first + base + at * WORKGROUP_SIZE;
                let value = held[at];
                if column < last {
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
            }
        }
        return sample_mass(lid, local_max, local_total);
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
        let total = row_mass(task, lid, source, row, columns, 0u32, columns);
        let row_max = scratch[0u32];
        workgroup_barrier();
        let count = rank_row(task, lid, source, row, columns, 0u32, columns, keep);
        workgroup_barrier();
        let mut chosen = NO_VALUE;
        if lid == 0u32 {
            let mut mass = 0.0;
            for candidate in stride(0u32, count, 1u32) {
                if bitcast_u32(scratch[SCRATCH_SAMPLE + CANDIDATES + candidate]) != NO_VALUE {
                    mass = mass + softmax_exp(scratch[SCRATCH_SAMPLE + candidate] - row_max);
                }
            }
            let tail = total - mass;
            let threshold = cumulative * total;
            let mut boundary = count;
            let mut running = 0.0;
            for candidate in stride(0u32, count, 1u32) {
                if bitcast_u32(scratch[SCRATCH_SAMPLE + CANDIDATES + candidate]) != NO_VALUE {
                    running = running + softmax_exp(scratch[SCRATCH_SAMPLE + candidate] - row_max);
                    if running + tail >= threshold && candidate + 1u32 < boundary {
                        boundary = candidate + 1u32;
                    }
                }
            }
            let mut best = max_identity();
            for candidate in stride(0u32, boundary, 1u32) {
                let value = scratch[SCRATCH_SAMPLE + candidate];
                let column = bitcast_u32(scratch[SCRATCH_SAMPLE + CANDIDATES + candidate]);
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

    fn sample_chunk(task: Task, lid: u32, source: Value, seed: u32, keep: u32, cumulative: f32) {
        let kept_values = values[task.out];
        let carried = values[task.extra];
        let columns = source.dims.w;
        let rows = kept_values.dims.z;
        let first = columns * task.section / task.splits;
        let last = columns * (task.section + 1u32) / task.splits;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let total = row_mass(task, lid, source, row, columns, first, last);
            let row_max = scratch[0u32];
            workgroup_barrier();
            let count = rank_row(task, lid, source, row, columns, first, last, keep);
            let base = (task.section * rows + row) * (2u32 * CANDIDATES + 4u32);
            if lid == 0u32 {
                publish(carried, base + 2u32 * CANDIDATES, f32(count));
                publish(carried, base + 2u32 * CANDIDATES + 1u32, row_max);
                publish(carried, base + 2u32 * CANDIDATES + 2u32, total);
            }
            for candidate in stride(lid, keep, WORKGROUP_SIZE) {
                let index = bitcast_u32(scratch[SCRATCH_SAMPLE + CANDIDATES + candidate]);
                let inside = candidate < count;
                let value = scratch[SCRATCH_SAMPLE + candidate];
                let column = min(index, columns - 1u32);
                publish(
                    kept_values,
                    (task.section * rows + row) * CANDIDATES + candidate,
                    select(max_identity(), value, inside),
                );
                publish(
                    carried,
                    base + 2u32 * candidate,
                    f32(select(NO_VALUE, index, inside)),
                );
                publish(
                    carried,
                    base + 2u32 * candidate + 1u32,
                    value + gumbel_noise(seed, coordinates(row * columns + column, source.dims)),
                );
            }
            workgroup_barrier();
        }
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
        let whole =
            declared > 0.0 && declared < f32(CANDIDATES + 1u32) && trunc(declared) == declared;
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
            strategy::WORKGROUP_ROW => {
                if task.splits > 1u32 {
                    sample_chunk(task, lid, source, seed, keep, cumulative);
                } else {
                    sample_rows(task, lid, source, seed, keep, cumulative);
                }
            }
            _ => refuse(kind::SAMPLE, refusal::GEOMETRY, task.geometry),
        }
    }

    fn merge_row(
        lid: u32,
        values_in: Value,
        carriers: Value,
        pair_stride: u32,
        weight_offset: u32,
        block: u32,
        rows: u32,
        sections: u32,
        keep: u32,
        row: u32,
    ) -> u32 {
        if lid != 0u32 {
            return 0u32;
        }
        for section in stride(0u32, sections, 1u32) {
            let base = (section * rows + row) * block;
            let count = u32(fetch(carriers, base + block - 4u32));
            let inside = count > 0u32;
            scratch[SCRATCH_MERGE_VALUE + section] = select(
                max_identity(),
                fetch(values_in, (section * rows + row) * CANDIDATES),
                inside,
            );
            scratch[SCRATCH_MERGE_WEIGHT + section] = select(
                max_identity(),
                fetch(carriers, base + weight_offset),
                inside,
            );
            scratch[SCRATCH_MERGE_INDEX + section] =
                bitcast_f32(select(NO_VALUE, u32(fetch(carriers, base)), inside));
            scratch[SCRATCH_MERGE_CURSOR + section] = bitcast_f32(0u32);
            scratch[SCRATCH_MERGE_COUNT + section] = bitcast_f32(count);
        }
        let mut count = 0u32;
        let mut slot = 0u32;
        loop {
            if slot >= keep {
                break;
            }
            let mut best_value = max_identity();
            let mut best_index = NO_VALUE;
            let mut best_section = NO_VALUE;
            let mut base_section = 0u32;
            loop {
                if base_section >= sections {
                    break;
                }
                let mut held_value = scalar_array(max_identity(), 4u32);
                let mut held_index = scalar_array(NO_VALUE, 4u32);
                let mut held_section = scalar_array(0u32, 4u32);
                for at in unroll(0u32, 4u32, 1u32) {
                    let probe = base_section + at;
                    if probe < sections {
                        held_index[at] = bitcast_u32(scratch[SCRATCH_MERGE_INDEX + probe]);
                        held_value[at] = scratch[SCRATCH_MERGE_VALUE + probe];
                        held_section[at] = probe;
                    }
                }
                for at in unroll(0u32, 4u32, 1u32) {
                    if held_index[at] != NO_VALUE
                        && better(held_value[at], held_index[at], best_value, best_index)
                    {
                        best_value = held_value[at];
                        best_index = held_index[at];
                        best_section = held_section[at];
                    }
                }
                base_section = base_section + 4u32;
            }
            if best_index == NO_VALUE {
                break;
            }
            scratch[SCRATCH_MERGE_OUT + slot] = best_value;
            scratch[SCRATCH_MERGE_OUT + CANDIDATES + slot] = bitcast_f32(best_index);
            scratch[SCRATCH_MERGE_OUT + 2u32 * CANDIDATES + slot] =
                scratch[SCRATCH_MERGE_WEIGHT + best_section];
            slot = slot + 1u32;
            count = count + 1u32;
            let position = bitcast_u32(scratch[SCRATCH_MERGE_CURSOR + best_section]);
            let limit = bitcast_u32(scratch[SCRATCH_MERGE_COUNT + best_section]);
            let held = min(position + 1u32, CANDIDATES - 1u32);
            let inside = position + 1u32 < limit;
            let base = (best_section * rows + row) * block;
            let head = (best_section * rows + row) * CANDIDATES;
            scratch[SCRATCH_MERGE_VALUE + best_section] =
                select(max_identity(), fetch(values_in, head + held), inside);
            scratch[SCRATCH_MERGE_WEIGHT + best_section] = select(
                max_identity(),
                fetch(carriers, base + held * pair_stride + weight_offset),
                inside,
            );
            scratch[SCRATCH_MERGE_INDEX + best_section] = bitcast_f32(select(
                NO_VALUE,
                u32(fetch(carriers, base + held * pair_stride)),
                inside,
            ));
            scratch[SCRATCH_MERGE_CURSOR + best_section] = bitcast_f32(position + 1u32);
        }
        return count;
    }

    fn run_top_k_fold(task: Task, lid: u32) {
        let values_in = values[task.a];
        let carried = values[task.b];
        let out_values = values[task.out];
        let out_indices = values[task.extra];
        let keep = task.keep;
        let sections = task.splits;
        let rows = out_values.dims.z;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let count = merge_row(
                lid,
                values_in,
                carried,
                1u32,
                0u32,
                CANDIDATES + 4u32,
                rows,
                sections,
                keep,
                row,
            );
            if lid == 0u32 {
                for slot in stride(0u32, keep, 1u32) {
                    let inside = slot < count;
                    publish(
                        out_values,
                        row * keep + slot,
                        select(max_identity(), scratch[SCRATCH_MERGE_OUT + slot], inside),
                    );
                    publish(
                        out_indices,
                        row * keep + slot,
                        select(
                            -1.0,
                            f32(bitcast_u32(scratch[SCRATCH_MERGE_OUT + CANDIDATES + slot])),
                            inside,
                        ),
                    );
                }
            }
        }
    }

    fn run_sample_fold(task: Task, lid: u32) {
        let values_in = values[task.a];
        let seed = bitcast_u32(fetch(values[task.b], 0u32));
        let declared = fetch(values[task.c], 0u32);
        let cumulative = fetch(values[task.d], 0u32);
        let candidates = values[task.e];
        let output = values[task.out];
        let whole =
            declared > 0.0 && declared < f32(CANDIDATES + 1u32) && trunc(declared) == declared;
        if !whole {
            refuse(kind::SAMPLE_FOLD, refusal::SAMPLE, 0u32);
            return;
        }
        if !(cumulative > 0.0) || !(cumulative <= 1.0) {
            refuse(kind::SAMPLE_FOLD, refusal::SAMPLE, 1u32);
            return;
        }
        let keep = u32(declared);
        let sections = task.splits;
        let rows = output.dims.z;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let count = merge_row(
                lid,
                values_in,
                candidates,
                2u32,
                1u32,
                2u32 * CANDIDATES + 4u32,
                rows,
                sections,
                keep,
                row,
            );
            if lid == 0u32 {
                let mut row_max = max_identity();
                for section in stride(0u32, sections, 1u32) {
                    row_max = max(
                        row_max,
                        fetch(
                            candidates,
                            (section * rows + row) * (2u32 * CANDIDATES + 4u32)
                                + 2u32 * CANDIDATES
                                + 1u32,
                        ),
                    );
                }
                let mut total = 0.0;
                for section in stride(0u32, sections, 1u32) {
                    let base =
                        (section * rows + row) * (2u32 * CANDIDATES + 4u32) + 2u32 * CANDIDATES;
                    let section_max = fetch(candidates, base + 1u32);
                    let section_total = fetch(candidates, base + 2u32);
                    total = total + section_total * softmax_exp(section_max - row_max);
                }
                let mut mass = 0.0;
                for candidate in stride(0u32, count, 1u32) {
                    mass = mass + softmax_exp(scratch[SCRATCH_MERGE_OUT + candidate] - row_max);
                }
                let tail = total - mass;
                let threshold = cumulative * total;
                let mut boundary = count;
                let mut running = 0.0;
                for candidate in stride(0u32, count, 1u32) {
                    running =
                        running + softmax_exp(scratch[SCRATCH_MERGE_OUT + candidate] - row_max);
                    if running + tail >= threshold && candidate + 1u32 < boundary {
                        boundary = candidate + 1u32;
                    }
                }
                let mut best = max_identity();
                let mut chosen = NO_VALUE;
                for candidate in stride(0u32, boundary, 1u32) {
                    let weight = scratch[SCRATCH_MERGE_OUT + 2u32 * CANDIDATES + candidate];
                    let column = bitcast_u32(scratch[SCRATCH_MERGE_OUT + CANDIDATES + candidate]);
                    if chosen == NO_VALUE || choice_precedes(best, chosen, weight, column) {
                        best = weight;
                        chosen = column;
                    }
                }
                if chosen == NO_VALUE {
                    chosen = 0u32;
                }
                publish(
                    output,
                    row,
                    chained(task, coordinates(row, output.dims), f32(chosen)),
                );
            }
        }
    }

    fn top_k_row(task: Task, lid: u32, source: Value, row: u32, columns: u32, keep: u32) {
        let kept_values = values[task.out];
        let kept_indices = values[task.extra];
        rank_row(task, lid, source, row, columns, 0u32, columns, keep);
        for candidate in stride(lid, keep, WORKGROUP_SIZE) {
            publish(
                kept_values,
                row * keep + candidate,
                scratch[SCRATCH_SAMPLE + candidate],
            );
            publish(
                kept_indices,
                row * keep + candidate,
                f32(bitcast_u32(
                    scratch[SCRATCH_SAMPLE + CANDIDATES + candidate],
                )),
            );
        }
    }

    fn top_k_chunk(task: Task, lid: u32) {
        let source = values[task.a];
        let kept_values = values[task.out];
        let carried = values[task.extra];
        let keep = task.keep;
        let columns = source.dims.w;
        let rows = kept_values.dims.z;
        let first = columns * task.section / task.splits;
        let last = columns * (task.section + 1u32) / task.splits;
        for row in stride(task.first, task.first + task.count, 1u32) {
            let count = rank_row(task, lid, source, row, columns, first, last, keep);
            let base = (task.section * rows + row) * (CANDIDATES + 4u32);
            if lid == 0u32 {
                publish(carried, base + CANDIDATES, f32(count));
            }
            for candidate in stride(lid, keep, WORKGROUP_SIZE) {
                publish(
                    kept_values,
                    (task.section * rows + row) * CANDIDATES + candidate,
                    scratch[SCRATCH_SAMPLE + candidate],
                );
                publish(
                    carried,
                    base + candidate,
                    f32(bitcast_u32(
                        scratch[SCRATCH_SAMPLE + CANDIDATES + candidate],
                    )),
                );
            }
            workgroup_barrier();
        }
    }

    fn run_top_k(task: Task, lid: u32) {
        let source = values[task.a];
        let keep = task.keep;
        let columns = source.dims.w;
        match task.geometry {
            strategy::WORKGROUP_ROW => {
                if task.splits > 1u32 {
                    top_k_chunk(task, lid);
                } else {
                    for row in stride(task.first, task.first + task.count, 1u32) {
                        top_k_row(task, lid, source, row, columns, keep);
                        workgroup_barrier();
                    }
                }
            }
            _ => refuse(kind::TOP_K, refusal::GEOMETRY, task.geometry),
        }
    }
}
