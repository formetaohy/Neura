use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler, carried: bool) {
    device::define(compiler);
    if carried {
        compiler.select("patch_extents", "patch");
    } else {
        compiler.select("patch_none", "patch");
    }
}

#[neura_compiler::module]
mod device {
    fn ceil_div(total: u32, divisor: u32) -> u32 {
        return (total + divisor - 1u32) / divisor;
    }

    fn elements_per_word(storage: u32) -> u32 {
        if storage == element::HALF || storage == element::BFLOAT16 {
            return 2u32;
        }
        if storage == element::INT8 || storage == element::FP8_E4M3 || storage == element::FP8_E5M2
        {
            return 4u32;
        }
        if storage == element::INT4 || storage == element::FP4_E2M1 {
            return 8u32;
        }
        return 1u32;
    }

    fn walked_extent(slot: u32, bound: u32) -> u32 {
        let free = slot != NO_SLOT;
        return select(
            bound,
            tables[EXTENTS_FIRST + select(0u32, slot, free)],
            free,
        );
    }

    fn measure_word(measure: u32, field: u32) -> u32 {
        return tables[MEASURES_FIRST + measure * MEASURE_WORDS + field];
    }

    fn patch_word(patch: u32, field: u32) -> u32 {
        return tables[PATCHES_FIRST + patch * PATCH_WORDS + field];
    }

    fn patch_list_word(index: u32) -> u32 {
        return tables[PATCH_LIST_FIRST + index];
    }

    fn bounds_of(value: Value) -> uvec4 {
        let bounds = value.bounds;
        return uvec4(
            walked_extent(slot_of(value.free, 0u32), bounds.x),
            walked_extent(slot_of(value.free, 1u32), bounds.y),
            walked_extent(slot_of(value.free, 2u32), bounds.z),
            walked_extent(slot_of(value.free, 3u32), bounds.w),
        );
    }

    fn dense_strides(dims: uvec4) -> uvec4 {
        let x = select(0u32, dims.y * dims.z * dims.w, dims.x != 1u32);
        let y = select(0u32, dims.z * dims.w, dims.y != 1u32);
        let z = select(0u32, dims.w, dims.z != 1u32);
        let w = select(0u32, 1u32, dims.w != 1u32);
        return uvec4(x, y, z, w);
    }

    fn base_elements(value: Value) -> u32 {
        let dims = bounds_of(value);
        return dims.x * dims.y * dims.z * dims.w;
    }

    fn base_span(kind: u32, value: u32, rows: u32, columns: u32) -> u32 {
        let tensor = values[value];
        if kind == measure::ELEMENTS {
            return base_elements(tensor);
        }
        if kind == measure::ROWS {
            let dims = bounds_of(tensor);
            return dims.x * dims.y * dims.z;
        }
        if kind == measure::TOKENS {
            return bounds_of(tensor).z;
        }
        if kind == measure::WORDS {
            return ceil_div(base_elements(tensor), elements_per_word(tensor.element));
        }
        if kind == measure::TILES {
            let dims = bounds_of(tensor);
            return dims.x * dims.y * ceil_div(dims.z, rows) * ceil_div(dims.w, columns);
        }
        refuse(refusal::TENSOR, refusal::GEOMETRY, kind);
        return 0u32;
    }

    fn measure_total(measure: u32) -> u32 {
        return base_span(
            measure_word(measure, MEASURE_KIND),
            measure_word(measure, MEASURE_VALUE),
            measure_word(measure, MEASURE_ROWS),
            measure_word(measure, MEASURE_COLUMNS),
        );
    }

    fn value_dims(value: Value) -> uvec4 {
        return bounds_of(value);
    }

    fn value_strides(value: Value) -> uvec4 {
        if slot_of(value.source, 0u32) == NO_SLOT {
            return dense_strides(value_dims(value));
        }
        let owner = dense_strides(value_dims(values[value.storage]));
        return uvec4(
            component(owner, slot_of(value.source, 0u32)),
            component(owner, slot_of(value.source, 1u32)),
            component(owner, slot_of(value.source, 2u32)),
            component(owner, slot_of(value.source, 3u32)),
        );
    }

    fn walked_total(task: Task) -> u32 {
        if task.split == split::PLANE {
            let dims = value_dims(values[measure_word(task.measure, MEASURE_VALUE)]);
            if task.plane >= dims.x * dims.y || task.plane >= task.planes {
                return 0u32;
            }
            return dims.z;
        }
        return measure_total(task.measure);
    }

    fn walked_boundary(total: u32, piece: u32, group: u32) -> u32 {
        let shared = total / group;
        let rest = total % group;
        return piece * shared + min(piece, rest);
    }

    fn span_first(task: Task) -> u32 {
        if task.split == split::RANGE {
            return task.first;
        }
        if task.split == split::SEGMENT {
            return task.index;
        }
        let total = walked_total(task);
        if task.split == split::PLANE && task.planes > 1u32 {
            return task.plane * total + walked_boundary(total, task.index, task.group);
        }
        return min(task.planned_first, total);
    }

    fn span_count(task: Task) -> u32 {
        if task.split == split::RANGE {
            return task.count;
        }
        if task.split == split::SEGMENT {
            return 1u32;
        }
        let total = walked_total(task);
        if task.split == split::PLANE && task.planes > 1u32 {
            return walked_boundary(total, task.index + 1u32, task.group)
                - walked_boundary(total, task.index, task.group);
        }
        let first = min(task.planned_first, total);
        return min(task.planned_count, total - first);
    }

    fn segment_keys(task: Task, bound: u32) -> u32 {
        let offsets = values[task.segment];
        let start = whole_index(
            fetch(offsets, task.plane),
            bound,
            refusal::TENSOR,
            refusal::EXTENT,
        );
        let end = whole_index(
            fetch(offsets, task.plane + 1u32),
            bound,
            refusal::TENSOR,
            refusal::EXTENT,
        );
        if end >= start {
            return end - start;
        }
        refuse(refusal::TENSOR, refusal::EXTENT, 0u32);
        return 0u32;
    }
    fn live_segments(offsets: Value, total: u32) -> u32 {
        let mut segments = 0u32;
        loop {
            if fetch(offsets, segments) >= f32(total) {
                break;
            }
            segments = segments + 1u32;
        }
        return segments;
    }

    fn patch_segment(task: Task, id: u32, segments: u32, live: u32) {
        tasks[id].first = task.index;
        tasks[id].count = 0u32;
        if task.plane >= segments {
            return;
        }
        let measured = measure_word(task.measure, MEASURE_VALUE);
        let column_blocks = ceil_div(
            value_dims(values[measured]).w,
            measure_word(task.measure, MEASURE_COLUMNS),
        );
        let offsets = values[task.segment];
        let start = whole_index(
            fetch(offsets, task.plane),
            live + 1u32,
            refusal::TENSOR,
            refusal::EXTENT,
        );
        let end = whole_index(
            fetch(offsets, task.plane + 1u32),
            live + 1u32,
            refusal::TENSOR,
            refusal::EXTENT,
        );
        let row = (task.index / column_blocks) * measure_word(task.measure, MEASURE_ROWS);
        if row < end - start {
            tasks[id].keys = end - start - row;
            tasks[id].count = 1u32;
        }
    }

    fn patch_ragged(
        task: Task,
        id: u32,
        grid: Value,
        segments: u32,
        live: u32,
        owns_keys: bool,
        owns_tokens: bool,
    ) {
        tasks[id].first = 0u32;
        tasks[id].count = 0u32;
        if owns_keys {
            tasks[id].keys = 0u32;
        }
        if owns_tokens {
            tasks[id].tokens = 0u32;
        }
        if task.plane >= segments {
            return;
        }
        let rows = axis_rows(grid, task.plane, live + 1u32, refusal::TENSOR);
        let first = walked_boundary(rows, task.index, task.group);
        let end = walked_boundary(rows, task.index + 1u32, task.group);
        tasks[id].first = first;
        tasks[id].count = end - first;
        if owns_keys {
            tasks[id].keys = rows;
        }
        if owns_tokens {
            tasks[id].tokens = rows;
        }
    }

    fn patch_extents(patch: u32, lid: u32) {
        let count = patch_word(patch, PATCH_COUNT);
        let slots_first = patch_word(patch, PATCH_SLOTS);
        let slots_count = patch_word(patch, PATCH_SLOTS_COUNT);
        let segment = patch_word(patch, PATCH_SEGMENT);
        let values_first = patch_word(patch, PATCH_VALUES);
        let values_count = patch_word(patch, PATCH_VALUES_COUNT);
        let tasks_first = patch_word(patch, PATCH_TASKS);
        let tasks_count = patch_word(patch, PATCH_TASKS_COUNT);
        let author = values[count];
        let declared = tables[EXTENTS_FIRST + patch_list_word(slots_first)];
        storage_barrier();
        let live = whole_index(
            fetch(author, 0u32),
            declared + 1u32,
            refusal::TENSOR,
            refusal::EXTENT,
        );
        for step in stride(lid, slots_count, WORKGROUP_SIZE) {
            let slot = patch_list_word(slots_first + step);
            if live > tables[EXTENTS_FIRST + slot] {
                refuse(refusal::TENSOR, refusal::EXTENT, 0u32);
            }
            tables[EXTENTS_FIRST + slot] = live;
        }
        storage_barrier();
        for step in stride(lid, values_count, WORKGROUP_SIZE) {
            let id = patch_list_word(values_first + step);
            let value = values[id];
            values[id].dims = value_dims(value);
            values[id].strides = value_strides(value);
        }
        storage_barrier();
        let mut segments = 0u32;
        if segment != NO_VALUE {
            segments = live_segments(values[segment], live);
        }
        for step in stride(lid, tasks_count, WORKGROUP_SIZE) {
            let id = patch_list_word(tasks_first + step);
            let task = tasks[id];
            let owns_keys = segment != NO_VALUE && task.segment == segment;
            let owns_tokens = segment != NO_VALUE && task.queries == segment;
            let owns_grid = segment != NO_VALUE && task.grid == segment;
            if task.split == split::SEGMENT {
                if owns_keys {
                    patch_segment(task, id, segments, live);
                }
            } else if task.split == split::RAGGED {
                if owns_grid {
                    patch_ragged(
                        task,
                        id,
                        values[task.grid],
                        segments,
                        live,
                        task.grid == task.segment,
                        owns_tokens,
                    );
                } else if owns_tokens {
                    if task.plane < segments {
                        tasks[id].tokens = axis_rows(
                            values[task.queries],
                            task.plane,
                            live + 1u32,
                            refusal::TENSOR,
                        );
                    } else {
                        tasks[id].tokens = 0u32;
                    }
                }
                if !owns_grid && owns_keys {
                    if task.plane < segments {
                        tasks[id].keys = segment_keys(task, live + 1u32);
                    } else {
                        tasks[id].keys = 0u32;
                    }
                }
            } else {
                let count = span_count(task);
                tasks[id].first = span_first(task);
                tasks[id].count = count;
                if owns_keys {
                    if count > 0u32 && task.plane < segments {
                        tasks[id].keys = segment_keys(task, live + 1u32);
                    } else {
                        tasks[id].keys = 0u32;
                    }
                }
            }
        }
    }

    fn patch_none(patch: u32, lid: u32) {}
}
