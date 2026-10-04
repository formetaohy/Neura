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

    fn slot_of(packed: u32, axis: u32) -> u32 {
        return (packed >> (axis * 8u32)) & 0xffu32;
    }

    fn walked_extent(slot: u32, bound: u32) -> u32 {
        let free = slot != NO_SLOT;
        return select(bound, extents[select(0u32, slot, free)], free);
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
        let record = measures[measure];
        return base_span(record.kind, record.value, record.rows, record.columns);
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
            let record = measures[task.measure];
            let dims = value_dims(values[record.value]);
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
        let within = walked_boundary(total, task.index, task.group);
        if task.split == split::PLANE {
            return task.plane * total + within;
        }
        return within;
    }

    fn span_count(task: Task) -> u32 {
        if task.split == split::RANGE {
            return task.count;
        }
        if task.split == split::SEGMENT {
            return 1u32;
        }
        let total = walked_total(task);
        return walked_boundary(total, task.index + 1u32, task.group)
            - walked_boundary(total, task.index, task.group);
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
        let tiles = measures[task.measure];
        let column_blocks = ceil_div(value_dims(values[tiles.value]).w, tiles.columns);
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
        let row = (task.index / column_blocks) * tiles.rows;
        if row < end - start {
            tasks[id].keys = end - start - row;
            tasks[id].count = 1u32;
        }
    }

    fn patch_ragged(task: Task, id: u32, segments: u32, live: u32) {
        tasks[id].first = 0u32;
        tasks[id].count = 0u32;
        if task.plane >= segments {
            return;
        }
        tasks[id].count = segment_keys(task, live + 1u32);
    }

    fn patch_extents(patch: u32, lid: u32) {
        let record = patches[patch];
        let author = values[record.count];
        let declared = extents[patch_list[record.slots]];
        storage_barrier();
        let live = whole_index(
            fetch(author, 0u32),
            declared + 1u32,
            refusal::TENSOR,
            refusal::EXTENT,
        );
        for step in stride(lid, record.slots_count, WORKGROUP_SIZE) {
            let slot = patch_list[record.slots + step];
            if live > extents[slot] {
                refuse(refusal::TENSOR, refusal::EXTENT, 0u32);
            }
            extents[slot] = live;
        }
        storage_barrier();
        for step in stride(lid, record.values_count, WORKGROUP_SIZE) {
            let id = patch_list[record.values + step];
            let value = values[id];
            values[id].dims = value_dims(value);
            values[id].strides = value_strides(value);
        }
        storage_barrier();
        let mut segments = 0u32;
        if record.segment != NO_VALUE {
            segments = live_segments(values[record.segment], live);
        }
        for step in stride(lid, record.tasks_count, WORKGROUP_SIZE) {
            let id = patch_list[record.tasks + step];
            let task = tasks[id];
            if task.split == split::SEGMENT {
                if record.segment != NO_VALUE && task.segment == record.segment {
                    patch_segment(task, id, segments, live);
                }
            } else if task.split == split::RAGGED {
                if record.segment != NO_VALUE && task.segment == record.segment {
                    patch_ragged(task, id, segments, live);
                }
            } else {
                let count = span_count(task);
                tasks[id].first = span_first(task);
                tasks[id].count = count;
                if record.segment != NO_VALUE && task.segment == record.segment && count > 0u32 {
                    tasks[id].keys = segment_keys(task, live + 1u32);
                }
            }
        }
    }

    fn patch_none(patch: u32, lid: u32) {}
}
