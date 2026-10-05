use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
    compiler.select("workgroup_sum_tree", "workgroup_sum");
    compiler.select("workgroup_max_tree", "workgroup_max");
}

#[neura_compiler::module]
mod device {
    fn workgroup_sum_tree(lid: u32, start: f32) -> f32 {
        scratch[lid] = start;
        workgroup_barrier();
        let mut stride = WORKGROUP_SIZE / 2u32;
        loop {
            if stride == 0u32 {
                break;
            }
            if lid < stride {
                scratch[lid] = scratch[lid] + scratch[lid + stride];
            }
            workgroup_barrier();
            stride = stride / 2u32;
        }
        let total = scratch[0u32];
        workgroup_barrier();
        return total;
    }

    fn workgroup_max_tree(lid: u32, start: f32) -> f32 {
        scratch[lid] = start;
        workgroup_barrier();
        let mut stride = WORKGROUP_SIZE / 2u32;
        loop {
            if stride == 0u32 {
                break;
            }
            if lid < stride {
                scratch[lid] = max(scratch[lid], scratch[lid + stride]);
            }
            workgroup_barrier();
            stride = stride / 2u32;
        }
        let total = scratch[0u32];
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

    fn unit_axis(axis: u32) -> uvec4 {
        return uvec4(
            select(0u32, 1u32, axis == 0u32),
            select(0u32, 1u32, axis == 1u32),
            select(0u32, 1u32, axis == 2u32),
            select(0u32, 1u32, axis == 3u32),
        );
    }

    fn sum_axis_chunk(task: Task, lid: u32, source: Value, target: Value, axis: u32) {
        let step = unit_axis(axis);
        let rate = component(source.strides, axis);
        let walked = component(source.dims, axis);
        let pieces = component(target.dims, axis);
        let shared = walked / pieces;
        let rest = walked % pieces;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, target.dims);
            let chunk = component(at, axis);
            let base = at - step * chunk;
            let first = chunk * shared + min(chunk, rest);
            let last = (chunk + 1u32) * shared + min(chunk + 1u32, rest);
            let start = read_address(base, source.strides);
            let mut total = 0.0;
            if task.prelude_steps == 0u32 {
                for fold in stride(first, last, 1u32) {
                    total = total + fetch(source, start + fold * rate);
                }
            } else {
                for fold in stride(first, last, 1u32) {
                    total = total + read_frame(task, source, base + step * fold);
                }
            }
            publish(target, index, chained(task, base, total));
        }
    }

    fn sum_axis_rows(task: Task, lid: u32, source: Value, target: Value, axis: u32) {
        let step = unit_axis(axis);
        let rate = component(source.strides, axis);
        let walked = component(source.dims, axis);
        let pieces = component(target.dims, axis);
        let shared = walked / pieces;
        let rest = walked % pieces;
        for index in stride(task.first, task.first + task.count, 1u32) {
            let at = coordinates(index, target.dims);
            let chunk = component(at, axis);
            let base = at - step * chunk;
            let first = chunk * shared + min(chunk, rest);
            let last = (chunk + 1u32) * shared + min(chunk + 1u32, rest);
            let start = read_address(base, source.strides);
            let mut local = 0.0;
            if task.prelude_steps == 0u32 {
                for fold in stride(first + lid, last, WORKGROUP_SIZE) {
                    local = local + fetch(source, start + fold * rate);
                }
            } else {
                for fold in stride(first + lid, last, WORKGROUP_SIZE) {
                    local = local + read_frame(task, source, base + step * fold);
                }
            }
            let total = workgroup_sum(lid, local);
            if lid == 0u32 {
                publish(target, index, chained(task, base, total));
            }
            workgroup_barrier();
        }
    }

    fn run_segment_sum(task: Task, lid: u32) {
        let source = values[task.a];
        let partials = values[task.out];
        if task.plane >= partials.dims.x * partials.dims.y {
            return;
        }
        let head = task.plane / partials.dims.y;
        let batch = task.plane % partials.dims.y;
        let mut start = 0u32;
        if task.count > 0u32 {
            let offsets = values[task.segment];
            start = whole_index(
                fetch(offsets, task.plane),
                source.dims.z + 1u32,
                kind::SEGMENT_SUM,
                refusal::INDEX,
            );
        }
        let base = read_address(uvec4(head, batch, task.index, 0u32), partials.strides);
        for column in stride(lid, partials.dims.w, WORKGROUP_SIZE) {
            let mut total = 0.0;
            for row in stride(0u32, task.count, 1u32) {
                total = total
                    + read_frame(
                        task,
                        source,
                        uvec4(0u32, 0u32, start + task.first + row, column),
                    );
            }
            publish(partials, base + column * partials.strides.w, total);
        }
    }

    fn run_sum_axis(task: Task, lid: u32) {
        let source = values[task.a];
        let target = values[task.out];
        match task.geometry {
            strategy::THREAD_ELEMENT => sum_axis_chunk(task, lid, source, target, task.slot),
            strategy::WORKGROUP_ROW => sum_axis_rows(task, lid, source, target, task.slot),
            _ => refuse(kind::SUM_AXIS, refusal::GEOMETRY, task.geometry),
        }
    }
}
