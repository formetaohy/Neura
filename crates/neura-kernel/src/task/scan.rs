use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn elements(value: Value) -> u32 {
        return value.dims.x * value.dims.y * value.dims.z * value.dims.w;
    }

    fn scan_sum(lid: u32, start: f32) -> f32 {
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

    fn run_prefix_chunk(task: Task, lid: u32) {
        let lengths = values[task.a];
        let partials = values[task.out];
        let mut local = 0.0;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            local = local + fetch(lengths, index);
        }
        let total = scan_sum(lid, local);
        if lid == 0u32 {
            publish(partials, task.slot, total);
        }
    }

    fn run_prefix_scan(task: Task, lid: u32) {
        let lengths = values[task.a];
        let partials = values[task.b];
        let offsets = values[task.out];
        let mut base = 0.0;
        for index in stride(0u32, task.slot, 1u32) {
            base = base + fetch(partials, index);
        }
        let sub = (task.count + WORKGROUP_SIZE - 1u32) / WORKGROUP_SIZE;
        let first = task.first + lid * sub;
        let last = min(first + sub, task.first + task.count);
        let mut mine = 0.0;
        for index in stride(first, last, 1u32) {
            mine = mine + fetch(lengths, index);
        }
        scratch[lid] = mine;
        workgroup_barrier();
        let mut running = base;
        for index in stride(0u32, lid, 1u32) {
            running = running + scratch[index];
        }
        for index in stride(first, last, 1u32) {
            let length = fetch(lengths, index);
            publish(offsets, index, running);
            running = running + length;
        }
    }

    fn run_prefix_close(task: Task, lid: u32) {
        let offsets = values[task.a];
        let lengths = values[task.b];
        if lid == 0u32 {
            let planes = elements(lengths);
            let mut total = 0.0;
            if planes > 0u32 {
                total = fetch(offsets, planes - 1u32) + fetch(lengths, planes - 1u32);
            }
            if elements(offsets) > planes {
                publish(offsets, planes, total);
            }
            publish(values[task.out], 0u32, total);
        }
    }
}
