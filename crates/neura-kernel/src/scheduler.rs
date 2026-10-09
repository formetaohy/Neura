use neura_compiler::Compiler;

pub(crate) use device::ENTRY;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn remaining(wave: u32) -> u32 {
        return control::COUNTERS + control::WAVE_STRIDE * wave;
    }

    fn total(wave: u32) -> u32 {
        return remaining(wave) + 1u32;
    }

    fn gate(wave: u32) {
        loop {
            if atomic_add(&state[control::FRONTIER], 0u32) >= wave {
                break;
            }
        }
    }

    fn finish(wave: u32, lid: u32) {
        let mut rest = 0u32;
        if lid == 0u32 {
            rest = atomic_sub(&state[remaining(wave)], 1u32);
        }
        storage_barrier();
        if rest == 1u32 {
            if lid == 0u32 {
                atomic_store(&state[control::FRONTIER], wave + 1u32);
                atomic_add(&state[remaining(wave)], state[total(wave)]);
            }
        }
    }

    fn run_task(task: Task, lid: u32) {
        match task.kind {
            _ => refuse(task.kind, refusal::TASK, 0u32),
        }
    }

    #[neura_compiler::kernel]
    fn main(lid: u32) {
        if lid == 0u32 {
            claim[0u32] = state[control::SEGMENTS];
            claim[2u32] = state[control::FIRST_TASK];
            claim[3u32] = state[control::LAST_TASK];
        }
        let total = workgroup_uniform_load(&claim[0u32]);
        let from = workgroup_uniform_load(&claim[2u32]);
        let to = workgroup_uniform_load(&claim[3u32]);
        loop {
            if lid == 0u32 {
                claim[1u32] = atomic_add(&state[control::CURSOR], 1u32);
            }
            let ticket = workgroup_uniform_load(&claim[1u32]);
            if ticket >= total {
                break;
            }
            let segment = segments[ticket];
            let first = max(segment.first, from);
            let last = min(segment.first + segment.count, to);
            for index in stride(first, last, 1u32) {
                if lid == 0u32 {
                    gate(segment.wave);
                }
                storage_barrier();
                let task = tasks[index];
                run_task(task, lid);
                if task.patch != NO_VALUE {
                    storage_barrier();
                    patch(task.patch, lid);
                }
                storage_barrier();
                finish(segment.wave, lid);
            }
        }
    }
}
