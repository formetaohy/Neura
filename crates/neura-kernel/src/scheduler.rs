use neura_compiler::Compiler;

pub(crate) use device::ENTRY;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn remaining(wave: u32) -> u32 {
        return progress::COUNTERS + progress::WAVE_STRIDE * wave;
    }

    fn total(wave: u32) -> u32 {
        return remaining(wave) + 1u32;
    }

    fn gate(wave: u32) {
        loop {
            if atomic_add(&progress[progress::FRONTIER], 0u32) >= wave {
                break;
            }
        }
    }

    fn finish(wave: u32, lid: u32) {
        let mut rest = 0u32;
        if lid == 0u32 {
            rest = atomic_sub(&progress[remaining(wave)], 1u32);
        }
        storage_barrier();
        if rest == 1u32 {
            if lid == 0u32 {
                atomic_store(&progress[progress::FRONTIER], wave + 1u32);
                atomic_add(&progress[remaining(wave)], progress[total(wave)]);
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
            claim[0u32] = progress[progress::SEGMENTS];
        }
        let total = workgroup_uniform_load(&claim[0u32]);
        loop {
            if lid == 0u32 {
                claim[1u32] = atomic_add(&progress[progress::CURSOR], 1u32);
            }
            let ticket = workgroup_uniform_load(&claim[1u32]);
            if ticket >= total {
                break;
            }
            let segment = segments[ticket];
            for index in stride(segment.first, segment.first + segment.count, 1u32) {
                if lid == 0u32 {
                    gate(segment.wave);
                }
                storage_barrier();
                let task = tasks[index];
                run_task(task, lid);
                if task.patch != NO_VALUE {
                    patch(task.patch, lid);
                }
                storage_barrier();
                finish(segment.wave, lid);
            }
        }
    }
}
