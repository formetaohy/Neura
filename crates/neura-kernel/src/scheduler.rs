use neura_compiler::Compiler;

pub(crate) use device::ENTRY;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn remaining(segment: u32) -> u32 {
        return control::COUNTERS + segment;
    }

    fn wave_of(waves: u32, wave: u32) -> u32 {
        return waves + wave;
    }

    fn finish(
        segment_id: u32,
        segment: Segment,
        lid: u32,
        first: u32,
        last: u32,
        queue: u32,
        generation: u32,
        waves: u32,
        wave_table: u32,
        wave_count: u32,
    ) {
        for index in stride(lid, segment.successors_count, WORKGROUP_SIZE) {
            let next = successors[segment.successors + index];
            if next >= first && next < last {
                if atomic_sub(&state[remaining(next)], 1u32) == 1u32 {
                    let slot = atomic_add(&state[control::TAIL], 1u32);
                    atomic_store(
                        &state[queue + slot],
                        (generation << control::ENTRY_SHIFT) | next,
                    );
                }
            }
        }
        if lid == 0u32 {
            atomic_store(&state[remaining(segment_id)], segment.predecessors);
            claim[0u32] = atomic_sub(&state[wave_of(waves, segment.wave)], 1u32);
        }
        let reached = workgroup_uniform_load(&claim[0u32]);
        if reached != 1u32 {
            return;
        }
        if lid == 0u32 {
            let mut opened = NO_VALUE;
            loop {
                if atomic_add(&state[control::PREFIX_LOCK], 1u32) == 0u32 {
                    break;
                }
            }
            let prefix = atomic_add(&state[control::PREFIX], 0u32);
            let mut next = prefix;
            loop {
                if next >= wave_count {
                    break;
                }
                if atomic_add(&state[wave_of(waves, next)], 0u32) != 0u32 {
                    break;
                }
                next = next + 1u32;
            }
            if next != prefix {
                atomic_store(&state[control::PREFIX], next);
                if next < wave_count {
                    let held = wave_table + 1u32 + next * 4u32;
                    if successors[held + 2u32] != 0u32 {
                        opened = next;
                    }
                }
            }
            atomic_store(&state[control::PREFIX_LOCK], 0u32);
            claim[1u32] = opened;
        }
        let opened = workgroup_uniform_load(&claim[1u32]);
        if opened != NO_VALUE {
            let held = wave_table + 1u32 + opened * 4u32;
            let wave_first = successors[held];
            let wave_segments = successors[held + 1u32];
            if wave_first >= first && wave_first + wave_segments <= last {
                for offset in stride(lid, wave_segments, WORKGROUP_SIZE) {
                    let segment = wave_first + offset;
                    let slot = atomic_add(&state[control::TAIL], 1u32);
                    atomic_store(
                        &state[queue + slot],
                        (generation << control::ENTRY_SHIFT) | segment,
                    );
                }
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
            claim[0u32] = state[control::COUNT];
            claim[1u32] = state[control::FIRST];
            claim[2u32] = state[control::QUEUE];
            claim[3u32] = state[control::GENERATION];
            claim[4u32] = state[control::WAVES];
            claim[5u32] = state[control::WAVE_TABLE];
            claim[6u32] = successors[claim[5u32]];
        }
        let count = workgroup_uniform_load(&claim[0u32]);
        let first = workgroup_uniform_load(&claim[1u32]);
        let queue = workgroup_uniform_load(&claim[2u32]);
        let generation = workgroup_uniform_load(&claim[3u32]);
        let waves = workgroup_uniform_load(&claim[4u32]);
        let wave_table = workgroup_uniform_load(&claim[5u32]);
        let wave_count = workgroup_uniform_load(&claim[6u32]);
        let last = first + count;
        loop {
            if lid == 0u32 {
                claim[6u32] = atomic_add(&state[control::CURSOR], 1u32);
            }
            let ticket = workgroup_uniform_load(&claim[6u32]);
            if ticket >= count {
                break;
            }
            let slot = queue + ticket;
            loop {
                if lid == 0u32 {
                    claim[7u32] = atomic_add(&state[slot], 0u32);
                }
                let held = workgroup_uniform_load(&claim[7u32]);
                if (held >> control::ENTRY_SHIFT) == generation {
                    break;
                }
            }
            storage_barrier();
            let entry = workgroup_uniform_load(&claim[7u32]);
            let segment_id = entry & control::ENTRY_MASK;
            let segment = segments[segment_id];
            for index in stride(segment.first, segment.first + segment.count, 1u32) {
                let task = tasks[index];
                run_task(task, lid);
                if task.patch != NO_VALUE {
                    workgroup_barrier();
                    patch(task.patch, lid);
                }
                workgroup_barrier();
                storage_barrier();
            }
            finish(
                segment_id, segment, lid, first, last, queue, generation, waves, wave_table,
                wave_count,
            );
        }
    }
}
