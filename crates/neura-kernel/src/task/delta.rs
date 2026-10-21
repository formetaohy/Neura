use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    compiler.constant("DELTA_WIDTH", neura_profile::DELTA_WIDTH);
    compiler.constant("DELTA_STATE", 0u32);
    compiler.constant("DELTA_COLUMNS", neura_profile::DELTA_COLUMNS);
    compiler.constant("DELTA_SPLITS", neura_profile::DELTA_SPLITS);
    compiler.constant("DELTA_DELTA", neura_profile::DELTA_DELTA);
    compiler.constant("DELTA_Q", neura_profile::DELTA_Q);
    compiler.constant("DELTA_K", neura_profile::DELTA_K);
    compiler.constant("DELTA_V", neura_profile::DELTA_V);
    compiler.constant("DELTA_REDUCE", neura_profile::DELTA_REDUCE);
    compiler.constant("DELTA_PARTIALS", neura_profile::DELTA_PARTIALS);
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn run_delta(task: Task, lid: u32) {
        let state = values[task.a];
        let query = values[task.b];
        let key = values[task.c];
        let value = values[task.d];
        let beta = values[task.e];
        let log_decay = values[task.f];
        let out = values[task.out];
        let width = state.dims.w;
        let state_rows = state.dims.z;
        let rows = query.dims.z;
        let planes = state.dims.x * state.dims.y;
        let first = task.first;
        let count = task.count;
        if width > DELTA_WIDTH
            || state_rows != width
            || rows == 0u32
            || count == 0u32
            || count > DELTA_COLUMNS
            || first + count > width
            || task.plane >= planes
        {
            refuse(kind::DELTA, refusal::GEOMETRY, task.plane);
            return;
        }
        let head = task.plane / state.dims.y;
        let slot = task.plane % state.dims.y;
        let state_base = head * state.strides.x + slot * state.strides.y;
        let query_base = head * query.strides.x + slot * query.strides.y;
        let key_base = head * key.strides.x + slot * key.strides.y;
        let value_base = head * value.strides.x + slot * value.strides.y;
        let scalar_at = head * beta.strides.x + slot * beta.strides.y;
        for index in stride(lid, state_rows * count, WORKGROUP_SIZE) {
            let state_row = index / count;
            let column = index % count;
            scratch[DELTA_STATE + state_row * count + column] = fetch(
                state,
                state_base + state_row * state.strides.z + (first + column) * state.strides.w,
            );
        }
        workgroup_barrier();
        let lane_groups = min(WORKGROUP_SIZE / count, DELTA_SPLITS);
        let carried = DELTA_PARTIALS + DELTA_SPLITS * DELTA_COLUMNS;
        for row in stride(0u32, rows, 1u32) {
            for column in stride(lid, width, WORKGROUP_SIZE) {
                scratch[DELTA_Q + column] = fetch(
                    query,
                    query_base + row * query.strides.z + column * query.strides.w,
                );
                scratch[DELTA_K + column] =
                    fetch(key, key_base + row * key.strides.z + column * key.strides.w);
                scratch[DELTA_V + column] = fetch(
                    value,
                    value_base + row * value.strides.z + column * value.strides.w,
                );
            }
            workgroup_barrier();
            let decay = exp(fetch(log_decay, scalar_at + row * log_decay.strides.z));
            let gain = fetch(beta, scalar_at + row * beta.strides.z);
            let mut keep = 1.0;
            if task.bias != NO_VALUE {
                let mask = values[task.bias];
                keep = fetch(
                    mask,
                    head * mask.strides.x + slot * mask.strides.y + row * mask.strides.z,
                );
            }
            let column = lid % count;
            let lane = lid / count;
            let mut memory = 0.0;
            let mut attended = 0.0;
            if lane < lane_groups {
                for state_row in stride(lane, state_rows, lane_groups) {
                    let held = scratch[DELTA_STATE + state_row * count + column];
                    memory = memory + scratch[DELTA_K + state_row] * held;
                    attended = attended + scratch[DELTA_Q + state_row] * held;
                }
                scratch[DELTA_PARTIALS + lane * count + column] = memory;
                scratch[carried + lane * count + column] = attended;
            }
            if lid == 0u32 {
                let mut qk = 0.0;
                for at in stride(0u32, width, 1u32) {
                    qk = qk + scratch[DELTA_Q + at] * scratch[DELTA_K + at];
                }
                scratch[DELTA_REDUCE] = qk;
            }
            workgroup_barrier();
            if lid < count {
                let mut total_memory = 0.0;
                let mut total_attended = 0.0;
                for group in stride(0u32, lane_groups, 1u32) {
                    total_memory = total_memory + scratch[DELTA_PARTIALS + group * count + lid];
                    total_attended = total_attended + scratch[carried + group * count + lid];
                }
                let memory = decay * total_memory;
                let delta = (scratch[DELTA_V + first + lid] - memory) * gain;
                scratch[DELTA_DELTA + lid] = delta;
                let held = decay * total_attended + scratch[DELTA_REDUCE] * delta;
                publish(
                    out,
                    head * out.strides.x
                        + slot * out.strides.y
                        + row * out.strides.z
                        + (first + lid) * out.strides.w,
                    keep * held,
                );
            }
            workgroup_barrier();
            if keep != 0.0 {
                for index in stride(lid, state_rows * count, WORKGROUP_SIZE) {
                    let state_row = index / count;
                    let owned_column = index % count;
                    let at = DELTA_STATE + state_row * count + owned_column;
                    scratch[at] = decay * scratch[at]
                        + scratch[DELTA_K + state_row] * scratch[DELTA_DELTA + owned_column];
                }
            }
            workgroup_barrier();
        }
        for index in stride(lid, state_rows * count, WORKGROUP_SIZE) {
            let state_row = index / count;
            let column = index % count;
            publish(
                state,
                state_base + state_row * state.strides.z + (first + column) * state.strides.w,
                scratch[DELTA_STATE + state_row * count + column],
            );
        }
        storage_barrier();
    }
}
