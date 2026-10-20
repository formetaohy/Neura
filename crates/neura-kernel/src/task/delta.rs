use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    compiler.constant("DELTA_WIDTH", neura_profile::DELTA_WIDTH);
    compiler.constant("DELTA_TILE_ROWS", neura_profile::DELTA_TILE_ROWS);
    compiler.constant("DELTA_TILE_STRIDE", neura_profile::DELTA_TILE_STRIDE);
    compiler.constant("DELTA_TILE", 0u32);
    compiler.constant("DELTA_KH", neura_profile::DELTA_KH);
    compiler.constant("DELTA_QH", neura_profile::DELTA_QH);
    compiler.constant("DELTA_DELTA", neura_profile::DELTA_DELTA);
    compiler.constant("DELTA_Q", neura_profile::DELTA_Q);
    compiler.constant("DELTA_K", neura_profile::DELTA_K);
    compiler.constant("DELTA_V", neura_profile::DELTA_V);
    compiler.constant("DELTA_REDUCE", neura_profile::DELTA_REDUCE);
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
        let rows = state.dims.z;
        let planes = state.dims.x * state.dims.y;
        if width > DELTA_WIDTH || rows != width || task.plane >= planes {
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
        let decay = exp(fetch(log_decay, scalar_at));
        let gain = fetch(beta, scalar_at);
        let mut keep = 1.0;
        if task.bias != NO_VALUE {
            let mask = values[task.bias];
            keep = fetch(mask, head * mask.strides.x + slot * mask.strides.y);
        }
        for column in stride(lid, width, WORKGROUP_SIZE) {
            scratch[DELTA_Q + column] = fetch(query, query_base + column * query.strides.w);
            scratch[DELTA_K + column] = fetch(key, key_base + column * key.strides.w);
            scratch[DELTA_V + column] = fetch(value, value_base + column * value.strides.w);
            scratch[DELTA_KH + column] = 0.0;
            scratch[DELTA_QH + column] = 0.0;
        }
        workgroup_barrier();
        let tiles = (rows + DELTA_TILE_ROWS - 1u32) / DELTA_TILE_ROWS;
        for tile in stride(0u32, tiles, 1u32) {
            let first = tile * DELTA_TILE_ROWS;
            let count = min(DELTA_TILE_ROWS, rows - first);
            for index in stride(lid, count * width, WORKGROUP_SIZE) {
                let row = index / width;
                let column = index % width;
                scratch[DELTA_TILE + row * DELTA_TILE_STRIDE + column] = fetch(
                    state,
                    state_base + (first + row) * state.strides.z + column * state.strides.w,
                );
            }
            workgroup_barrier();
            for column in stride(lid, width, WORKGROUP_SIZE) {
                let mut memory = 0.0;
                let mut attended = 0.0;
                for row in stride(0u32, count, 1u32) {
                    let held = scratch[DELTA_TILE + row * DELTA_TILE_STRIDE + column];
                    memory = memory + scratch[DELTA_K + first + row] * held;
                    attended = attended + scratch[DELTA_Q + first + row] * held;
                }
                scratch[DELTA_KH + column] = scratch[DELTA_KH + column] + memory;
                scratch[DELTA_QH + column] = scratch[DELTA_QH + column] + attended;
            }
            workgroup_barrier();
        }
        for column in stride(lid, width, WORKGROUP_SIZE) {
            let memory = decay * scratch[DELTA_KH + column];
            scratch[DELTA_DELTA + column] = (scratch[DELTA_V + column] - memory) * gain;
        }
        workgroup_barrier();
        let mut local = 0.0;
        for column in stride(lid, width, WORKGROUP_SIZE) {
            local = local + scratch[DELTA_Q + column] * scratch[DELTA_K + column];
        }
        for index in stride(lid, DELTA_WIDTH, WORKGROUP_SIZE) {
            scratch[DELTA_REDUCE + index] = 0.0;
        }
        workgroup_barrier();
        if lid < DELTA_WIDTH {
            scratch[DELTA_REDUCE + lid] = local;
        }
        workgroup_barrier();
        let mut pending = DELTA_WIDTH / 2u32;
        loop {
            if pending == 0u32 {
                break;
            }
            if lid < pending {
                scratch[DELTA_REDUCE + lid] =
                    scratch[DELTA_REDUCE + lid] + scratch[DELTA_REDUCE + lid + pending];
            }
            workgroup_barrier();
            pending = pending / 2u32;
        }
        let qk = scratch[DELTA_REDUCE];
        workgroup_barrier();
        for column in stride(lid, width, WORKGROUP_SIZE) {
            let attended = decay * scratch[DELTA_QH + column] + qk * scratch[DELTA_DELTA + column];
            publish(
                out,
                head * out.strides.x + slot * out.strides.y + column * out.strides.w,
                keep * attended,
            );
        }
        if keep != 0.0 {
            for index in stride(lid, rows * width, WORKGROUP_SIZE) {
                let row = index / width;
                let column = index % width;
                let at = state_base + row * state.strides.z + column * state.strides.w;
                publish(
                    state,
                    at,
                    decay * fetch(state, at)
                        + scratch[DELTA_K + row] * scratch[DELTA_DELTA + column],
                );
            }
        }
        storage_barrier();
    }
}
