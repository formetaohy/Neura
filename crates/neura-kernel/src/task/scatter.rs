use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn run_scatter(task: Task, lid: u32) {
        let into = values[task.a];
        let indices = values[task.b];
        let updates = values[task.c];
        let width = into.dims.w;
        let rows = into.dims.x * into.dims.y * into.dims.z;
        for column in stride(lid, width, WORKGROUP_SIZE) {
            for row in stride(task.first, task.first + task.count, 1u32) {
                let chosen = whole_index(fetch(indices, row), rows, kind::SCATTER, refusal::INDEX);
                let address = chosen * width + column;
                publish(
                    into,
                    address,
                    fetch(into, address) + fetch(updates, row * width + column),
                );
            }
        }
    }

    fn run_compact(task: Task, lid: u32) {
        let mask = values[task.a];
        let positions = values[task.b];
        let indices = values[task.out];
        let rows = indices.dims.x * indices.dims.y * indices.dims.z;
        for row in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let flag = fetch(mask, row);
            if flag != 0.0 && flag != 1.0 {
                refuse(kind::COMPACT, refusal::MASK, 0u32);
            }
            if flag == 1.0 {
                let at = whole_index(fetch(positions, row), rows, kind::COMPACT, refusal::INDEX);
                publish(indices, at, f32(row));
            }
        }
    }

    fn run_scatter_write(task: Task, lid: u32) {
        let into = values[task.a];
        let indices = values[task.b];
        let updates = values[task.c];
        let width = into.dims.w;
        let rows = into.dims.x * into.dims.y * into.dims.z;
        for column in stride(lid, width, WORKGROUP_SIZE) {
            for row in stride(task.first, task.first + task.count, 1u32) {
                let chosen = whole_index(
                    fetch(indices, row),
                    rows,
                    kind::SCATTER_WRITE,
                    refusal::INDEX,
                );
                publish(
                    into,
                    chosen * width + column,
                    fetch(updates, row * width + column),
                );
            }
        }
    }
}
