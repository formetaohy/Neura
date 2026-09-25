#[neura_compiler::module]
mod source {
    fn run_scatter(task: Task, lid: u32) {
        let into = values[task.a];
        let indices = values[task.b];
        let updates = values[task.c];
        let width = into.dims.w;
        let rows = into.dims.x * into.dims.y * into.dims.z;
        for column in stride(lid, width, WORKGROUP_SIZE) {
            for row in stride(task.first, task.first + task.count, 1u32) {
                let chosen = whole_index(fetch(indices, row), rows, kind::SCATTER);
                let address = chosen * width + column;
                publish(
                    into,
                    address,
                    fetch(into, address) + fetch(updates, row * width + column),
                );
            }
        }
    }
}
