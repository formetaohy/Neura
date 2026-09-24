fn run_scatter(task: Task, lid: u32) {
    let into = values[task.a];
    let indices = values[task.b];
    let updates = values[task.c];
    let width = into.dims.w;
    let rows = into.dims.x * into.dims.y * into.dims.z;
    for (var column = lid; column < width; column = column + WORKGROUP_SIZE) {
        for (var row = task.first; row < task.first + task.count; row = row + 1u) {
            let chosen = whole_index(fetch(indices, row), rows, Scatter);
            let address = chosen * width + column;
            publish(into, address, fetch(into, address) + fetch(updates, row * width + column));
        }
    }
}
