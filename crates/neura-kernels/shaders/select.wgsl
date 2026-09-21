fn whole_index(value: f32, rows: u32, kind: u32) -> u32 {
    if (trunc(value) != value || !(value >= 0.0) || !(value < f32(rows))) {
        refuse(kind, 0u);
        return 0u;
    }
    return u32(value);
}

fn run_one_hot(task: Task, lid: u32) {
    let indices = values[task.a];
    let output = values[task.out];
    let classes = output.dims.w;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let row = index / classes;
        let column = index % classes;
        let chosen = whole_index(fetch(indices.base, row), classes, OneHot);
        publish(output.base, index, chained(task, index, select(0.0, 1.0, column == chosen)));
    }
}

fn run_gather(task: Task, lid: u32) {
    let table = values[task.a];
    let indices = values[task.b];
    let output = values[task.out];
    let width = output.dims.w;
    let rows = table.dims.x * table.dims.y * table.dims.z;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let row = index / width;
        let column = index % width;
        let chosen = whole_index(fetch(indices.base, row), rows, Gather);
        publish(output.base, index, chained(task, index, fetch(table.base, chosen * width + column)));
    }
}
