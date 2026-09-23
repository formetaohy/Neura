fn run_one_hot(task: Task, lid: u32) {
    let indices = values[task.a];
    let output = values[task.out];
    let classes = output.dims.w;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let row = index / classes;
        let column = index % classes;
        let at = coordinates(row, indices.dims);
        let chosen = whole_index(fetch(indices, read_address(at, indices.strides)), classes, OneHot);
        publish(output, index, chained(task, coordinates(index, output.dims), select(0.0, 1.0, column == chosen)));
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
        let at = coordinates(row, indices.dims);
        let chosen = whole_index(fetch(indices, read_address(at, indices.strides)), rows, Gather);
        publish(output, index, chained(task, coordinates(index, output.dims), fetch(table, row_origin(table, chosen) + column * table.strides.w)));
    }
}
