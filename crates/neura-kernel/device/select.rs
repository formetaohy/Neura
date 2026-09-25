#[neura_compiler::module]
mod source {
    fn run_one_hot(task: Task, lid: u32) {
        let indices = values[task.a];
        let output = values[task.out];
        let classes = output.dims.w;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let row = index / classes;
            let column = index % classes;
            let chosen = whole_index(fetch(indices, row), classes, kind::ONE_HOT);
            publish(
                output,
                index,
                chained(
                    task,
                    coordinates(index, output.dims),
                    select(0.0, 1.0, column == chosen),
                ),
            );
        }
    }

    fn run_gather(task: Task, lid: u32) {
        let table = values[task.a];
        let indices = values[task.b];
        let output = values[task.out];
        let width = output.dims.w;
        let rows = table.dims.x * table.dims.y * table.dims.z;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let row = index / width;
            let column = index % width;
            let chosen = whole_index(fetch(indices, row), rows, kind::GATHER);
            publish(
                output,
                index,
                chained(
                    task,
                    coordinates(index, output.dims),
                    fetch(table, chosen * width + column),
                ),
            );
        }
    }
}
