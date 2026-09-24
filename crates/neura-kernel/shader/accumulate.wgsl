fn run_accumulate(task: Task, lid: u32, slot: u32) {
    let view = values[task.a];
    let into = values[task.b];
    let source = values[task.c];
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, view.dims);
        let address = task.origin + read_address(at, view.strides);
        let added = fetch(source, read_address(at, source.strides));
        publish(into, address, fetch(into, address) + added);
    }
}
