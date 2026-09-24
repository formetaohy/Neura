fn run_concat(task: Task, lid: u32, slot: u32) {
    let source = values[task.a];
    let output = values[task.out];
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, source.dims);
        var shifted = at;
        if (task.slot == 0u) {
            shifted.x = shifted.x + task.origin;
        }
        if (task.slot == 1u) {
            shifted.y = shifted.y + task.origin;
        }
        if (task.slot == 2u) {
            shifted.z = shifted.z + task.origin;
        }
        if (task.slot == 3u) {
            shifted.w = shifted.w + task.origin;
        }
        let value = fetch(source, read_address(at, source.strides));
        publish(output, read_address(shifted, output.strides), chained(task, shifted, value, slot));
    }
}
