#[neura_compiler::module]
mod source {
    fn run_layout(task: Task, lid: u32) {
        let source = values[task.a];
        let layout = values[task.b];
        let output = values[task.out];
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, source.dims);
            publish(
                output,
                read_address(at, layout.strides),
                fetch(source, read_address(at, source.strides)),
            );
        }
    }
}
