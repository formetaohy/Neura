use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
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

    fn run_extend(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            let held = at.x < source.dims.x
                && at.y < source.dims.y
                && at.z < source.dims.z
                && at.w < source.dims.w;
            publish(
                output,
                read_address(at, output.strides),
                select(0.0, fetch(source, read_address(at, source.strides)), held),
            );
        }
    }

    fn axis_shift(task: Task) -> uvec4 {
        return uvec4(
            select(0u32, task.offset, task.axis == 0u32),
            select(0u32, task.offset, task.axis == 1u32),
            select(0u32, task.offset, task.axis == 2u32),
            select(0u32, task.offset, task.axis == 3u32),
        );
    }

    fn run_concat(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let shift = axis_shift(task);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, source.dims);
            publish(
                output,
                read_address(at + shift, output.strides),
                fetch(source, read_address(at, source.strides)),
            );
        }
    }

    fn run_slice(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let shift = axis_shift(task);
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, output.dims);
            publish(
                output,
                index,
                fetch(source, read_address(at + shift, source.strides)),
            );
        }
    }
}
