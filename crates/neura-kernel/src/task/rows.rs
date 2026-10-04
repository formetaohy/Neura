use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn run_rows(task: Task, lid: u32) {
        if task.count == 0u32 {
            return;
        }
        let offsets = values[task.segment];
        let planes = values[task.out];
        let positions = values[task.extra];
        let start = whole_index(
            fetch(offsets, task.plane),
            planes.dims.z,
            kind::ROWS,
            refusal::INDEX,
        );
        for row in stride(lid, task.count, WORKGROUP_SIZE) {
            publish(planes, start + row, f32(task.plane));
            publish(positions, start + row, f32(row));
        }
    }
}
