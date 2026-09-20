fn run_sum_chunk(task: Task, lid: u32) {
    let source = values[task.a];
    let output = values[task.out];
    var local = 0.0;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        local = local + arena[source.base + index];
    }
    let total = reduce_chunk_sum(lid, local);
    if (lid == 0u) {
        arena[output.base + task.slot] = total;
    }
}

fn run_sum_to(task: Task, lid: u32) {
    let source = values[task.a];
    let output = values[task.out];
    let copies_x = select(1u, source.dims.x, output.dims.x == 1u);
    let copies_y = select(1u, source.dims.y, output.dims.y == 1u);
    let copies_z = select(1u, source.dims.z, output.dims.z == 1u);
    let copies_w = select(1u, source.dims.w, output.dims.w == 1u);
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let w = index % output.dims.w;
        let z = (index / output.dims.w) % output.dims.z;
        let y = (index / (output.dims.w * output.dims.z)) % output.dims.y;
        let x = index / (output.dims.w * output.dims.z * output.dims.y);
        var total = 0.0;
        for (var rx = 0u; rx < copies_x; rx = rx + 1u) {
            for (var ry = 0u; ry < copies_y; ry = ry + 1u) {
                for (var rz = 0u; rz < copies_z; rz = rz + 1u) {
                    for (var rw = 0u; rw < copies_w; rw = rw + 1u) {
                        let ox = select(x, rx, output.dims.x == 1u);
                        let oy = select(y, ry, output.dims.y == 1u);
                        let oz = select(z, rz, output.dims.z == 1u);
                        let ow = select(w, rw, output.dims.w == 1u);
                        total = total + arena[source.base + ox * source.strides.x + oy * source.strides.y + oz * source.strides.z + ow * source.strides.w];
                    }
                }
            }
        }
        arena[output.base + index] = chained(task, index, total);
    }
}
