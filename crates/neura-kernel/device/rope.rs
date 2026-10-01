#[neura_compiler::module]
mod source {
    fn rope_position(task: Task, at: uvec4) -> u32 {
        if task.origin == NO_VALUE {
            return 0u32;
        }
        let cursor = values[task.origin];
        let raw = fetch(
            cursor,
            read_address(uvec4(at.x, at.y, 0u32, 0u32), cursor.strides),
        );
        if trunc(raw) != raw || !(raw >= 0.0) || !(raw < 4294967296.0) {
            refuse(task.kind, refusal::ORIGIN, 0u32);
            return 0u32;
        }
        return u32(raw);
    }

    fn rope_channel(at: uvec4, high: bool, half: u32) -> u32 {
        return select(at.w, at.w - half, high);
    }

    fn rope_pair(at: uvec4, high: bool, half: u32) -> uvec4 {
        return uvec4(at.x, at.y, at.z, select(at.w + half, at.w - half, high));
    }

    fn rope_angle(position: u32, channel: u32, width: u32, base: f32) -> f32 {
        return f32(position) * pow(base, -2.0 * f32(channel) / f32(width));
    }

    fn run_rope(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let dims = output.dims;
        let half = dims.w / 2u32;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, dims);
            let high = at.w >= half;
            let angle = rope_angle(
                rope_position(task, at) + at.z,
                rope_channel(at, high, half),
                dims.w,
                task.param,
            );
            let value = fetch(source, read_address(at, source.strides));
            let pair = rope_pair(at, high, half);
            let partner = fetch(source, read_address(pair, source.strides));
            let cosine = cos(angle);
            let sine = sin(angle);
            publish(
                output,
                index,
                chained(
                    task,
                    at,
                    select(
                        value * cosine - partner * sine,
                        value * cosine + partner * sine,
                        high,
                    ),
                ),
            );
        }
    }

    fn run_rope_grad(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let dims = output.dims;
        let half = dims.w / 2u32;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, dims);
            let high = at.w >= half;
            let angle = rope_angle(
                rope_position(task, at) + at.z,
                rope_channel(at, high, half),
                dims.w,
                task.param,
            );
            let value = fetch(source, read_address(at, source.strides));
            let pair = rope_pair(at, high, half);
            let partner = fetch(source, read_address(pair, source.strides));
            let cosine = cos(angle);
            let sine = sin(angle);
            publish(
                output,
                index,
                chained(
                    task,
                    at,
                    select(
                        value * cosine + partner * sine,
                        value * cosine - partner * sine,
                        high,
                    ),
                ),
            );
        }
    }
}
