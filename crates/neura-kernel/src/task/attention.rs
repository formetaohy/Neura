use neura_compiler::{Compiler, ast};
use neura_profile::Geometry;

pub(crate) fn install(compiler: &mut Compiler, geometry: &Geometry) {
    device::define(compiler);
    if geometry.attention().is_empty() {
        return;
    }
    let stage = geometry.attention_stage_words();
    compiler.constant("SCRATCH_ATTENTION_RIGHT", stage);
    for (index, tile) in geometry.attention().iter().enumerate() {
        let suffix = index.to_string();
        let constants = [("ATTN_KEYS", tile.keys()), ("ATTN_WIDTH", tile.width())];
        compiler.specialize(
            "template_stage_attention",
            &format!("stage_attention_{suffix}"),
            &constants,
        );
        for (template, dispatcher) in [
            ("template_attention_forward", "run_attention"),
            ("template_attention_query_grad", "run_attention_query_grad"),
            ("template_attention_key_grad", "run_attention_key_grad"),
            ("template_attention_value_grad", "run_attention_value_grad"),
        ] {
            let specialized = format!("{dispatcher}_{suffix}");
            compiler.specialize(template, &specialized, &constants);
            compiler.insert_case(
                dispatcher,
                ast::Arm {
                    pattern: ast::Pattern::Integer(
                        index
                            .try_into()
                            .expect("a geometry fits in one device word"),
                    ),
                    body: vec![ast::Statement::Expression(ast::Expression::call(
                        specialized,
                        vec![ast::Expression::name("task"), ast::Expression::name("lid")],
                    ))],
                },
            );
        }
    }
}

#[neura_compiler::module]
mod device {
    fn key_position(at: u32, keys: u32, written: u32) -> u32 {
        if written <= keys {
            return at;
        }
        return at + keys * ((written - 1u32 - at) / keys);
    }

    fn weighed(at: u32, position: u32, reach: u32) -> bool {
        if at > position {
            return false;
        }
        if reach > 0u32 && position - at >= reach {
            return false;
        }
        return true;
    }

    fn visible(at: u32, keys: u32, written: u32, position: u32, reach: u32, causal: bool) -> bool {
        if at >= keys {
            return false;
        }
        if !causal {
            return true;
        }
        return weighed(
            select(at, key_position(at, keys, written), reach > 0u32),
            position,
            reach,
        );
    }

    fn attended(
        at: u32,
        tokens: u32,
        keys: u32,
        column: u32,
        origin: u32,
        reach: u32,
        causal: bool,
    ) -> bool {
        if at >= tokens {
            return false;
        }
        if !causal {
            return true;
        }
        return weighed(
            select(
                column,
                key_position(column, keys, origin + tokens),
                reach > 0u32,
            ),
            at + origin,
            reach,
        );
    }

    fn block_origin(task: Task, head: u32, batch: u32, keys: u32, tokens: u32) -> u32 {
        if task.origin == NO_VALUE || keys == 0u32 {
            return 0u32;
        }
        if keys < tokens {
            refuse(task.kind, refusal::ORIGIN, 0u32);
            return 0u32;
        }
        let cursor = values[task.origin];
        let raw = fetch(
            cursor,
            read_address(uvec4(head, batch, 0u32, 0u32), cursor.strides),
        );
        let bound = select(keys - tokens + 1u32, EXACT_WALK_LIMIT, task.reach > 0u32);
        return whole_index(raw, bound, task.kind, refusal::ORIGIN);
    }

    fn template_stage_attention(
        lid: u32,
        block: u32,
        left_plane: u32,
        right_plane: u32,
        left: Value,
        right: Value,
        left_bound: u32,
        right_bound: u32,
    ) {
        for unit in stride(lid, ATTN_KEYS * ATTN_WIDTH, WORKGROUP_SIZE) {
            let column = unit / ATTN_WIDTH;
            let depth = unit % ATTN_WIDTH;
            let at = block * ATTN_KEYS + column;
            let inside = at < left_bound;
            let address = select(
                0u32,
                left_plane + at * left.strides.z + depth * left.strides.w,
                inside,
            );
            scratch[unit] = select(0.0, fetch(left, address), inside);
        }
        for unit in stride(lid, ATTN_KEYS * ATTN_WIDTH, WORKGROUP_SIZE) {
            let column = unit / ATTN_WIDTH;
            let depth = unit % ATTN_WIDTH;
            let at = block * ATTN_KEYS + column;
            let inside = at < right_bound;
            let address = select(
                0u32,
                right_plane + at * right.strides.z + depth * right.strides.w,
                inside,
            );
            scratch[SCRATCH_ATTENTION_RIGHT + unit] = select(0.0, fetch(right, address), inside);
        }
    }

    fn template_attention_forward(task: Task, lid: u32) {
        if task.count == 0u32 {
            return;
        }
        let query = values[task.a];
        let key = values[task.b];
        let value = values[task.c];
        let output = values[task.out];
        let statistic = values[task.extra];
        let tokens = query.dims.z;
        let segmented = task.segment != NO_VALUE;
        let keys = select(key.dims.z, task.keys, segmented);
        let groups = query.dims.x / key.dims.x;
        let plane = coordinates(
            task.first,
            uvec4(query.dims.x, query.dims.y, query.dims.z, 1u32),
        );
        let head = plane.x / groups;
        let query_plane = plane.x * query.strides.x + plane.y * query.strides.y;
        let mut key_plane = head * key.strides.x + plane.y * key.strides.y;
        let mut value_plane = head * value.strides.x + plane.y * value.strides.y;
        if segmented {
            let offsets = values[task.segment];
            let start = whole_index(
                fetch(offsets, task.plane),
                key.dims.z + 1u32,
                kind::ATTENTION,
                refusal::INDEX,
            );
            key_plane = start * key.strides.z;
            value_plane = start * value.strides.z;
        }
        let output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let statistic_plane = plane.x * statistic.strides.x + plane.y * statistic.strides.y;
        let row = plane.z + lid;
        let inside = lid < task.count;
        let causal = task.slot == 1u32;
        let mut origin = 0u32;
        if keys >= tokens || task.reach > 0u32 {
            origin = block_origin(task, head, plane.y, keys, tokens);
        }
        let position = origin + row;
        let written = origin + tokens;
        let reached = causal && task.origin == NO_VALUE;
        let blocks = (keys + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let walked = select(
            blocks,
            (plane.z + task.count + ATTN_KEYS - 1u32) / ATTN_KEYS,
            reached,
        );
        let back = task.reach - 1u32;
        let window = select(
            0u32,
            plane.z - min(plane.z, back),
            task.reach > 0u32 && task.origin == NO_VALUE,
        );
        let first_block = window / ATTN_KEYS;
        let mut queries = scalar_array(0.0, ATTN_WIDTH);
        let mut accumulated = scalar_array(0.0, ATTN_WIDTH);
        let mut weights = scalar_array(0.0, ATTN_KEYS);
        let mut largest = -3.4028235e38;
        let mut total = 0.0;
        if inside {
            for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                queries[depth] = fetch(
                    query,
                    query_plane + row * query.strides.z + depth * query.strides.w,
                );
            }
        }
        for block in stride(first_block, walked, 1u32) {
            workgroup_barrier();
            template_stage_attention(lid, block, key_plane, value_plane, key, value, keys, keys);
            workgroup_barrier();
            if inside {
                let mut block_largest = -3.4028235e38;
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let at = block * ATTN_KEYS + column;
                    let mut score = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        score = score + queries[depth] * scratch[column * ATTN_WIDTH + depth];
                    }
                    weights[column] = select(
                        -3.4028235e38,
                        score * task.param,
                        visible(at, keys, written, position, task.reach, causal),
                    );
                    block_largest = max(block_largest, weights[column]);
                }
                let next = max(largest, block_largest);
                let rescale = exp(largest - next);
                largest = next;
                let mut block_total = 0.0;
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let weight = exp(weights[column] - largest);
                    weights[column] = weight;
                    block_total = block_total + weight;
                }
                total = total * rescale + block_total;
                for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                    let mut carried = accumulated[depth] * rescale;
                    for column in unroll(0u32, ATTN_KEYS, 1u32) {
                        carried = carried
                            + weights[column]
                                * scratch[SCRATCH_ATTENTION_RIGHT + column * ATTN_WIDTH + depth];
                    }
                    accumulated[depth] = carried;
                }
            }
        }
        if inside {
            let carries_keys = total > 0.0;
            let normalized = select(0.0, 1.0 / total, carries_keys);
            let at = uvec4(plane.x, plane.y, row, 0u32);
            for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                publish(
                    output,
                    output_plane + row * output.strides.z + depth * output.strides.w,
                    chained(
                        task,
                        at + uvec4(0u32, 0u32, 0u32, depth),
                        accumulated[depth] * normalized,
                    ),
                );
            }
            publish(
                statistic,
                statistic_plane + row * statistic.strides.z,
                select(0.0, largest + log(total), carries_keys),
            );
        }
    }

    fn template_attention_query_grad(task: Task, lid: u32) {
        if task.segment != NO_VALUE {
            refuse(kind::ATTENTION_QUERY_GRAD, refusal::TASK, 0u32);
        }
        if task.count == 0u32 {
            return;
        }
        let query = values[task.a];
        let key = values[task.b];
        let value = values[task.c];
        let gradient = values[task.d];
        let output_grad = values[task.e];
        let statistic = values[task.f];
        let output = values[task.out];
        let tokens = query.dims.z;
        let keys = key.dims.z;
        let groups = query.dims.x / key.dims.x;
        let plane = coordinates(
            task.first,
            uvec4(query.dims.x, query.dims.y, query.dims.z, 1u32),
        );
        let head = plane.x / groups;
        let query_plane = plane.x * query.strides.x + plane.y * query.strides.y;
        let key_plane = head * key.strides.x + plane.y * key.strides.y;
        let value_plane = head * value.strides.x + plane.y * value.strides.y;
        let gradient_plane = plane.x * gradient.strides.x + plane.y * gradient.strides.y;
        let output_grad_plane = plane.x * output_grad.strides.x + plane.y * output_grad.strides.y;
        let statistic_plane = plane.x * statistic.strides.x + plane.y * statistic.strides.y;
        let output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let row = plane.z + lid;
        let inside = lid < task.count;
        let causal = task.slot == 1u32;
        let origin = block_origin(task, head, plane.y, keys, tokens);
        let position = origin + row;
        let written = origin + tokens;
        let reached = causal && task.origin == NO_VALUE;
        let blocks = (keys + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let walked = select(
            blocks,
            (plane.z + task.count + ATTN_KEYS - 1u32) / ATTN_KEYS,
            reached,
        );
        let back = task.reach - 1u32;
        let window = select(
            0u32,
            plane.z - min(plane.z, back),
            task.reach > 0u32 && task.origin == NO_VALUE,
        );
        let first_block = window / ATTN_KEYS;
        let mut queries = scalar_array(0.0, ATTN_WIDTH);
        let mut gradients = scalar_array(0.0, ATTN_WIDTH);
        let mut accumulated = scalar_array(0.0, ATTN_WIDTH);
        let mut row_dot = 0.0;
        let mut normalizer = 0.0;
        if inside {
            for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                queries[depth] = fetch(
                    query,
                    query_plane + row * query.strides.z + depth * query.strides.w,
                );
                gradients[depth] = fetch(
                    gradient,
                    gradient_plane + row * gradient.strides.z + depth * gradient.strides.w,
                );
                row_dot = row_dot
                    + gradients[depth]
                        * fetch(
                            output_grad,
                            output_grad_plane
                                + row * output_grad.strides.z
                                + depth * output_grad.strides.w,
                        );
            }
            normalizer = fetch(statistic, statistic_plane + row * statistic.strides.z);
        }
        for block in stride(first_block, walked, 1u32) {
            workgroup_barrier();
            template_stage_attention(lid, block, key_plane, value_plane, key, value, keys, keys);
            workgroup_barrier();
            if inside {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let at = block * ATTN_KEYS + column;
                    let mut score = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        score = score + queries[depth] * scratch[column * ATTN_WIDTH + depth];
                    }
                    let weight = select(
                        0.0,
                        exp(score * task.param - normalizer),
                        visible(at, keys, written, position, task.reach, causal),
                    );
                    let mut weighted = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        weighted = weighted
                            + gradients[depth]
                                * scratch[SCRATCH_ATTENTION_RIGHT + column * ATTN_WIDTH + depth];
                    }
                    let scored = weight * (weighted - row_dot) * task.param;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        accumulated[depth] =
                            accumulated[depth] + scored * scratch[column * ATTN_WIDTH + depth];
                    }
                }
            }
        }
        if inside {
            let at = uvec4(plane.x, plane.y, row, 0u32);
            for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                publish(
                    output,
                    output_plane + row * output.strides.z + depth * output.strides.w,
                    chained(
                        task,
                        at + uvec4(0u32, 0u32, 0u32, depth),
                        accumulated[depth],
                    ),
                );
            }
        }
    }

    fn template_attention_key_grad(task: Task, lid: u32) {
        if task.segment != NO_VALUE {
            refuse(kind::ATTENTION_KEY_GRAD, refusal::TASK, 0u32);
        }
        if task.count == 0u32 {
            return;
        }
        let query = values[task.a];
        let key = values[task.b];
        let value = values[task.c];
        let gradient = values[task.d];
        let output_grad = values[task.e];
        let statistic = values[task.f];
        let output = values[task.out];
        let tokens = query.dims.z;
        let keys = key.dims.z;
        let groups = query.dims.x / key.dims.x;
        let plane = coordinates(task.first, uvec4(key.dims.x, key.dims.y, key.dims.z, 1u32));
        let key_plane = plane.x * key.strides.x + plane.y * key.strides.y;
        let value_plane = plane.x * value.strides.x + plane.y * value.strides.y;
        let output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let column = plane.z + lid;
        let inside = lid < task.count;
        let causal = task.slot == 1u32;
        let origin = block_origin(task, plane.x, plane.y, keys, tokens);
        let blocks = (tokens + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let first = select(0u32, plane.z / ATTN_KEYS, causal && task.origin == NO_VALUE);
        let mut keys_row = scalar_array(0.0, ATTN_WIDTH);
        let mut values_row = scalar_array(0.0, ATTN_WIDTH);
        let mut accumulated = scalar_array(0.0, ATTN_WIDTH);
        if inside {
            for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                keys_row[depth] = fetch(
                    key,
                    key_plane + column * key.strides.z + depth * key.strides.w,
                );
                values_row[depth] = fetch(
                    value,
                    value_plane + column * value.strides.z + depth * value.strides.w,
                );
            }
        }
        for group in stride(0u32, groups, 1u32) {
            let query_plane =
                (plane.x * groups + group) * query.strides.x + plane.y * query.strides.y;
            let gradient_plane =
                (plane.x * groups + group) * gradient.strides.x + plane.y * gradient.strides.y;
            let output_grad_plane = (plane.x * groups + group) * output_grad.strides.x
                + plane.y * output_grad.strides.y;
            let statistic_plane =
                (plane.x * groups + group) * statistic.strides.x + plane.y * statistic.strides.y;
            for block in stride(first, blocks, 1u32) {
                workgroup_barrier();
                template_stage_attention(
                    lid,
                    block,
                    query_plane,
                    gradient_plane,
                    query,
                    gradient,
                    tokens,
                    tokens,
                );
                workgroup_barrier();
                if inside {
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        let mut score = 0.0;
                        for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                            score = score + scratch[step * ATTN_WIDTH + depth] * keys_row[depth];
                        }
                        let weight = select(
                            0.0,
                            exp(score * task.param
                                - fetch(statistic, statistic_plane + at * statistic.strides.z)),
                            attended(at, tokens, keys, column, origin, task.reach, causal),
                        );
                        let mut weighted = 0.0;
                        let mut row_dot = 0.0;
                        for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                            weighted = weighted
                                + scratch[SCRATCH_ATTENTION_RIGHT + step * ATTN_WIDTH + depth]
                                    * values_row[depth];
                            row_dot = row_dot
                                + scratch[SCRATCH_ATTENTION_RIGHT + step * ATTN_WIDTH + depth]
                                    * fetch(
                                        output_grad,
                                        output_grad_plane
                                            + at * output_grad.strides.z
                                            + depth * output_grad.strides.w,
                                    );
                        }
                        let scored = weight * (weighted - row_dot) * task.param;
                        for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                            accumulated[depth] =
                                accumulated[depth] + scored * scratch[step * ATTN_WIDTH + depth];
                        }
                    }
                }
            }
        }
        if inside {
            let at = uvec4(plane.x, plane.y, column, 0u32);
            for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                publish(
                    output,
                    output_plane + column * output.strides.z + depth * output.strides.w,
                    chained(
                        task,
                        at + uvec4(0u32, 0u32, 0u32, depth),
                        accumulated[depth],
                    ),
                );
            }
        }
    }

    fn template_attention_value_grad(task: Task, lid: u32) {
        if task.segment != NO_VALUE {
            refuse(kind::ATTENTION_VALUE_GRAD, refusal::TASK, 0u32);
        }
        if task.count == 0u32 {
            return;
        }
        let query = values[task.a];
        let key = values[task.b];
        let gradient = values[task.d];
        let statistic = values[task.f];
        let output = values[task.out];
        let tokens = query.dims.z;
        let keys = key.dims.z;
        let groups = query.dims.x / key.dims.x;
        let plane = coordinates(task.first, uvec4(key.dims.x, key.dims.y, key.dims.z, 1u32));
        let key_plane = plane.x * key.strides.x + plane.y * key.strides.y;
        let output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let column = plane.z + lid;
        let inside = lid < task.count;
        let causal = task.slot == 1u32;
        let origin = block_origin(task, plane.x, plane.y, keys, tokens);
        let blocks = (tokens + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let first = select(0u32, plane.z / ATTN_KEYS, causal && task.origin == NO_VALUE);
        let mut keys_row = scalar_array(0.0, ATTN_WIDTH);
        let mut accumulated = scalar_array(0.0, ATTN_WIDTH);
        if inside {
            for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                keys_row[depth] = fetch(
                    key,
                    key_plane + column * key.strides.z + depth * key.strides.w,
                );
            }
        }
        for group in stride(0u32, groups, 1u32) {
            let query_plane =
                (plane.x * groups + group) * query.strides.x + plane.y * query.strides.y;
            let gradient_plane =
                (plane.x * groups + group) * gradient.strides.x + plane.y * gradient.strides.y;
            let statistic_plane =
                (plane.x * groups + group) * statistic.strides.x + plane.y * statistic.strides.y;
            for block in stride(first, blocks, 1u32) {
                workgroup_barrier();
                template_stage_attention(
                    lid,
                    block,
                    query_plane,
                    gradient_plane,
                    query,
                    gradient,
                    tokens,
                    tokens,
                );
                workgroup_barrier();
                if inside {
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        let mut score = 0.0;
                        for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                            score = score + scratch[step * ATTN_WIDTH + depth] * keys_row[depth];
                        }
                        let weight = select(
                            0.0,
                            exp(score * task.param
                                - fetch(statistic, statistic_plane + at * statistic.strides.z)),
                            attended(at, tokens, keys, column, origin, task.reach, causal),
                        );
                        for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                            accumulated[depth] = accumulated[depth]
                                + weight
                                    * scratch[SCRATCH_ATTENTION_RIGHT + step * ATTN_WIDTH + depth];
                        }
                    }
                }
            }
        }
        if inside {
            let at = uvec4(plane.x, plane.y, column, 0u32);
            for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                publish(
                    output,
                    output_plane + column * output.strides.z + depth * output.strides.w,
                    chained(
                        task,
                        at + uvec4(0u32, 0u32, 0u32, depth),
                        accumulated[depth],
                    ),
                );
            }
        }
    }

    fn run_attention(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::ATTENTION, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_attention_query_grad(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::ATTENTION_QUERY_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_attention_key_grad(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::ATTENTION_KEY_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }

    fn run_attention_value_grad(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::ATTENTION_VALUE_GRAD, refusal::GEOMETRY, task.geometry),
        }
    }
}
