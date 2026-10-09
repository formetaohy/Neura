use neura_compiler::{Compiler, ast};
use neura_profile::{AttentionTile, Geometry};

pub(crate) fn install(compiler: &mut Compiler, geometry: &Geometry) {
    device::define(compiler);
    if geometry.attention().is_empty() {
        return;
    }
    compiler.constant("ATTN_REDUCTIONS", AttentionTile::REDUCTIONS);
    for (index, tile) in geometry.attention().iter().enumerate() {
        let suffix = index.to_string();
        let constants = [
            ("ATTN_KEYS", tile.keys()),
            ("ATTN_WIDTH", tile.width()),
            ("ATTN_SLICES", tile.slices()),
            ("ATTN_SLICE", tile.slice_width()),
            ("SCRATCH_ATTENTION_RIGHT", tile.stage_words()),
            ("SCRATCH_ATTENTION_REDUCE", 2 * tile.stage_words()),
        ];
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

    fn walks_the_rows_its_offsets_close(task: Task, query: Value, key: Value) -> bool {
        return task.segment != NO_VALUE
            && (task.queries != NO_VALUE || slot_of(query.free, 2u32) == slot_of(key.free, 2u32));
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
        let cursor = values[task.origin];
        let raw = select(
            fetch(
                cursor,
                read_address(uvec4(head, batch, 0u32, 0u32), cursor.strides),
            ),
            fetch(cursor, task.plane),
            task.queries != NO_VALUE,
        );
        let origin = whole_index(raw, EXACT_WALK_LIMIT, task.kind, refusal::ORIGIN);
        if task.reach == 0u32 && origin + tokens > keys {
            refuse(task.kind, refusal::ORIGIN, 0u32);
        }
        return origin;
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
        let segmented = task.segment != NO_VALUE;
        let packed = walks_the_rows_its_offsets_close(task, query, key);
        let keys = select(key.dims.z, task.keys, segmented);
        let tokens = select(
            query.dims.z,
            select(keys, task.tokens, task.queries != NO_VALUE),
            packed,
        );
        let groups = query.dims.x / key.dims.x;
        let plane = coordinates(
            task.first,
            uvec4(query.dims.x, query.dims.y, query.dims.z, 1u32),
        );
        let head = plane.x / groups;
        let mut query_plane = plane.x * query.strides.x + plane.y * query.strides.y;
        let mut key_plane = head * key.strides.x + plane.y * key.strides.y;
        let mut value_plane = head * value.strides.x + plane.y * value.strides.y;
        let mut output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let mut statistic_plane = plane.x * statistic.strides.x + plane.y * statistic.strides.y;
        let mut start = 0u32;
        let mut query_start = 0u32;
        let mut origin = 0u32;
        if segmented {
            let offsets = values[task.segment];
            start = axis_start(offsets, task.plane, key.dims.z + 1u32, kind::ATTENTION);
            let rows = axis_rows(offsets, task.plane, key.dims.z + 1u32, kind::ATTENTION);
            key_plane = start * key.strides.z;
            value_plane = start * value.strides.z;
            if packed {
                query_start = start;
                if task.queries != NO_VALUE {
                    let queries = values[task.queries];
                    query_start =
                        axis_start(queries, task.plane, query.dims.z + 1u32, kind::ATTENTION);
                    if task.tokens > rows && task.origin == NO_VALUE {
                        refuse(kind::ATTENTION, refusal::EXTENT, 0u32);
                    }
                    origin = rows - min(rows, task.tokens);
                }
                query_plane = query_start * query.strides.z;
                output_plane = query_start * output.strides.z;
                statistic_plane = query_start * statistic.strides.z;
            }
        }
        let slice = lid % ATTN_SLICES;
        let unit = lid / ATTN_SLICES;
        let row = plane.z + unit;
        let inside = unit < task.count;
        let causal = task.slot == 1u32;
        let cursor_head = select(head, plane.x, segmented);
        if task.origin != NO_VALUE {
            let cursor = block_origin(task, cursor_head, plane.y, keys, tokens);
            origin = cursor;
        }
        let position = origin + row;
        let written = origin + tokens;
        let reached = causal && task.origin == NO_VALUE && task.queries == NO_VALUE;
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
            task.reach > 0u32 && task.origin == NO_VALUE && task.queries == NO_VALUE,
        );
        let first_block = window / ATTN_KEYS;
        let start_depth = slice * ATTN_SLICE;
        let mut queries = scalar_array(0.0, ATTN_SLICE);
        let mut accumulated = scalar_array(0.0, ATTN_SLICE);
        let mut weights = scalar_array(0.0, ATTN_KEYS);
        let mut largest = max_identity();
        let mut total = 0.0;
        if inside {
            for at in unroll(0u32, ATTN_SLICE, 1u32) {
                let depth = start_depth + at;
                let held = depth < ATTN_WIDTH;
                queries[at] = select(
                    0.0,
                    fetch(
                        query,
                        select(
                            0u32,
                            query_plane + row * query.strides.z + depth * query.strides.w,
                            held,
                        ),
                    ),
                    held,
                );
            }
        }
        for block in stride(first_block, walked, 1u32) {
            workgroup_barrier();
            template_stage_attention(lid, block, key_plane, value_plane, key, value, keys, keys);
            workgroup_barrier();
            if inside {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let mut score = 0.0;
                    for at in unroll(0u32, ATTN_SLICE, 1u32) {
                        let depth = start_depth + at;
                        score = score
                            + select(
                                0.0,
                                queries[at]
                                    * scratch
                                        [(column * ATTN_WIDTH + min(depth, ATTN_WIDTH - 1u32))],
                                depth < ATTN_WIDTH,
                            );
                    }
                    weights[column] = score;
                }
            }
            if ATTN_SLICES > 1u32 {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    scratch[(SCRATCH_ATTENTION_REDUCE
                        + (unit * ATTN_KEYS + column) * ATTN_REDUCTIONS * ATTN_SLICES)
                        + slice] = weights[column];
                }
                workgroup_barrier();
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let mut sum = 0.0;
                    for other in unroll(0u32, ATTN_SLICES, 1u32) {
                        sum = sum
                            + scratch[SCRATCH_ATTENTION_REDUCE
                                + (unit * ATTN_KEYS + column) * ATTN_REDUCTIONS * ATTN_SLICES
                                + other];
                    }
                    weights[column] = sum;
                }
            }
            if inside {
                let mut block_largest = max_identity();
                let mut block_keys = 0u32;
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let at = block * ATTN_KEYS + column;
                    let seen = visible(at, keys, written, position, task.reach, causal);
                    block_keys = block_keys + select(0u32, 1u32, seen);
                    let weight = select(max_identity(), weights[column] * task.param, seen);
                    weights[column] = weight;
                    block_largest = max(block_largest, weight);
                }
                let next = max(largest, block_largest);
                let rescale = select(1.0, softmax_exp(largest - next), largest != max_identity());
                largest = next;
                total = total * rescale;
                for at in unroll(0u32, ATTN_SLICE, 1u32) {
                    accumulated[at] = accumulated[at] * rescale;
                }
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let weight = select(
                        0.0,
                        softmax_exp(weights[column] - largest),
                        block_keys > 0u32,
                    );
                    total = total + weight;
                    for at in unroll(0u32, ATTN_SLICE, 1u32) {
                        let depth = start_depth + at;
                        accumulated[at] = accumulated[at]
                            + select(
                                0.0,
                                weight
                                    * scratch[SCRATCH_ATTENTION_RIGHT
                                        + (column * ATTN_WIDTH + min(depth, ATTN_WIDTH - 1u32))],
                                depth < ATTN_WIDTH,
                            );
                    }
                }
            }
        }
        if inside {
            let carries_keys = total > 0.0;
            let normalized = select(0.0, 1.0 / total, carries_keys);
            let at = uvec4(
                plane.x,
                plane.y,
                select(row, query_start + row, packed),
                0u32,
            );
            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                let walked = start_depth + depth;
                if walked < ATTN_WIDTH {
                    publish(
                        output,
                        output_plane + row * output.strides.z + walked * output.strides.w,
                        chained(
                            task,
                            at + uvec4(0u32, 0u32, 0u32, walked),
                            accumulated[depth] * normalized,
                        ),
                    );
                }
            }
            publish(
                statistic,
                statistic_plane + row * statistic.strides.z,
                select(0.0, largest + positive_log(total), carries_keys),
            );
        }
    }

    fn template_attention_query_grad(task: Task, lid: u32) {
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
        let segmented = task.segment != NO_VALUE;
        let packed = walks_the_rows_its_offsets_close(task, query, key);
        let keys = select(key.dims.z, task.keys, segmented);
        let tokens = select(
            query.dims.z,
            select(keys, task.tokens, task.queries != NO_VALUE),
            packed,
        );
        let groups = query.dims.x / key.dims.x;
        let plane = coordinates(
            task.first,
            uvec4(query.dims.x, query.dims.y, query.dims.z, 1u32),
        );
        let head = plane.x / groups;
        let mut query_plane = plane.x * query.strides.x + plane.y * query.strides.y;
        let mut key_plane = head * key.strides.x + plane.y * key.strides.y;
        let mut value_plane = head * value.strides.x + plane.y * value.strides.y;
        let mut gradient_plane = plane.x * gradient.strides.x + plane.y * gradient.strides.y;
        let mut output_grad_plane =
            plane.x * output_grad.strides.x + plane.y * output_grad.strides.y;
        let mut statistic_plane = plane.x * statistic.strides.x + plane.y * statistic.strides.y;
        let mut output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let mut start = 0u32;
        let mut query_start = 0u32;
        let mut origin = 0u32;
        if segmented {
            let offsets = values[task.segment];
            start = axis_start(
                offsets,
                task.plane,
                key.dims.z + 1u32,
                kind::ATTENTION_QUERY_GRAD,
            );
            let rows = axis_rows(
                offsets,
                task.plane,
                key.dims.z + 1u32,
                kind::ATTENTION_QUERY_GRAD,
            );
            key_plane = start * key.strides.z;
            value_plane = start * value.strides.z;
            if packed {
                query_start = start;
                if task.queries != NO_VALUE {
                    let queries = values[task.queries];
                    query_start = axis_start(
                        queries,
                        task.plane,
                        query.dims.z + 1u32,
                        kind::ATTENTION_QUERY_GRAD,
                    );
                    if task.tokens > rows && task.origin == NO_VALUE {
                        refuse(kind::ATTENTION_QUERY_GRAD, refusal::EXTENT, 0u32);
                    }
                    origin = rows - min(rows, task.tokens);
                }
                query_plane = query_start * query.strides.z;
                gradient_plane = query_start * gradient.strides.z;
                output_grad_plane = query_start * output_grad.strides.z;
                statistic_plane = query_start * statistic.strides.z;
                output_plane = query_start * output.strides.z;
            }
        }
        let slice = lid % ATTN_SLICES;
        let unit = lid / ATTN_SLICES;
        let row = plane.z + unit;
        let inside = unit < task.count;
        let causal = task.slot == 1u32;
        let cursor_head = select(head, plane.x, segmented);
        if task.origin != NO_VALUE {
            let cursor = block_origin(task, cursor_head, plane.y, keys, tokens);
            origin = cursor;
        }
        let position = origin + row;
        let written = origin + tokens;
        let reached = causal && task.origin == NO_VALUE && task.queries == NO_VALUE;
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
            task.reach > 0u32 && task.origin == NO_VALUE && task.queries == NO_VALUE,
        );
        let first_block = window / ATTN_KEYS;
        let start_depth = slice * ATTN_SLICE;
        let mut queries = scalar_array(0.0, ATTN_SLICE);
        let mut gradients = scalar_array(0.0, ATTN_SLICE);
        let mut accumulated = scalar_array(0.0, ATTN_SLICE);
        let mut weights = scalar_array(0.0, ATTN_KEYS);
        let mut partial = scalar_array(0.0, ATTN_KEYS);
        let mut row_dot = 0.0;
        let mut normalizer = 0.0;
        if inside {
            for at in unroll(0u32, ATTN_SLICE, 1u32) {
                let depth = start_depth + at;
                let held = depth < ATTN_WIDTH;
                queries[at] = select(
                    0.0,
                    fetch(
                        query,
                        select(
                            0u32,
                            query_plane + row * query.strides.z + depth * query.strides.w,
                            held,
                        ),
                    ),
                    held,
                );
                gradients[at] = select(
                    0.0,
                    fetch(
                        gradient,
                        select(
                            0u32,
                            gradient_plane + row * gradient.strides.z + depth * gradient.strides.w,
                            held,
                        ),
                    ),
                    held,
                );
                row_dot = row_dot
                    + select(
                        0.0,
                        gradients[at]
                            * fetch(
                                output_grad,
                                select(
                                    0u32,
                                    output_grad_plane
                                        + row * output_grad.strides.z
                                        + depth * output_grad.strides.w,
                                    held,
                                ),
                            ),
                        held,
                    );
            }
            normalizer = fetch(statistic, statistic_plane + row * statistic.strides.z);
        }
        if ATTN_SLICES > 1u32 {
            scratch[(SCRATCH_ATTENTION_REDUCE
                + (unit * ATTN_KEYS + 0u32) * ATTN_REDUCTIONS * ATTN_SLICES)
                + slice] = row_dot;
            workgroup_barrier();
            let mut sum = 0.0;
            for other in unroll(0u32, ATTN_SLICES, 1u32) {
                sum = sum
                    + scratch[(SCRATCH_ATTENTION_REDUCE
                        + (unit * ATTN_KEYS + 0u32) * ATTN_REDUCTIONS * ATTN_SLICES)
                        + other];
            }
            row_dot = sum;
        }
        for block in stride(first_block, walked, 1u32) {
            workgroup_barrier();
            template_stage_attention(lid, block, key_plane, value_plane, key, value, keys, keys);
            workgroup_barrier();
            if inside {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let mut weighted = 0.0;
                    for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                        let walked = start_depth + depth;
                        weighted = weighted
                            + select(
                                0.0,
                                gradients[depth]
                                    * scratch[SCRATCH_ATTENTION_RIGHT
                                        + (column * ATTN_WIDTH + min(walked, ATTN_WIDTH - 1u32))],
                                walked < ATTN_WIDTH,
                            );
                    }
                    partial[column] = weighted;
                }
            }
            if ATTN_SLICES > 1u32 {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    scratch[(SCRATCH_ATTENTION_REDUCE
                        + (unit * ATTN_KEYS + column) * ATTN_REDUCTIONS * ATTN_SLICES)
                        + slice] = partial[column];
                }
                workgroup_barrier();
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let mut sum = 0.0;
                    for other in unroll(0u32, ATTN_SLICES, 1u32) {
                        sum = sum
                            + scratch[SCRATCH_ATTENTION_REDUCE
                                + (unit * ATTN_KEYS + column) * ATTN_REDUCTIONS * ATTN_SLICES
                                + other];
                    }
                    partial[column] = sum;
                }
            }
            if inside {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let mut score = 0.0;
                    for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                        let walked = start_depth + depth;
                        score = score
                            + select(
                                0.0,
                                queries[depth]
                                    * scratch
                                        [(column * ATTN_WIDTH + min(walked, ATTN_WIDTH - 1u32))],
                                walked < ATTN_WIDTH,
                            );
                    }
                    weights[column] = score;
                }
            }
            if ATTN_SLICES > 1u32 {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    scratch[(SCRATCH_ATTENTION_REDUCE
                        + (unit * ATTN_KEYS + column) * ATTN_REDUCTIONS * ATTN_SLICES)
                        + slice] = weights[column];
                }
                workgroup_barrier();
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let mut sum = 0.0;
                    for other in unroll(0u32, ATTN_SLICES, 1u32) {
                        sum = sum
                            + scratch[SCRATCH_ATTENTION_REDUCE
                                + (unit * ATTN_KEYS + column) * ATTN_REDUCTIONS * ATTN_SLICES
                                + other];
                    }
                    weights[column] = sum;
                }
            }
            if inside {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let at = block * ATTN_KEYS + column;
                    let weight = select(
                        0.0,
                        softmax_exp(weights[column] * task.param - normalizer),
                        visible(at, keys, written, position, task.reach, causal),
                    );
                    let scored = weight * (partial[column] - row_dot) * task.param;
                    for at in unroll(0u32, ATTN_SLICE, 1u32) {
                        let depth = start_depth + at;
                        accumulated[at] = accumulated[at]
                            + select(
                                0.0,
                                scored
                                    * scratch
                                        [(column * ATTN_WIDTH + min(depth, ATTN_WIDTH - 1u32))],
                                depth < ATTN_WIDTH,
                            );
                    }
                }
            }
        }
        if inside {
            let at = uvec4(
                plane.x,
                plane.y,
                select(row, query_start + row, packed),
                0u32,
            );
            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                let walked = start_depth + depth;
                if walked < ATTN_WIDTH {
                    publish(
                        output,
                        output_plane + row * output.strides.z + walked * output.strides.w,
                        chained(
                            task,
                            at + uvec4(0u32, 0u32, 0u32, walked),
                            accumulated[depth],
                        ),
                    );
                }
            }
        }
    }

    fn template_attention_key_grad(task: Task, lid: u32) {
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
        let segmented = task.segment != NO_VALUE;
        let packed = walks_the_rows_its_offsets_close(task, query, key);
        let keys = select(key.dims.z, task.keys, segmented);
        let tokens = select(
            query.dims.z,
            select(keys, task.tokens, task.queries != NO_VALUE),
            packed,
        );
        let groups = query.dims.x / key.dims.x;
        let plane = coordinates(task.first, uvec4(key.dims.x, key.dims.y, key.dims.z, 1u32));
        let mut key_plane = plane.x * key.strides.x + plane.y * key.strides.y;
        let mut value_plane = plane.x * value.strides.x + plane.y * value.strides.y;
        let mut output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let mut head = plane.x;
        let mut batch = plane.y;
        let mut start = 0u32;
        let mut query_start = 0u32;
        let mut origin = 0u32;
        if segmented {
            head = task.plane / query.dims.y;
            batch = task.plane % query.dims.y;
            let offsets = values[task.segment];
            start = axis_start(
                offsets,
                task.plane,
                key.dims.z + 1u32,
                kind::ATTENTION_KEY_GRAD,
            );
            let rows = axis_rows(
                offsets,
                task.plane,
                key.dims.z + 1u32,
                kind::ATTENTION_KEY_GRAD,
            );
            key_plane = start * key.strides.z;
            value_plane = start * value.strides.z;
            output_plane = start * output.strides.z;
            query_start = start;
            if task.queries != NO_VALUE {
                let queries = values[task.queries];
                query_start = axis_start(
                    queries,
                    task.plane,
                    query.dims.z + 1u32,
                    kind::ATTENTION_KEY_GRAD,
                );
                if task.tokens > rows && task.origin == NO_VALUE {
                    refuse(kind::ATTENTION_KEY_GRAD, refusal::EXTENT, 0u32);
                }
                origin = rows - min(rows, task.tokens);
            }
        }
        let slice = lid % ATTN_SLICES;
        let unit = lid / ATTN_SLICES;
        let column = plane.z + unit;
        let inside = unit < task.count;
        let causal = task.slot == 1u32;
        if task.origin != NO_VALUE {
            let cursor = block_origin(task, head, batch, keys, tokens);
            origin = cursor;
        }
        let heads = select(groups, 1u32, segmented);
        let blocks = (tokens + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let first = select(
            0u32,
            plane.z / ATTN_KEYS,
            causal && task.origin == NO_VALUE && task.queries == NO_VALUE,
        );
        let start_depth = slice * ATTN_SLICE;
        let mut keys_row = scalar_array(0.0, ATTN_SLICE);
        let mut values_row = scalar_array(0.0, ATTN_SLICE);
        let mut accumulated = scalar_array(0.0, ATTN_SLICE);
        let mut weights = scalar_array(0.0, ATTN_KEYS);
        let mut partial = scalar_array(0.0, ATTN_KEYS);
        if inside {
            for at in unroll(0u32, ATTN_SLICE, 1u32) {
                let depth = start_depth + at;
                let held = depth < ATTN_WIDTH;
                keys_row[at] = select(
                    0.0,
                    fetch(
                        key,
                        select(
                            0u32,
                            key_plane + column * key.strides.z + depth * key.strides.w,
                            held,
                        ),
                    ),
                    held,
                );
                values_row[at] = select(
                    0.0,
                    fetch(
                        value,
                        select(
                            0u32,
                            value_plane + column * value.strides.z + depth * value.strides.w,
                            held,
                        ),
                    ),
                    held,
                );
            }
        }
        for group in stride(0u32, heads, 1u32) {
            let plane_head = select(plane.x * groups + group, head, segmented);
            let plane_batch = select(plane.y, batch, segmented);
            let query_plane = select(
                plane_head * query.strides.x + plane_batch * query.strides.y,
                query_start * query.strides.z,
                packed,
            );
            let gradient_plane = select(
                plane_head * gradient.strides.x + plane_batch * gradient.strides.y,
                query_start * gradient.strides.z,
                packed,
            );
            let output_grad_plane = select(
                plane_head * output_grad.strides.x + plane_batch * output_grad.strides.y,
                query_start * output_grad.strides.z,
                packed,
            );
            let statistic_plane = select(
                plane_head * statistic.strides.x + plane_batch * statistic.strides.y,
                query_start * statistic.strides.z,
                packed,
            );
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
                        if at < tokens {
                            let mut difference = 0.0;
                            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                                let walked = start_depth + depth;
                                let held = walked < ATTN_WIDTH;
                                let forward = fetch(
                                    output_grad,
                                    select(
                                        0u32,
                                        output_grad_plane
                                            + at * output_grad.strides.z
                                            + walked * output_grad.strides.w,
                                        held,
                                    ),
                                );
                                difference = difference
                                    + select(
                                        0.0,
                                        scratch[SCRATCH_ATTENTION_RIGHT
                                            + (step * ATTN_WIDTH + min(walked, ATTN_WIDTH - 1u32))]
                                            * (values_row[depth] - forward),
                                        held,
                                    );
                            }
                            partial[step] = difference;
                        }
                    }
                }
                if ATTN_SLICES > 1u32 {
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            scratch[(SCRATCH_ATTENTION_REDUCE
                                + (unit * ATTN_KEYS + step) * ATTN_REDUCTIONS * ATTN_SLICES)
                                + slice] = partial[step];
                        }
                    }
                    workgroup_barrier();
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            let mut sum = 0.0;
                            for other in unroll(0u32, ATTN_SLICES, 1u32) {
                                sum = sum
                                    + scratch[SCRATCH_ATTENTION_REDUCE
                                        + (unit * ATTN_KEYS + step)
                                            * ATTN_REDUCTIONS
                                            * ATTN_SLICES
                                        + other];
                            }
                            partial[step] = sum;
                        }
                    }
                }
                if inside {
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            let mut score = 0.0;
                            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                                let walked = start_depth + depth;
                                score = score
                                    + select(
                                        0.0,
                                        scratch
                                            [(step * ATTN_WIDTH + min(walked, ATTN_WIDTH - 1u32))]
                                            * keys_row[depth],
                                        walked < ATTN_WIDTH,
                                    );
                            }
                            weights[step] = score;
                        }
                    }
                }
                if ATTN_SLICES > 1u32 {
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            scratch[(SCRATCH_ATTENTION_REDUCE
                                + (unit * ATTN_KEYS + step) * ATTN_REDUCTIONS * ATTN_SLICES)
                                + slice] = weights[step];
                        }
                    }
                    workgroup_barrier();
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            let mut sum = 0.0;
                            for other in unroll(0u32, ATTN_SLICES, 1u32) {
                                sum = sum
                                    + scratch[SCRATCH_ATTENTION_REDUCE
                                        + (unit * ATTN_KEYS + step)
                                            * ATTN_REDUCTIONS
                                            * ATTN_SLICES
                                        + other];
                            }
                            weights[step] = sum;
                        }
                    }
                }
                if inside {
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            let weight = select(
                                0.0,
                                softmax_exp(
                                    weights[step] * task.param
                                        - fetch(
                                            statistic,
                                            statistic_plane + at * statistic.strides.z,
                                        ),
                                ),
                                attended(at, tokens, keys, column, origin, task.reach, causal),
                            );
                            let scored = weight * partial[step] * task.param;
                            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                                let walked = start_depth + depth;
                                accumulated[depth] = accumulated[depth]
                                    + select(
                                        0.0,
                                        scored
                                            * scratch[(step * ATTN_WIDTH
                                                + min(walked, ATTN_WIDTH - 1u32))],
                                        walked < ATTN_WIDTH,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if inside {
            let at = uvec4(
                select(plane.x, 0u32, segmented),
                select(plane.y, 0u32, segmented),
                select(column, start + column, segmented),
                0u32,
            );
            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                let walked = start_depth + depth;
                if walked < ATTN_WIDTH {
                    publish(
                        output,
                        output_plane + column * output.strides.z + walked * output.strides.w,
                        chained(
                            task,
                            at + uvec4(0u32, 0u32, 0u32, walked),
                            accumulated[depth],
                        ),
                    );
                }
            }
        }
    }

    fn template_attention_value_grad(task: Task, lid: u32) {
        if task.count == 0u32 {
            return;
        }
        let query = values[task.a];
        let key = values[task.b];
        let gradient = values[task.d];
        let statistic = values[task.f];
        let output = values[task.out];
        let segmented = task.segment != NO_VALUE;
        let packed = walks_the_rows_its_offsets_close(task, query, key);
        let keys = select(key.dims.z, task.keys, segmented);
        let tokens = select(
            query.dims.z,
            select(keys, task.tokens, task.queries != NO_VALUE),
            packed,
        );
        let groups = query.dims.x / key.dims.x;
        let plane = coordinates(task.first, uvec4(key.dims.x, key.dims.y, key.dims.z, 1u32));
        let mut key_plane = plane.x * key.strides.x + plane.y * key.strides.y;
        let mut output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let mut head = plane.x;
        let mut batch = plane.y;
        let mut start = 0u32;
        let mut query_start = 0u32;
        let mut origin = 0u32;
        if segmented {
            head = task.plane / query.dims.y;
            batch = task.plane % query.dims.y;
            let offsets = values[task.segment];
            start = axis_start(
                offsets,
                task.plane,
                key.dims.z + 1u32,
                kind::ATTENTION_VALUE_GRAD,
            );
            let rows = axis_rows(
                offsets,
                task.plane,
                key.dims.z + 1u32,
                kind::ATTENTION_VALUE_GRAD,
            );
            key_plane = start * key.strides.z;
            output_plane = start * output.strides.z;
            query_start = start;
            if task.queries != NO_VALUE {
                let queries = values[task.queries];
                query_start = axis_start(
                    queries,
                    task.plane,
                    query.dims.z + 1u32,
                    kind::ATTENTION_VALUE_GRAD,
                );
                if task.tokens > rows && task.origin == NO_VALUE {
                    refuse(kind::ATTENTION_VALUE_GRAD, refusal::EXTENT, 0u32);
                }
                origin = rows - min(rows, task.tokens);
            }
        }
        let slice = lid % ATTN_SLICES;
        let unit = lid / ATTN_SLICES;
        let column = plane.z + unit;
        let inside = unit < task.count;
        let causal = task.slot == 1u32;
        if task.origin != NO_VALUE {
            let cursor = block_origin(task, head, batch, keys, tokens);
            origin = cursor;
        }
        let heads = select(groups, 1u32, segmented);
        let blocks = (tokens + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let first = select(
            0u32,
            plane.z / ATTN_KEYS,
            causal && task.origin == NO_VALUE && task.queries == NO_VALUE,
        );
        let start_depth = slice * ATTN_SLICE;
        let mut keys_row = scalar_array(0.0, ATTN_SLICE);
        let mut accumulated = scalar_array(0.0, ATTN_SLICE);
        let mut weights = scalar_array(0.0, ATTN_KEYS);
        if inside {
            for at in unroll(0u32, ATTN_SLICE, 1u32) {
                let depth = start_depth + at;
                let held = depth < ATTN_WIDTH;
                keys_row[at] = select(
                    0.0,
                    fetch(
                        key,
                        select(
                            0u32,
                            key_plane + column * key.strides.z + depth * key.strides.w,
                            held,
                        ),
                    ),
                    held,
                );
            }
        }
        for group in stride(0u32, heads, 1u32) {
            let plane_head = select(plane.x * groups + group, head, segmented);
            let plane_batch = select(plane.y, batch, segmented);
            let query_plane = select(
                plane_head * query.strides.x + plane_batch * query.strides.y,
                query_start * query.strides.z,
                packed,
            );
            let gradient_plane = select(
                plane_head * gradient.strides.x + plane_batch * gradient.strides.y,
                query_start * gradient.strides.z,
                packed,
            );
            let statistic_plane = select(
                plane_head * statistic.strides.x + plane_batch * statistic.strides.y,
                query_start * statistic.strides.z,
                packed,
            );
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
                        if at < tokens {
                            let mut score = 0.0;
                            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                                let walked = start_depth + depth;
                                score = score
                                    + select(
                                        0.0,
                                        scratch
                                            [(step * ATTN_WIDTH + min(walked, ATTN_WIDTH - 1u32))]
                                            * keys_row[depth],
                                        walked < ATTN_WIDTH,
                                    );
                            }
                            weights[step] = score;
                        }
                    }
                }
                if ATTN_SLICES > 1u32 {
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            scratch[(SCRATCH_ATTENTION_REDUCE
                                + (unit * ATTN_KEYS + step) * ATTN_REDUCTIONS * ATTN_SLICES)
                                + slice] = weights[step];
                        }
                    }
                    workgroup_barrier();
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            let mut sum = 0.0;
                            for other in unroll(0u32, ATTN_SLICES, 1u32) {
                                sum = sum
                                    + scratch[SCRATCH_ATTENTION_REDUCE
                                        + (unit * ATTN_KEYS + step)
                                            * ATTN_REDUCTIONS
                                            * ATTN_SLICES
                                        + other];
                            }
                            weights[step] = sum;
                        }
                    }
                }
                if inside {
                    for step in unroll(0u32, ATTN_KEYS, 1u32) {
                        let at = block * ATTN_KEYS + step;
                        if at < tokens {
                            let weight = select(
                                0.0,
                                softmax_exp(
                                    weights[step] * task.param
                                        - fetch(
                                            statistic,
                                            statistic_plane + at * statistic.strides.z,
                                        ),
                                ),
                                attended(at, tokens, keys, column, origin, task.reach, causal),
                            );
                            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                                let walked = start_depth + depth;
                                accumulated[depth] = accumulated[depth]
                                    + select(
                                        0.0,
                                        weight
                                            * scratch[SCRATCH_ATTENTION_RIGHT
                                                + (step * ATTN_WIDTH
                                                    + min(walked, ATTN_WIDTH - 1u32))],
                                        walked < ATTN_WIDTH,
                                    );
                            }
                        }
                    }
                }
            }
        }
        if inside {
            let at = uvec4(
                select(plane.x, 0u32, segmented),
                select(plane.y, 0u32, segmented),
                select(column, start + column, segmented),
                0u32,
            );
            for depth in unroll(0u32, ATTN_SLICE, 1u32) {
                let walked = start_depth + depth;
                if walked < ATTN_WIDTH {
                    publish(
                        output,
                        output_plane + column * output.strides.z + walked * output.strides.w,
                        chained(
                            task,
                            at + uvec4(0u32, 0u32, 0u32, walked),
                            accumulated[depth],
                        ),
                    );
                }
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
