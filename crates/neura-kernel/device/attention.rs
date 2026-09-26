#[neura_compiler::module]
mod source {
    fn visible(at: u32, tokens: u32, row: u32, causal: bool) -> bool {
        if at >= tokens {
            return false;
        }
        if causal && at > row {
            return false;
        }
        return true;
    }

    fn attended(at: u32, tokens: u32, column: u32, causal: bool) -> bool {
        if at >= tokens {
            return false;
        }
        if causal && at < column {
            return false;
        }
        return true;
    }

    fn template_stage_attention(
        lid: u32,
        block: u32,
        left_plane: u32,
        right_plane: u32,
        left: Value,
        right: Value,
    ) {
        for unit in stride(lid, ATTN_KEYS * ATTN_WIDTH, WORKGROUP_SIZE) {
            let column = unit / ATTN_WIDTH;
            let depth = unit % ATTN_WIDTH;
            let at = block * ATTN_KEYS + column;
            let inside = at < left.dims.z;
            let address = select(
                0u32,
                left_plane + at * left.strides.z + depth * left.strides.w,
                inside,
            );
            attention_left[unit] = select(0.0, fetch(left, address), inside);
        }
        for unit in stride(lid, ATTN_KEYS * ATTN_WIDTH, WORKGROUP_SIZE) {
            let column = unit / ATTN_WIDTH;
            let depth = unit % ATTN_WIDTH;
            let at = block * ATTN_KEYS + column;
            let inside = at < right.dims.z;
            let address = select(
                0u32,
                right_plane + at * right.strides.z + depth * right.strides.w,
                inside,
            );
            attention_right[unit] = select(0.0, fetch(right, address), inside);
        }
    }

    fn template_attention_forward(task: Task, lid: u32) {
        let query = values[task.a];
        let key = values[task.b];
        let value = values[task.c];
        let output = values[task.out];
        let statistic = values[task.extra];
        let tokens = query.dims.z;
        let keys = key.dims.z;
        let plane = coordinates(
            task.first,
            uvec4(query.dims.x, query.dims.y, query.dims.z, 1u32),
        );
        let query_plane = plane.x * query.strides.x + plane.y * query.strides.y;
        let key_plane = plane.x * key.strides.x + plane.y * key.strides.y;
        let value_plane = plane.x * value.strides.x + plane.y * value.strides.y;
        let output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let statistic_plane = plane.x * statistic.strides.x + plane.y * statistic.strides.y;
        let row = plane.z + lid;
        let inside = lid < task.count;
        let causal = task.slot == 1u32;
        let blocks = (keys + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let walked = select(
            blocks,
            (plane.z + task.count + ATTN_KEYS - 1u32) / ATTN_KEYS,
            causal,
        );
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
        for block in stride(0u32, walked, 1u32) {
            workgroup_barrier();
            template_stage_attention(lid, block, key_plane, value_plane, key, value);
            workgroup_barrier();
            if inside {
                let mut block_largest = -3.4028235e38;
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let at = block * ATTN_KEYS + column;
                    let mut score = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        score =
                            score + queries[depth] * attention_left[column * ATTN_WIDTH + depth];
                    }
                    weights[column] = select(
                        -3.4028235e38,
                        score * task.param,
                        visible(at, keys, row, causal),
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
                            + weights[column] * attention_right[column * ATTN_WIDTH + depth];
                    }
                    accumulated[depth] = carried;
                }
            }
        }
        if inside {
            let normalized = 1.0 / total;
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
                largest + log(total),
            );
        }
    }

    fn template_attention_query_grad(task: Task, lid: u32) {
        let query = values[task.a];
        let key = values[task.b];
        let value = values[task.c];
        let gradient = values[task.d];
        let output_grad = values[task.e];
        let statistic = values[task.f];
        let output = values[task.out];
        let tokens = query.dims.z;
        let keys = key.dims.z;
        let plane = coordinates(
            task.first,
            uvec4(query.dims.x, query.dims.y, query.dims.z, 1u32),
        );
        let query_plane = plane.x * query.strides.x + plane.y * query.strides.y;
        let key_plane = plane.x * key.strides.x + plane.y * key.strides.y;
        let value_plane = plane.x * value.strides.x + plane.y * value.strides.y;
        let gradient_plane = plane.x * gradient.strides.x + plane.y * gradient.strides.y;
        let output_grad_plane = plane.x * output_grad.strides.x + plane.y * output_grad.strides.y;
        let statistic_plane = plane.x * statistic.strides.x + plane.y * statistic.strides.y;
        let output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let row = plane.z + lid;
        let inside = lid < task.count;
        let causal = task.slot == 1u32;
        let blocks = (keys + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let walked = select(
            blocks,
            (plane.z + task.count + ATTN_KEYS - 1u32) / ATTN_KEYS,
            causal,
        );
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
        for block in stride(0u32, walked, 1u32) {
            workgroup_barrier();
            template_stage_attention(lid, block, key_plane, value_plane, key, value);
            workgroup_barrier();
            if inside {
                for column in unroll(0u32, ATTN_KEYS, 1u32) {
                    let at = block * ATTN_KEYS + column;
                    let mut score = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        score =
                            score + queries[depth] * attention_left[column * ATTN_WIDTH + depth];
                    }
                    let weight = select(
                        0.0,
                        exp(score * task.param - normalizer),
                        visible(at, keys, row, causal),
                    );
                    let mut weighted = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        weighted = weighted
                            + gradients[depth] * attention_right[column * ATTN_WIDTH + depth];
                    }
                    let scored = weight * (weighted - row_dot) * task.param;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        accumulated[depth] = accumulated[depth]
                            + scored * attention_left[column * ATTN_WIDTH + depth];
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
        let query = values[task.a];
        let key = values[task.b];
        let value = values[task.c];
        let gradient = values[task.d];
        let output_grad = values[task.e];
        let statistic = values[task.f];
        let output = values[task.out];
        let tokens = query.dims.z;
        let keys = key.dims.z;
        let plane = coordinates(task.first, uvec4(key.dims.x, key.dims.y, key.dims.z, 1u32));
        let query_plane = plane.x * query.strides.x + plane.y * query.strides.y;
        let key_plane = plane.x * key.strides.x + plane.y * key.strides.y;
        let value_plane = plane.x * value.strides.x + plane.y * value.strides.y;
        let gradient_plane = plane.x * gradient.strides.x + plane.y * gradient.strides.y;
        let output_grad_plane = plane.x * output_grad.strides.x + plane.y * output_grad.strides.y;
        let statistic_plane = plane.x * statistic.strides.x + plane.y * statistic.strides.y;
        let output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let column = plane.z + lid;
        let inside = lid < task.count;
        let causal = task.slot == 1u32;
        let blocks = (tokens + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let first = select(0u32, plane.z / ATTN_KEYS, causal);
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
        for block in stride(first, blocks, 1u32) {
            workgroup_barrier();
            template_stage_attention(lid, block, query_plane, gradient_plane, query, gradient);
            workgroup_barrier();
            if inside {
                for step in unroll(0u32, ATTN_KEYS, 1u32) {
                    let at = block * ATTN_KEYS + step;
                    let mut score = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        score = score + attention_left[step * ATTN_WIDTH + depth] * keys_row[depth];
                    }
                    let weight = select(
                        0.0,
                        exp(score * task.param
                            - fetch(statistic, statistic_plane + at * statistic.strides.z)),
                        attended(at, tokens, column, causal),
                    );
                    let mut weighted = 0.0;
                    let mut row_dot = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        weighted = weighted
                            + attention_right[step * ATTN_WIDTH + depth] * values_row[depth];
                        row_dot = row_dot
                            + attention_right[step * ATTN_WIDTH + depth]
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
                            accumulated[depth] + scored * attention_left[step * ATTN_WIDTH + depth];
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
        let query = values[task.a];
        let key = values[task.b];
        let gradient = values[task.d];
        let statistic = values[task.f];
        let output = values[task.out];
        let tokens = query.dims.z;
        let plane = coordinates(task.first, uvec4(key.dims.x, key.dims.y, key.dims.z, 1u32));
        let query_plane = plane.x * query.strides.x + plane.y * query.strides.y;
        let key_plane = plane.x * key.strides.x + plane.y * key.strides.y;
        let gradient_plane = plane.x * gradient.strides.x + plane.y * gradient.strides.y;
        let statistic_plane = plane.x * statistic.strides.x + plane.y * statistic.strides.y;
        let output_plane = plane.x * output.strides.x + plane.y * output.strides.y;
        let column = plane.z + lid;
        let inside = lid < task.count;
        let causal = task.slot == 1u32;
        let blocks = (tokens + ATTN_KEYS - 1u32) / ATTN_KEYS;
        let first = select(0u32, plane.z / ATTN_KEYS, causal);
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
        for block in stride(first, blocks, 1u32) {
            workgroup_barrier();
            template_stage_attention(lid, block, query_plane, gradient_plane, query, gradient);
            workgroup_barrier();
            if inside {
                for step in unroll(0u32, ATTN_KEYS, 1u32) {
                    let at = block * ATTN_KEYS + step;
                    let mut score = 0.0;
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        score = score + attention_left[step * ATTN_WIDTH + depth] * keys_row[depth];
                    }
                    let weight = select(
                        0.0,
                        exp(score * task.param
                            - fetch(statistic, statistic_plane + at * statistic.strides.z)),
                        attended(at, tokens, column, causal),
                    );
                    for depth in unroll(0u32, ATTN_WIDTH, 1u32) {
                        accumulated[depth] = accumulated[depth]
                            + weight * attention_right[step * ATTN_WIDTH + depth];
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
            _ => refuse(kind::ATTENTION, task.geometry),
        }
    }

    fn run_attention_query_grad(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::ATTENTION_QUERY_GRAD, task.geometry),
        }
    }

    fn run_attention_key_grad(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::ATTENTION_KEY_GRAD, task.geometry),
        }
    }

    fn run_attention_value_grad(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::ATTENTION_VALUE_GRAD, task.geometry),
        }
    }
}
