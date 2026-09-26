#[neura_compiler::module]
mod source {
    fn refuse(kind: u32, code: u32) {
        atomic_store(&refusal[0u32], (kind << 16u32) | (code + 1u32));
    }

    fn coordinates(flat: u32, dims: uvec4) -> uvec4 {
        let w = flat % dims.w;
        let z = flat / dims.w % dims.z;
        let y = flat / (dims.w * dims.z) % dims.y;
        let x = flat / (dims.w * dims.z * dims.y);
        return uvec4(x, y, z, w);
    }

    fn read_address(at: uvec4, strides: uvec4) -> u32 {
        return at.x * strides.x + at.y * strides.y + at.z * strides.z + at.w * strides.w;
    }

    fn component(at: uvec4, axis: u32) -> u32 {
        return select(
            select(at.x, at.y, axis == 1u32),
            select(at.z, at.w, axis == 3u32),
            axis >= 2u32,
        );
    }

    fn whole_index(value: f32, rows: u32, kind: u32) -> u32 {
        if trunc(value) != value || !(value >= 0.0) || !(value < f32(rows)) {
            refuse(kind, 0u32);
            return 0u32;
        }
        return u32(value);
    }

    fn base_of(value: Value) -> u32 {
        return select(
            placement.weights,
            placement.tensors,
            value.store == store::TENSORS,
        );
    }

    fn word_of(value: Value, word: u32) -> u32 {
        return base_of(value) + value.base + word;
    }

    fn publish(value: Value, at: u32, data: f32) {
        heap[word_of(value, at)] = data;
    }

    fn publish_word(value: Value, word: u32, packed: u32) {
        heap[word_of(value, word)] = bitcast_f32(packed);
    }

    fn fetch_single(value: Value, at: u32) -> f32 {
        return heap[word_of(value, at)];
    }

    fn fetch_half(value: Value, at: u32) -> f32 {
        let pair = unpack2x16float(bitcast_u32(heap[word_of(value, at >> 1u32)]));
        return select(pair.x, pair.y, (at & 1u32) == 1u32);
    }

    fn fetch_bfloat16(value: Value, at: u32) -> f32 {
        let word = bitcast_u32(heap[word_of(value, at >> 1u32)]);
        return select(
            bitcast_f32(word << 16u32),
            bitcast_f32(word & 0xffff0000u32),
            (at & 1u32) == 1u32,
        );
    }

    fn fetch_by_element(value: Value, at: u32) -> f32 {
        match value.element {
            _ => {
                refuse(refusal::ELEMENT, value.element);
                return 0.0;
            }
        }
    }

    fn run_task(index: u32, lid: u32) {
        let task = tasks[index];
        match task.kind {
            kind::MATMUL => run_matmul(task, lid),
            kind::MATMUL_FOLD => run_matmul_fold(task, lid),
            kind::BINARY => run_binary(task, lid),
            kind::UNARY => run_unary(task, lid),
            kind::PARTIAL => run_partial(task, lid),
            kind::FILL => run_fill(task, lid),
            kind::BROADCAST => run_broadcast(task, lid),
            kind::SUM_CHUNK => run_sum_chunk(task, lid),
            kind::SUM_AXIS => run_sum_axis(task, lid),
            kind::SOFTMAX => run_softmax(task, lid),
            kind::SOFTMAX_GRAD => run_softmax_grad(task, lid),
            kind::LOG_SOFTMAX => run_log_softmax(task, lid),
            kind::LOG_SOFTMAX_GRAD => run_log_softmax_grad(task, lid),
            kind::ARGMAX => run_argmax(task, lid),
            kind::CATEGORICAL => run_categorical(task, lid),
            kind::ONE_HOT => run_one_hot(task, lid),
            kind::GATHER => run_gather(task, lid),
            kind::SCATTER => run_scatter(task, lid),
            kind::PACK => run_pack(task, lid),
            kind::CONV2D => run_conv2d(task, lid),
            kind::CONV2D_INPUT_GRAD => run_conv2d_input_grad(task, lid),
            kind::CONV2D_WEIGHT_GRAD => run_conv2d_weight_grad(task, lid),
            _ => refuse(task.kind, 0u32),
        }
    }

    #[neura_compiler::kernel]
    fn main(lid: u32, group: uvec3) {
        let segment = segments[bounds.first_segment + group.x];
        for index in stride(segment.first, segment.first + segment.count, 1u32) {
            run_task(index, lid);
            storage_barrier();
        }
    }
}
