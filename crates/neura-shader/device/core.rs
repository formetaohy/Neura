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

    fn run_task(task: Task, lid: u32) {
        match task.kind {
            _ => refuse(task.kind, 0u32),
        }
    }

    #[neura_compiler::kernel]
    fn main(lid: u32, group: uvec3) {
        let segment = segments[bounds.first_segment + group.x];
        for index in stride(segment.first, segment.first + segment.count, 1u32) {
            run_task(tasks[index], lid);
            storage_barrier();
        }
    }
}
