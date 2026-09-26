#[neura_compiler::module]
mod source {
    fn run_pack_half(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let last = task.first + task.count;
        for at in stride(task.first + 2u32 * lid, last, 2u32 * WORKGROUP_SIZE) {
            let second = select(at, at + 1u32, at + 1u32 < last);
            publish_word(
                output,
                at >> 1u32,
                pack2x16float(fetch(source, at), fetch(source, second)),
            );
        }
    }

    fn run_pack_bfloat16(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let last = task.first + task.count;
        for at in stride(task.first + 2u32 * lid, last, 2u32 * WORKGROUP_SIZE) {
            let second = select(at, at + 1u32, at + 1u32 < last);
            let low = bitcast_u32(fetch(source, at));
            let high = bitcast_u32(fetch(source, second));
            let packed_low = (low + 0x7fffu32 + ((low >> 16u32) & 1u32)) >> 16u32;
            let packed_high = (high + 0x7fffu32 + ((high >> 16u32) & 1u32)) >> 16u32;
            publish_word(output, at >> 1u32, (packed_high << 16u32) | packed_low);
        }
    }

    fn run_pack(task: Task, lid: u32) {
        match task.geometry {
            _ => refuse(kind::PACK, refusal::GEOMETRY, task.geometry),
        }
    }
}
