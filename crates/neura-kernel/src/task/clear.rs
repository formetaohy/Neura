use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn clear_lane(packed: u32, lane: u32, per_word: u32) -> u32 {
        let bits = 32u32 / per_word;
        let mask = (0xffff_ffffu32 >> (32u32 - bits)) << (lane * bits);
        return packed ^ (packed & mask);
    }

    fn run_clear(task: Task, lid: u32) {
        let target = values[task.out];
        let condition = task.a;
        let per_word = elements_per_word(target.element);
        let elements = target.dims.x * target.dims.y * target.dims.z * target.dims.w;
        let source = values[condition];
        for word in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let mut packed = bitcast_u32(peek(word_of(target, word)));
            for lane in stride(0u32, per_word, 1u32) {
                let index = word * per_word + lane;
                if index < elements {
                    let at = coordinates(index, target.dims);
                    if fetch(source, read_address(at, source.strides)) != 0.0 {
                        packed = clear_lane(packed, lane, per_word);
                    }
                }
            }
            publish_word(target, word, packed);
        }
    }
}
