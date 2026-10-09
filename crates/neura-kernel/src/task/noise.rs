use neura_compiler::Compiler;

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
}

#[neura_compiler::module]
mod device {
    fn noise_hash(seed: u32, key: u32, salt: u32) -> u32 {
        let mut hash = seed ^ (key * 0x9e3779b9u32) ^ (salt * 0x85ebca6bu32);
        hash = hash ^ (hash >> 16u32);
        hash = hash * 0x7feb352du32;
        hash = hash ^ (hash >> 15u32);
        hash = hash * 0x846ca68bu32;
        hash = hash ^ (hash >> 16u32);
        return hash;
    }

    fn unit_noise(seed: u32, key: u32, salt: u32) -> f32 {
        return (f32(noise_hash(seed, key, salt) >> 8u32) + 0.5) * (1.0 / 16777216.0);
    }

    fn normal_noise(seed: u32, key: u32) -> f32 {
        let angle = unit_noise(seed, key, 1u32);
        let radius = unit_noise(seed, key, 0u32);
        return sqrt(-2.0 * log(radius)) * cos(6.2831855 * angle);
    }

    fn noise_value(op: u32, seed: u32, key: u32) -> f32 {
        match op {
            noise::UNIFORM => {
                return unit_noise(seed, key, 0u32);
            }
            noise::NORMAL => {
                return normal_noise(seed, key);
            }
            _ => {
                refuse(kind::NOISE, refusal::OP, op);
                return 0.0;
            }
        }
    }

    fn run_noise(task: Task, lid: u32) {
        let output = values[task.out];
        let seed = bitcast_u32(fetch(values[task.a], 0u32));
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = coordinates(index, dims);
            publish(
                output,
                index,
                chained(task, at, noise_value(task.op, seed, element_key(at))),
            );
        }
    }
}
