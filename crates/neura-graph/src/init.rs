#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Init {
    Zero,
    Constant(f32),
    Uniform { low: f32, high: f32 },
}

impl Init {
    pub fn samples(self, elements: u32, entropy: &mut u32) -> Vec<f32> {
        match self {
            Self::Zero => vec![0.0; elements as usize],
            Self::Constant(value) => vec![value; elements as usize],
            Self::Uniform { low, high } => {
                assert!(
                    low < high,
                    "a uniform init spans {low} to {high} without a range",
                );
                (0..elements)
                    .map(|_| low + (high - low) * unit(entropy))
                    .collect()
            }
        }
    }
}

pub fn unit(entropy: &mut u32) -> f32 {
    *entropy ^= *entropy << 13;
    *entropy ^= *entropy >> 17;
    *entropy ^= *entropy << 5;
    (*entropy >> 8) as f32 / (1u32 << 24) as f32
}
