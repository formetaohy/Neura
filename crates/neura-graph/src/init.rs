#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Fan {
    pub inputs: u32,
    pub outputs: u32,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Init {
    Zero,
    Constant(f32),
    Uniform { low: f32, high: f32 },
    Normal { mean: f32, deviation: f32 },
    Xavier { gain: f32 },
    Kaiming { gain: f32 },
}

impl Init {
    pub fn spread(self, fan: Fan) -> Self {
        match self {
            Self::Xavier { gain } => {
                assert!(
                    fan.inputs > 0 && fan.outputs > 0,
                    "a Xavier init of {} inputs and {} outputs spreads no gradient",
                    fan.inputs,
                    fan.outputs,
                );
                assert!(
                    gain > 0.0 && gain.is_finite(),
                    "a Xavier init of gain {gain} scales every weight to nothing",
                );
                let reach = gain * (6.0 / (fan.inputs + fan.outputs) as f32).sqrt();
                Self::Uniform {
                    low: -reach,
                    high: reach,
                }
            }
            Self::Kaiming { gain } => {
                assert!(
                    fan.inputs > 0,
                    "a Kaiming init of {} inputs spreads no gradient",
                    fan.inputs,
                );
                assert!(
                    gain > 0.0 && gain.is_finite(),
                    "a Kaiming init of gain {gain} scales every weight to nothing",
                );
                let reach = gain * (3.0 / fan.inputs as f32).sqrt();
                Self::Uniform {
                    low: -reach,
                    high: reach,
                }
            }
            other => other,
        }
    }

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
            Self::Normal { mean, deviation } => {
                assert!(
                    mean.is_finite() && deviation > 0.0 && deviation.is_finite(),
                    "a normal init of mean {mean} and deviation {deviation} draws every number at one point",
                );
                (0..elements)
                    .map(|_| mean + deviation * normal(entropy))
                    .collect()
            }
            Self::Xavier { .. } | Self::Kaiming { .. } => panic!(
                "a fan aware init spreads its numbers across the fan of a layer, and {elements} numbers carry no fan",
            ),
        }
    }
}

pub fn unit(entropy: &mut u32) -> f32 {
    *entropy ^= *entropy << 13;
    *entropy ^= *entropy >> 17;
    *entropy ^= *entropy << 5;
    (*entropy >> 8) as f32 / (1u32 << 24) as f32
}

fn normal(entropy: &mut u32) -> f32 {
    let radius = (-2.0 * unit(entropy).max(f32::MIN_POSITIVE).ln()).sqrt();
    radius * (std::f32::consts::TAU * unit(entropy)).cos()
}
