use neura_graph::{Fan, Init};

fn refuses(action: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    std::panic::catch_unwind(action).is_err()
}

fn moments(samples: &[f32]) -> (f32, f32) {
    let count = samples.len() as f32;
    let mean = samples.iter().sum::<f32>() / count;
    let deviation = (samples
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f32>()
        / count)
        .sqrt();
    (mean, deviation)
}

#[test]
fn a_normal_init_draws_the_mean_and_the_deviation_it_names() {
    let mut entropy = 0x2b7e_1516u32;
    let samples = Init::Normal {
        mean: 2.5,
        deviation: 0.75,
    }
    .samples(65_536, &mut entropy);
    let (mean, deviation) = moments(&samples);
    assert!(
        (mean - 2.5).abs() < 0.01,
        "a normal init of mean 2.5 drew a mean of {mean}",
    );
    assert!(
        (deviation - 0.75).abs() < 0.01,
        "a normal init of deviation 0.75 drew a deviation of {deviation}",
    );
    assert!(
        samples.iter().all(|value| value.is_finite()),
        "a normal init draws no infinite number",
    );
}

#[test]
fn a_fan_aware_init_spreads_the_uniform_its_fan_names() {
    let fan = Fan {
        inputs: 48,
        outputs: 96,
    };
    let xavier = (6.0 / (fan.inputs + fan.outputs) as f32).sqrt();
    let kaiming = (2.0f32).sqrt() * (3.0 / fan.inputs as f32).sqrt();
    for (init, reach) in [
        (Init::Xavier { gain: 1.0 }, xavier),
        (
            Init::Kaiming {
                gain: (2.0f32).sqrt(),
            },
            kaiming,
        ),
    ] {
        let mut spread = 0x1357_9bdfu32;
        let mut uniform = 0x1357_9bdfu32;
        assert_eq!(
            init.spread(fan).samples(512, &mut spread),
            Init::Uniform {
                low: -reach,
                high: reach,
            }
            .samples(512, &mut uniform),
            "a fan aware init spreads the uniform its fan names",
        );
    }
    let mut entropy = 0x2468_ace0u32;
    let samples = Init::Kaiming {
        gain: (2.0f32).sqrt(),
    }
    .spread(fan)
    .samples(4096, &mut entropy);
    let (mean, deviation) = moments(&samples);
    assert!(
        mean.abs() < 0.05 * kaiming,
        "a Kaiming init drew a mean of {mean} where its reach is {kaiming}",
    );
    assert!(
        (deviation - kaiming / (3.0f32).sqrt()).abs() < 0.02 * kaiming,
        "a Kaiming init of reach {kaiming} drew a deviation of {deviation}",
    );
}

#[test]
fn a_fan_aware_init_refuses_a_fan_that_spreads_nothing() {
    assert!(refuses(|| {
        Init::Xavier { gain: 1.0 }.spread(Fan {
            inputs: 0,
            outputs: 8,
        });
    }));
    assert!(refuses(|| {
        Init::Xavier { gain: 1.0 }.spread(Fan {
            inputs: 8,
            outputs: 0,
        });
    }));
    assert!(refuses(|| {
        Init::Kaiming { gain: 1.0 }.spread(Fan {
            inputs: 0,
            outputs: 8,
        });
    }));
    assert!(refuses(|| {
        Init::Xavier { gain: 0.0 }.spread(Fan {
            inputs: 8,
            outputs: 8,
        });
    }));
    assert!(refuses(|| {
        Init::Kaiming {
            gain: f32::INFINITY,
        }
        .spread(Fan {
            inputs: 8,
            outputs: 8,
        });
    }));
    assert!(refuses(|| {
        let _ = Init::Kaiming { gain: 1.0 }.samples(8, &mut 1u32);
    }));
    assert!(refuses(|| {
        let _ = Init::Normal {
            mean: 0.0,
            deviation: 0.0,
        }
        .samples(8, &mut 1u32);
    }));
    assert!(refuses(|| {
        let _ = Init::Uniform {
            low: 1.0,
            high: 1.0,
        }
        .samples(8, &mut 1u32);
    }));
}

#[test]
fn every_init_draws_the_same_numbers_from_the_same_seed() {
    let init = Init::Normal {
        mean: -0.25,
        deviation: 1.5,
    };
    let mut first = 0x0f1e_2d3cu32;
    let mut second = 0x0f1e_2d3cu32;
    let drawn = init.samples(128, &mut first);
    assert_eq!(drawn, init.samples(128, &mut second));
    let mut other = 0x0f1e_2d3cu32;
    assert_ne!(
        drawn,
        Init::Normal {
            mean: -0.25,
            deviation: 1.4,
        }
        .samples(128, &mut other),
        "two normal inits of different deviations draw different numbers",
    );
}
