use neura_graph::{Graph, Shape, Value};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Sampling {
    Greedy,
    Sampled {
        temperature: f32,
        keep: u32,
        cumulative: f32,
    },
}

impl Sampling {
    pub const GREEDY: Self = Self::Greedy;

    pub fn sampled(temperature: f32, keep: u32, cumulative: f32) -> Self {
        assert!(
            temperature > 0.0 && temperature.is_finite(),
            "a sampling temperature of {temperature} spreads no logits",
        );
        assert!(
            (1..=neura_abi::CANDIDATES).contains(&keep),
            "a sample keeps {keep} candidates where a row holds between one and {}",
            neura_abi::CANDIDATES,
        );
        assert!(
            cumulative > 0.0 && cumulative <= 1.0,
            "a cumulative mass of {cumulative} keeps no candidate or names more than every candidate",
        );
        Self::Sampled {
            temperature,
            keep,
            cumulative,
        }
    }

    pub fn temperature(self) -> Option<f32> {
        match self {
            Self::Greedy => None,
            Self::Sampled { temperature, .. } => Some(temperature),
        }
    }

    pub fn keep(self) -> Option<u32> {
        match self {
            Self::Greedy => None,
            Self::Sampled { keep, .. } => Some(keep),
        }
    }

    pub fn cumulative(self) -> Option<f32> {
        match self {
            Self::Greedy => None,
            Self::Sampled { cumulative, .. } => Some(cumulative),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Draw {
    pub(crate) sampling: Sampling,
    pub(crate) seed: Option<Value>,
    pub(crate) advance: Option<Value>,
    pub(crate) temperature: Option<Value>,
    pub(crate) keep: Option<Value>,
    pub(crate) cumulative: Option<Value>,
}

pub(crate) fn chosen(graph: &Graph, logits: Value, draw: Draw) -> Value {
    let chosen = match draw.sampling {
        Sampling::Greedy => graph.argmax(logits),
        Sampling::Sampled {
            temperature: _,
            keep,
            cumulative,
        } => {
            let seed = draw
                .seed
                .expect("a sampled program walks a seed of its own");
            let temperature = draw
                .temperature
                .expect("a sampled program walks the temperature of its own");
            let logits = graph.mul(logits, graph.recip(temperature));
            if let Some(advance) = draw.advance {
                graph.add_into(seed, advance);
            }
            let keep = draw
                .keep
                .unwrap_or_else(|| graph.fill(Shape::scalar(), keep as f32));
            let cumulative = draw
                .cumulative
                .unwrap_or_else(|| graph.fill(Shape::scalar(), cumulative));
            graph.sample(logits, seed, keep, cumulative)
        }
    };
    graph.retain(chosen);
    chosen
}
