use neura_abi::Element;
use neura_graph::{Gradients, Graph, Init, Shape, Value};

pub struct Sgd<'g> {
    descent: Value<'g>,
    decay: Option<Value<'g>>,
    tracked: Tracked<'g>,
}

enum Tracked<'g> {
    Plain(Vec<Value<'g>>),
    Momentum {
        decay: Value<'g>,
        velocities: Vec<Velocity<'g>>,
    },
}

struct Velocity<'g> {
    parameter: Value<'g>,
    velocity: Value<'g>,
}

impl<'g> Sgd<'g> {
    pub fn new(graph: &Graph<'g>, rate: f32, weight_decay: f32) -> Self {
        assert!(rate > 0.0, "a descent rate of {rate} moves nothing");
        assert!(
            weight_decay >= 0.0 && weight_decay.is_finite(),
            "a weight decay of {weight_decay} grows a weight instead of shrinking it",
        );
        Self {
            descent: graph.fill(Shape::scalar(), -rate),
            decay: (weight_decay > 0.0).then(|| graph.fill(Shape::scalar(), weight_decay)),
            tracked: Tracked::Plain(Vec::new()),
        }
    }

    pub fn momentum(graph: &Graph<'g>, rate: f32, momentum_decay: f32, weight_decay: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&momentum_decay),
            "a momentum of {momentum_decay} carries no velocity or forgets it whole",
        );
        let mut descent = Self::new(graph, rate, weight_decay);
        descent.tracked = Tracked::Momentum {
            decay: graph.fill(Shape::scalar(), momentum_decay),
            velocities: Vec::new(),
        };
        descent
    }

    pub fn track(&mut self, graph: &Graph<'g>, parameter: Value<'g>) {
        match &mut self.tracked {
            Tracked::Plain(parameters) => {
                assert!(
                    !parameters.contains(&parameter),
                    "a parameter enters one descent once",
                );
                parameters.push(parameter);
            }
            Tracked::Momentum { velocities, .. } => {
                assert!(
                    !velocities
                        .iter()
                        .any(|velocity| velocity.parameter == parameter),
                    "a parameter carries one velocity",
                );
                let velocity = graph.parameter(graph.shape(parameter), Init::Zero, Element::Single);
                velocities.push(Velocity {
                    parameter,
                    velocity,
                });
            }
        }
    }

    pub fn track_all(&mut self, graph: &Graph<'g>, parameters: &[Value<'g>]) {
        for parameter in parameters {
            self.track(graph, *parameter);
        }
    }

    pub fn step(&self, graph: &Graph<'g>, gradients: &Gradients<'g>) {
        match &self.tracked {
            Tracked::Plain(parameters) => {
                assert!(
                    !parameters.is_empty(),
                    "a step without tracked parameters leaves the model as it was",
                );
                for parameter in parameters {
                    let gradient = self.regularized(graph, *parameter, gradients.of(*parameter));
                    self.descend(graph, *parameter, gradient);
                }
            }
            Tracked::Momentum { decay, velocities } => {
                assert!(
                    !velocities.is_empty(),
                    "a step without tracked parameters leaves the model as it was",
                );
                for velocity in velocities {
                    let gradient = self.regularized(
                        graph,
                        velocity.parameter,
                        gradients.of(velocity.parameter),
                    );
                    let carried = graph.add(graph.mul(velocity.velocity, *decay), gradient);
                    graph.copy_into(velocity.velocity, carried);
                    self.descend(graph, velocity.parameter, carried);
                }
            }
        }
    }

    fn regularized(
        &self,
        graph: &Graph<'g>,
        parameter: Value<'g>,
        gradient: Value<'g>,
    ) -> Value<'g> {
        match self.decay {
            Some(decay) => graph.add(gradient, graph.mul(parameter, decay)),
            None => gradient,
        }
    }

    fn descend(&self, graph: &Graph<'g>, parameter: Value<'g>, gradient: Value<'g>) {
        graph.add_into(parameter, graph.mul(gradient, self.descent));
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Moments<'g> {
    pub parameter: Value<'g>,
    pub mean: Value<'g>,
    pub variance: Value<'g>,
}

pub struct AdamW<'g> {
    descent: Value<'g>,
    mean_decay: Value<'g>,
    variance_decay: Value<'g>,
    mean_freshness: Value<'g>,
    variance_freshness: Value<'g>,
    floor: Value<'g>,
    decay: Option<Value<'g>>,
    clock: Value<'g>,
    one: Value<'g>,
    moments: Vec<Moments<'g>>,
}

impl<'g> AdamW<'g> {
    pub fn new(
        graph: &Graph<'g>,
        rate: f32,
        mean_decay: f32,
        variance_decay: f32,
        floor: f32,
        weight_decay: f32,
    ) -> Self {
        assert!(rate > 0.0, "a descent rate of {rate} moves nothing");
        assert!(
            (0.0..1.0).contains(&mean_decay),
            "a mean decay of {mean_decay} forgets nothing or everything",
        );
        assert!(
            (0.0..1.0).contains(&variance_decay),
            "a variance decay of {variance_decay} forgets nothing or everything",
        );
        assert!(floor > 0.0, "a floor of {floor} divides by zero");
        assert!(
            weight_decay >= 0.0 && weight_decay.is_finite(),
            "a weight decay of {weight_decay} grows a weight instead of shrinking it",
        );
        Self {
            descent: graph.fill(Shape::scalar(), -rate),
            mean_decay: graph.fill(Shape::scalar(), mean_decay),
            variance_decay: graph.fill(Shape::scalar(), variance_decay),
            mean_freshness: graph.fill(Shape::scalar(), 1.0 - mean_decay),
            variance_freshness: graph.fill(Shape::scalar(), 1.0 - variance_decay),
            floor: graph.fill(Shape::scalar(), floor),
            decay: (weight_decay > 0.0).then(|| graph.fill(Shape::scalar(), weight_decay)),
            clock: graph.parameter(Shape::scalar(), Init::Zero, Element::Single),
            one: graph.fill(Shape::scalar(), 1.0),
            moments: Vec::new(),
        }
    }

    pub fn track(&mut self, graph: &Graph<'g>, parameter: Value<'g>) -> Moments<'g> {
        assert!(
            !self
                .moments
                .iter()
                .any(|moments| moments.parameter == parameter),
            "a parameter carries one pair of moments",
        );
        let moments = Moments {
            parameter,
            mean: graph.parameter(graph.shape(parameter), Init::Zero, Element::Single),
            variance: graph.parameter(graph.shape(parameter), Init::Zero, Element::Single),
        };
        self.moments.push(moments);
        moments
    }

    pub fn track_all(&mut self, graph: &Graph<'g>, parameters: &[Value<'g>]) -> Vec<Moments<'g>> {
        parameters
            .iter()
            .map(|parameter| self.track(graph, *parameter))
            .collect()
    }

    pub fn moments(&self) -> &[Moments<'g>] {
        &self.moments
    }

    pub fn step(&self, graph: &Graph<'g>, gradients: &Gradients<'g>) {
        assert!(
            !self.moments.is_empty(),
            "a step without tracked parameters leaves the model as it was",
        );
        graph.add_into(self.clock, self.one);
        let mean_scale = graph.recip(graph.sub(self.one, graph.pow(self.mean_decay, self.clock)));
        let variance_scale =
            graph.recip(graph.sub(self.one, graph.pow(self.variance_decay, self.clock)));
        for moments in &self.moments {
            let gradient = gradients.of(moments.parameter);
            graph.mul_into(moments.mean, self.mean_decay);
            graph.add_into(moments.mean, graph.mul(gradient, self.mean_freshness));
            graph.mul_into(moments.variance, self.variance_decay);
            graph.add_into(
                moments.variance,
                graph.mul(graph.mul(gradient, gradient), self.variance_freshness),
            );
            let mean = graph.mul(moments.mean, mean_scale);
            let variance = graph.mul(moments.variance, variance_scale);
            let scaled = graph.mul(
                mean,
                graph.recip(graph.add(graph.sqrt(variance), self.floor)),
            );
            let mut update = graph.mul(scaled, self.descent);
            if let Some(decay) = self.decay {
                update = graph.add(
                    update,
                    graph.mul(graph.mul(moments.parameter, decay), self.descent),
                );
            }
            graph.add_into(moments.parameter, update);
        }
    }
}
