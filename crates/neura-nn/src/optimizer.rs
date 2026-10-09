use neura_abi::Element;
use neura_graph::{Gradients, Graph, Init, Shape, Value};

fn knob<'g>(graph: &Graph<'g>, name: &str, knob: &str, value: f32) -> Value<'g> {
    graph.named_state(
        &format!("{name}.{knob}"),
        Shape::scalar(),
        Init::Constant(value),
        Element::Single,
    )
}

pub struct Sgd<'g> {
    rate: Value<'g>,
    minus_one: Value<'g>,
    weight_decay: Option<Value<'g>>,
    tracked: Tracked<'g>,
}

enum Tracked<'g> {
    Plain(Vec<Value<'g>>),
    Momentum {
        momentum_decay: Value<'g>,
        velocities: Vec<Velocity<'g>>,
    },
}

struct Velocity<'g> {
    parameter: Value<'g>,
    velocity: Value<'g>,
}

impl<'g> Sgd<'g> {
    pub fn new(graph: &Graph<'g>, name: &str, rate: f32, weight_decay: f32) -> Self {
        assert!(rate > 0.0, "a descent rate of {rate} moves nothing");
        assert!(
            weight_decay >= 0.0 && weight_decay.is_finite(),
            "a weight decay of {weight_decay} grows a weight instead of shrinking it",
        );
        Self {
            rate: knob(graph, name, "rate", rate),
            minus_one: graph.fill(Shape::scalar(), -1.0),
            weight_decay: (weight_decay > 0.0)
                .then(|| knob(graph, name, "weight_decay", weight_decay)),
            tracked: Tracked::Plain(Vec::new()),
        }
    }

    pub fn momentum(
        graph: &Graph<'g>,
        name: &str,
        rate: f32,
        momentum_decay: f32,
        weight_decay: f32,
    ) -> Self {
        assert!(
            (0.0..1.0).contains(&momentum_decay),
            "a momentum of {momentum_decay} carries no velocity or forgets it whole",
        );
        let mut descent = Self::new(graph, name, rate, weight_decay);
        descent.tracked = Tracked::Momentum {
            momentum_decay: knob(graph, name, "momentum_decay", momentum_decay),
            velocities: Vec::new(),
        };
        descent
    }

    pub fn rate(&self) -> Value<'g> {
        self.rate
    }

    pub fn weight_decay(&self) -> Option<Value<'g>> {
        self.weight_decay
    }

    pub fn momentum_decay(&self) -> Option<Value<'g>> {
        match &self.tracked {
            Tracked::Plain(_) => None,
            Tracked::Momentum { momentum_decay, .. } => Some(*momentum_decay),
        }
    }

    pub fn track(&mut self, graph: &Graph<'g>, parameter: Value<'g>) {
        assert!(
            graph.trains(parameter),
            "a descent follows a parameter a gradient reaches, and value {} learns nothing",
            parameter.id(),
        );
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
                let velocity = graph.named_state(
                    &state_name(graph, parameter, ".velocity"),
                    graph.shape(parameter),
                    Init::Zero,
                    Element::Single,
                );
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
        assert!(
            match &self.tracked {
                Tracked::Plain(parameters) => !parameters.is_empty(),
                Tracked::Momentum { velocities, .. } => !velocities.is_empty(),
            },
            "a step without tracked parameters leaves the model as it was",
        );
        let descent = graph.mul(self.rate, self.minus_one);
        match &self.tracked {
            Tracked::Plain(parameters) => {
                for parameter in parameters {
                    let gradient = self.regularized(graph, *parameter, gradients.of(*parameter));
                    self.descend(graph, *parameter, gradient, descent);
                }
            }
            Tracked::Momentum {
                momentum_decay,
                velocities,
            } => {
                for velocity in velocities {
                    let gradient = self.regularized(
                        graph,
                        velocity.parameter,
                        gradients.of(velocity.parameter),
                    );
                    let carried =
                        graph.add(graph.mul(velocity.velocity, *momentum_decay), gradient);
                    graph.copy_into(velocity.velocity, carried);
                    self.descend(graph, velocity.parameter, carried, descent);
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
        match self.weight_decay {
            Some(decay) => graph.add(gradient, graph.mul(parameter, decay)),
            None => gradient,
        }
    }

    fn descend(
        &self,
        graph: &Graph<'g>,
        parameter: Value<'g>,
        gradient: Value<'g>,
        descent: Value<'g>,
    ) {
        graph.add_into(parameter, graph.mul(gradient, descent));
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Moments<'g> {
    pub parameter: Value<'g>,
    pub mean: Value<'g>,
    pub variance: Value<'g>,
}

pub struct AdamW<'g> {
    rate: Value<'g>,
    mean_decay: Value<'g>,
    variance_decay: Value<'g>,
    floor: Value<'g>,
    weight_decay: Option<Value<'g>>,
    clock: Value<'g>,
    one: Value<'g>,
    minus_one: Value<'g>,
    moments: Vec<Moments<'g>>,
}

impl<'g> AdamW<'g> {
    pub fn new(
        graph: &Graph<'g>,
        name: &str,
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
            rate: knob(graph, name, "rate", rate),
            mean_decay: knob(graph, name, "mean_decay", mean_decay),
            variance_decay: knob(graph, name, "variance_decay", variance_decay),
            floor: knob(graph, name, "floor", floor),
            weight_decay: (weight_decay > 0.0)
                .then(|| knob(graph, name, "weight_decay", weight_decay)),
            clock: graph.named_state(
                &format!("{name}.step"),
                Shape::scalar(),
                Init::Zero,
                Element::Single,
            ),
            one: graph.fill(Shape::scalar(), 1.0),
            minus_one: graph.fill(Shape::scalar(), -1.0),
            moments: Vec::new(),
        }
    }

    pub fn rate(&self) -> Value<'g> {
        self.rate
    }

    pub fn mean_decay(&self) -> Value<'g> {
        self.mean_decay
    }

    pub fn variance_decay(&self) -> Value<'g> {
        self.variance_decay
    }

    pub fn floor(&self) -> Value<'g> {
        self.floor
    }

    pub fn weight_decay(&self) -> Option<Value<'g>> {
        self.weight_decay
    }

    pub fn track(&mut self, graph: &Graph<'g>, parameter: Value<'g>) -> Moments<'g> {
        assert!(
            graph.trains(parameter),
            "a descent follows a parameter a gradient reaches, and value {} learns nothing",
            parameter.id(),
        );
        assert!(
            !self
                .moments
                .iter()
                .any(|moments| moments.parameter == parameter),
            "a parameter carries one pair of moments",
        );
        let moments = Moments {
            parameter,
            mean: graph.named_state(
                &state_name(graph, parameter, ".mean"),
                graph.shape(parameter),
                Init::Zero,
                Element::Single,
            ),
            variance: graph.named_state(
                &state_name(graph, parameter, ".variance"),
                graph.shape(parameter),
                Init::Zero,
                Element::Single,
            ),
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
        let mean_freshness = graph.sub(self.one, self.mean_decay);
        let variance_freshness = graph.sub(self.one, self.variance_decay);
        let mean_scale = graph.recip(graph.sub(self.one, graph.pow(self.mean_decay, self.clock)));
        let variance_scale =
            graph.recip(graph.sub(self.one, graph.pow(self.variance_decay, self.clock)));
        let descent = graph.mul(self.rate, self.minus_one);
        for moments in &self.moments {
            let gradient = gradients.of(moments.parameter);
            graph.mul_into(moments.mean, self.mean_decay);
            graph.add_into(moments.mean, graph.mul(gradient, mean_freshness));
            graph.mul_into(moments.variance, self.variance_decay);
            graph.add_into(
                moments.variance,
                graph.mul(graph.mul(gradient, gradient), variance_freshness),
            );
            let mean = graph.mul(moments.mean, mean_scale);
            let variance = graph.mul(moments.variance, variance_scale);
            let scaled = graph.mul(
                mean,
                graph.recip(graph.add(graph.sqrt(variance), self.floor)),
            );
            let mut update = graph.mul(scaled, descent);
            if let Some(decay) = self.weight_decay {
                update = graph.add(
                    update,
                    graph.mul(graph.mul(moments.parameter, decay), descent),
                );
            }
            graph.add_into(moments.parameter, update);
        }
    }
}

fn state_name<'g>(graph: &Graph<'g>, parameter: Value<'g>, suffix: &str) -> String {
    let name = graph.name_of(parameter).unwrap_or_else(|| {
        panic!(
            "value {} carries no name, and a descent names the state it keeps of every parameter it follows; declare the parameter with a named parameter",
            parameter.id(),
        )
    });
    format!("{name}{suffix}")
}
