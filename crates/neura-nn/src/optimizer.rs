use neura_program::{Gradients, Graph, Init, Shape, Value};

pub struct Sgd<'g> {
    descent: Value<'g>,
}

impl<'g> Sgd<'g> {
    pub fn new(graph: &Graph<'g>, rate: f32) -> Self {
        assert!(rate > 0.0, "a descent rate of {rate} moves nothing");
        Self {
            descent: graph.fill(Shape::scalar(), -rate),
        }
    }

    pub fn step(&self, graph: &Graph<'g>, gradients: &Gradients<'g>, parameters: &[Value<'g>]) {
        assert!(
            !parameters.is_empty(),
            "a step without parameters leaves the model as it was",
        );
        for parameter in parameters {
            let step = graph.mul(gradients.of(*parameter), self.descent);
            graph.add_into(*parameter, step);
        }
    }
}

pub struct Adam<'g> {
    descent: Value<'g>,
    mean_decay: Value<'g>,
    variance_decay: Value<'g>,
    mean_freshness: Value<'g>,
    variance_freshness: Value<'g>,
    floor: Value<'g>,
    moments: Vec<Moments<'g>>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Moments<'g> {
    pub parameter: Value<'g>,
    pub mean: Value<'g>,
    pub variance: Value<'g>,
}

impl<'g> Adam<'g> {
    pub fn new(
        graph: &Graph<'g>,
        rate: f32,
        mean_decay: f32,
        variance_decay: f32,
        floor: f32,
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
        Self {
            descent: graph.fill(Shape::scalar(), -rate),
            mean_decay: graph.fill(Shape::scalar(), mean_decay),
            variance_decay: graph.fill(Shape::scalar(), variance_decay),
            mean_freshness: graph.fill(Shape::scalar(), 1.0 - mean_decay),
            variance_freshness: graph.fill(Shape::scalar(), 1.0 - variance_decay),
            floor: graph.fill(Shape::scalar(), floor),
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
            mean: graph.parameter(graph.shape(parameter), Init::Zero),
            variance: graph.parameter(graph.shape(parameter), Init::Zero),
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
        for moments in &self.moments {
            let gradient = gradients.of(moments.parameter);
            let mean_step = graph.mul(gradient, self.mean_freshness);
            graph.mul_into(moments.mean, self.mean_decay);
            graph.add_into(moments.mean, mean_step);
            let squared_step = graph.mul(graph.mul(gradient, gradient), self.variance_freshness);
            graph.mul_into(moments.variance, self.variance_decay);
            graph.add_into(moments.variance, squared_step);
            let root = graph.sqrt(moments.variance);
            let scaled = graph.mul(moments.mean, graph.recip(graph.add(root, self.floor)));
            graph.add_into(moments.parameter, graph.mul(scaled, self.descent));
        }
    }
}
