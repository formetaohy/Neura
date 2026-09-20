use neura_program::{Graph, Init, Shape, Value};

pub struct Linear {
    weight: Value,
    bias: Value,
}

impl Linear {
    pub fn new(graph: &Graph, inputs: u32, outputs: u32, init: Init) -> Self {
        assert!(
            inputs > 0 && outputs > 0,
            "a dense layer of {inputs} by {outputs} carries no weight",
        );
        Self {
            weight: graph.parameter(Shape::matrix(inputs, outputs), init),
            bias: graph.parameter(Shape::vector(outputs), Init::Zero),
        }
    }

    pub fn forward(&self, graph: &Graph, input: Value) -> Value {
        graph.add(graph.matmul(input, self.weight), self.bias)
    }

    pub fn weight(&self) -> Value {
        self.weight
    }

    pub fn bias(&self) -> Value {
        self.bias
    }

    pub fn parameters(&self) -> [Value; 2] {
        [self.weight, self.bias]
    }
}

pub struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    pub fn new(graph: &Graph, widths: &[u32], init: Init) -> Self {
        assert!(
            widths.len() >= 2,
            "a multilayer perceptron spans at least two widths",
        );
        let layers = widths
            .windows(2)
            .map(|pair| Linear::new(graph, pair[0], pair[1], init))
            .collect();
        Self { layers }
    }

    pub fn forward(&self, graph: &Graph, input: Value) -> Value {
        let mut value = input;
        for (index, layer) in self.layers.iter().enumerate() {
            let dense = layer.forward(graph, value);
            value = if index + 1 == self.layers.len() {
                dense
            } else {
                graph.relu(dense)
            };
        }
        value
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn layers(&self) -> &[Linear] {
        &self.layers
    }
}
