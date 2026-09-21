use neura_program::{Graph, Init, Shape, Value, Window};

pub struct Linear<'g> {
    weight: Value<'g>,
    bias: Value<'g>,
}

impl<'g> Linear<'g> {
    pub fn new(graph: &Graph<'g>, inputs: u32, outputs: u32, init: Init) -> Self {
        assert!(
            inputs > 0 && outputs > 0,
            "a dense layer of {inputs} by {outputs} carries no weight",
        );
        Self {
            weight: graph.parameter(Shape::matrix(inputs, outputs), init),
            bias: graph.parameter(Shape::vector(outputs), Init::Zero),
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>) -> Value<'g> {
        graph.add(graph.matmul(input, self.weight), self.bias)
    }

    pub fn weight(&self) -> Value<'g> {
        self.weight
    }

    pub fn bias(&self) -> Value<'g> {
        self.bias
    }

    pub fn parameters(&self) -> [Value<'g>; 2] {
        [self.weight, self.bias]
    }
}

pub struct Conv2d<'g> {
    filter: Value<'g>,
    bias: Value<'g>,
    window: Window,
}

impl<'g> Conv2d<'g> {
    pub fn new(graph: &Graph<'g>, channels: [u32; 2], window: Window, init: Init) -> Self {
        let [inputs, outputs] = channels;
        assert!(
            inputs > 0 && outputs > 0,
            "a convolution of {inputs} channels into {outputs} carries no filter",
        );
        Self {
            filter: graph.parameter(
                Shape::of([outputs, inputs, window.reach_rows(), window.reach_columns()]),
                init,
            ),
            bias: graph.parameter(Shape::of([1, outputs, 1, 1]), Init::Zero),
            window,
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>) -> Value<'g> {
        graph.add(graph.conv2d(input, self.filter, self.window), self.bias)
    }

    pub fn filter(&self) -> Value<'g> {
        self.filter
    }

    pub fn bias(&self) -> Value<'g> {
        self.bias
    }

    pub fn parameters(&self) -> [Value<'g>; 2] {
        [self.filter, self.bias]
    }
}

pub struct LayerNorm<'g> {
    columns: u32,
    scale: Value<'g>,
    shift: Value<'g>,
    share: Value<'g>,
    floor: Value<'g>,
}

impl<'g> LayerNorm<'g> {
    pub fn new(graph: &Graph<'g>, columns: u32, init: Init, floor: f32) -> Self {
        assert!(
            columns > 0,
            "a layer of {columns} columns normalizes nothing"
        );
        assert!(floor > 0.0, "a floor of {floor} divides by zero");
        Self {
            columns,
            scale: graph.parameter(Shape::vector(columns), init),
            shift: graph.parameter(Shape::vector(columns), Init::Zero),
            share: graph.fill(Shape::scalar(), 1.0 / columns as f32),
            floor: graph.fill(Shape::scalar(), floor),
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>) -> Value<'g> {
        assert_eq!(
            graph.shape(input).columns(),
            self.columns,
            "a layer of {} columns normalizes {:?}",
            self.columns,
            graph.shape(input).dims(),
        );
        let mean = graph.mul(graph.sum_rows(input), self.share);
        let centered = graph.sub(input, mean);
        let spread = graph.mul(graph.sum_rows(graph.mul(centered, centered)), self.share);
        let deviation = graph.sqrt(graph.add(spread, self.floor));
        let sharpened = graph.mul(centered, graph.recip(deviation));
        graph.add(graph.mul(sharpened, self.scale), self.shift)
    }

    pub fn scale(&self) -> Value<'g> {
        self.scale
    }

    pub fn shift(&self) -> Value<'g> {
        self.shift
    }

    pub fn parameters(&self) -> [Value<'g>; 2] {
        [self.scale, self.shift]
    }
}

pub struct Embedding<'g> {
    table: Value<'g>,
}

impl<'g> Embedding<'g> {
    pub fn new(graph: &Graph<'g>, rows: u32, width: u32, init: Init) -> Self {
        assert!(
            rows > 0 && width > 0,
            "an embedding of {rows} rows of {width} numbers holds nothing",
        );
        Self {
            table: graph.parameter(Shape::matrix(rows, width), init),
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, indices: Value<'g>) -> Value<'g> {
        graph.gather(self.table, indices)
    }

    pub fn table(&self) -> Value<'g> {
        self.table
    }

    pub fn parameters(&self) -> [Value<'g>; 1] {
        [self.table]
    }
}

pub struct Mlp<'g> {
    layers: Vec<Linear<'g>>,
}

impl<'g> Mlp<'g> {
    pub fn new(graph: &Graph<'g>, widths: &[u32], init: Init) -> Self {
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

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>) -> Value<'g> {
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

    pub fn parameters(&self) -> Vec<Value<'g>> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    pub fn layers(&self) -> &[Linear<'g>] {
        &self.layers
    }
}
