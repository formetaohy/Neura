use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value, Window};

pub struct Linear<'g> {
    weight: Value<'g>,
    bias: Value<'g>,
}

impl<'g> Linear<'g> {
    pub fn new(graph: &Graph<'g>, inputs: u32, outputs: u32, init: Init, element: Element) -> Self {
        assert!(
            inputs > 0 && outputs > 0,
            "a dense layer of {inputs} by {outputs} carries no weight",
        );
        Self {
            weight: graph.parameter(Shape::matrix(inputs, outputs), init, element),
            bias: graph.parameter(Shape::vector(outputs), Init::Zero, element),
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
    pub fn new(
        graph: &Graph<'g>,
        channels: [u32; 2],
        groups: u32,
        window: Window,
        init: Init,
        element: Element,
    ) -> Self {
        let [inputs, outputs] = channels;
        assert!(
            inputs > 0 && outputs > 0 && groups > 0,
            "a convolution of {inputs} channels into {outputs} over {groups} groups carries no filter",
        );
        assert!(
            inputs.is_multiple_of(groups) && outputs.is_multiple_of(groups),
            "a convolution of {inputs} channels into {outputs} cuts {groups} groups",
        );
        Self {
            filter: graph.parameter(
                Shape::of([
                    outputs,
                    inputs / groups,
                    window.reach_rows(),
                    window.reach_columns(),
                ]),
                init,
                element,
            ),
            bias: graph.parameter(Shape::of([1, outputs, 1, 1]), Init::Zero, element),
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
    pub fn new(graph: &Graph<'g>, columns: u32, init: Init, floor: f32, element: Element) -> Self {
        assert!(
            columns > 0,
            "a layer of {columns} columns normalizes nothing"
        );
        assert!(floor > 0.0, "a floor of {floor} divides by zero");
        Self {
            columns,
            scale: graph.parameter(Shape::vector(columns), init, element),
            shift: graph.parameter(Shape::vector(columns), Init::Zero, element),
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
    pub fn new(graph: &Graph<'g>, rows: u32, width: u32, init: Init, element: Element) -> Self {
        assert!(
            rows > 0 && width > 0,
            "an embedding of {rows} rows of {width} numbers holds nothing",
        );
        Self {
            table: graph.parameter(Shape::matrix(rows, width), init, element),
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

pub struct MultiHeadAttention<'g> {
    heads: u32,
    width: u32,
    queries: Value<'g>,
    keys: Value<'g>,
    values: Value<'g>,
    output: Value<'g>,
    query_bias: Value<'g>,
    key_bias: Value<'g>,
    value_bias: Value<'g>,
    output_bias: Value<'g>,
    options: AttentionOptions<'g>,
}

impl<'g> MultiHeadAttention<'g> {
    pub fn new(
        graph: &Graph<'g>,
        heads: u32,
        width: u32,
        init: Init,
        element: Element,
        options: AttentionOptions<'g>,
    ) -> Self {
        assert!(
            heads > 0 && width > 0,
            "an attention of {heads} heads of {width} numbers carries no weight",
        );
        let projection =
            |columns: u32| graph.parameter(Shape::of([heads, 1, columns, columns]), init, element);
        let shift =
            |columns: u32| graph.parameter(Shape::of([heads, 1, 1, columns]), Init::Zero, element);
        Self {
            heads,
            width,
            queries: projection(width),
            keys: projection(width),
            values: projection(width),
            output: projection(width),
            query_bias: shift(width),
            key_bias: shift(width),
            value_bias: shift(width),
            output_bias: shift(width),
            options,
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>) -> Value<'g> {
        let shape = graph.shape(input);
        assert_eq!(
            shape.dims()[0],
            self.heads,
            "an attention of {} heads reads a tensor of {} heads",
            self.heads,
            shape.dims()[0],
        );
        assert_eq!(
            shape.dims()[3],
            self.width,
            "an attention of width {} reads {} numbers per row",
            self.width,
            shape.dims()[3],
        );
        let project = |source: Value<'g>, weight: Value<'g>, bias: Value<'g>| {
            graph.add(graph.matmul(source, weight), bias)
        };
        let attended = graph.attention(
            project(input, self.queries, self.query_bias),
            project(input, self.keys, self.key_bias),
            project(input, self.values, self.value_bias),
            self.options,
        );
        project(attended, self.output, self.output_bias)
    }

    pub fn parameters(&self) -> [Value<'g>; 8] {
        [
            self.queries,
            self.keys,
            self.values,
            self.output,
            self.query_bias,
            self.key_bias,
            self.value_bias,
            self.output_bias,
        ]
    }

    pub fn heads(&self) -> u32 {
        self.heads
    }

    pub fn width(&self) -> u32 {
        self.width
    }
}

pub struct Mlp<'g> {
    layers: Vec<Linear<'g>>,
}

impl<'g> Mlp<'g> {
    pub fn new(graph: &Graph<'g>, widths: &[u32], init: Init, element: Element) -> Self {
        assert!(
            widths.len() >= 2,
            "a multilayer perceptron spans at least two widths",
        );
        let layers = widths
            .windows(2)
            .map(|pair| Linear::new(graph, pair[0], pair[1], init, element))
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

pub struct RmsNorm<'g> {
    columns: u32,
    scale: Value<'g>,
    share: Value<'g>,
    floor: Value<'g>,
}

impl<'g> RmsNorm<'g> {
    pub fn new(graph: &Graph<'g>, columns: u32, init: Init, floor: f32, element: Element) -> Self {
        assert!(
            columns > 0,
            "a root mean square of {columns} columns scales nothing",
        );
        assert!(floor > 0.0, "a floor of {floor} divides by zero");
        Self {
            columns,
            scale: graph.parameter(Shape::vector(columns), init, element),
            share: graph.fill(Shape::scalar(), 1.0 / columns as f32),
            floor: graph.fill(Shape::scalar(), floor),
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>) -> Value<'g> {
        assert_eq!(
            graph.shape(input).columns(),
            self.columns,
            "a root mean square of {} columns scales {:?}",
            self.columns,
            graph.shape(input).dims(),
        );
        let squares = graph.mul(input, input);
        let mean = graph.mul(graph.sum_rows(squares), self.share);
        let deviation = graph.sqrt(graph.add(mean, self.floor));
        graph.mul(graph.mul(input, graph.recip(deviation)), self.scale)
    }

    pub fn scale(&self) -> Value<'g> {
        self.scale
    }

    pub fn parameters(&self) -> [Value<'g>; 1] {
        [self.scale]
    }
}

pub struct GroupNorm<'g> {
    channels: u32,
    groups: u32,
    scale: Value<'g>,
    shift: Value<'g>,
    floor: Value<'g>,
}

impl<'g> GroupNorm<'g> {
    pub fn new(
        graph: &Graph<'g>,
        channels: u32,
        groups: u32,
        init: Init,
        floor: f32,
        element: Element,
    ) -> Self {
        assert!(
            channels > 0 && groups > 0 && channels.is_multiple_of(groups),
            "a normalization of {channels} channels cuts {groups} groups",
        );
        assert!(floor > 0.0, "a floor of {floor} divides by zero");
        Self {
            channels,
            groups,
            scale: graph.parameter(Shape::of([1, channels, 1, 1]), init, element),
            shift: graph.parameter(Shape::of([1, channels, 1, 1]), Init::Zero, element),
            floor: graph.fill(Shape::scalar(), floor),
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>) -> Value<'g> {
        let dims = graph.shape(input).dims();
        assert_eq!(
            dims[1], self.channels,
            "a normalization of {} channels reads {:?}",
            self.channels, dims,
        );
        let grouped = graph.reshape(
            input,
            Shape::of([
                dims[0] * self.groups,
                self.channels / self.groups,
                dims[2],
                dims[3],
            ]),
        );
        let mean = graph.mean_axis(graph.mean_axis(graph.mean_axis(grouped, 3), 2), 1);
        let centered = graph.sub(grouped, mean);
        let spread = graph.mean_axis(
            graph.mean_axis(graph.mean_axis(graph.mul(centered, centered), 3), 2),
            1,
        );
        let deviation = graph.sqrt(graph.add(spread, self.floor));
        let normalized = graph.mul(centered, graph.recip(deviation));
        let whole = graph.reshape(normalized, Shape::of(dims));
        graph.add(graph.mul(whole, self.scale), self.shift)
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
