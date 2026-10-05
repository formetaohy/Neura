use neura_abi::{Element, MAX_RANK};
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value, Window};

pub struct Adapter<'g> {
    down: Value<'g>,
    up: Value<'g>,
    scale: f32,
}

impl<'g> Adapter<'g> {
    pub fn new(
        graph: &Graph<'g>,
        name: &str,
        planes: [u32; 2],
        rank: u32,
        init: Init,
        element: Element,
        scale: f32,
    ) -> Self {
        let [inputs, outputs] = planes;
        assert!(
            inputs > 0 && outputs > 0 && rank > 0,
            "an adapter of {inputs} by {outputs} over {rank} ranks carries no bypass",
        );
        assert!(
            scale > 0.0 && scale.is_finite(),
            "an adapter scaled by {scale} contributes nothing",
        );
        Self {
            down: graph.named_parameter(
                &format!("{name}.down"),
                Shape::matrix(inputs, rank),
                init,
                element,
            ),
            up: graph.named_parameter(
                &format!("{name}.up"),
                Shape::matrix(rank, outputs),
                Init::Zero,
                element,
            ),
            scale,
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>, base: Value<'g>) -> Value<'g> {
        let bypass = graph.matmul(graph.matmul(input, self.down), self.up);
        graph.add(
            base,
            graph.mul(bypass, graph.fill(Shape::scalar(), self.scale)),
        )
    }

    pub fn down(&self) -> Value<'g> {
        self.down
    }

    pub fn up(&self) -> Value<'g> {
        self.up
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn parameters(&self) -> [Value<'g>; 2] {
        [self.down, self.up]
    }
}

pub struct Linear<'g> {
    weight: Value<'g>,
    bias: Value<'g>,
}

impl<'g> Linear<'g> {
    pub fn new(
        graph: &Graph<'g>,
        name: &str,
        inputs: u32,
        outputs: u32,
        init: Init,
        element: Element,
    ) -> Self {
        Self::declared(graph, inputs, outputs, element, |graph, weight, bias| {
            (
                graph.named_parameter(&format!("{name}.weight"), weight, init, element),
                graph.named_parameter(&format!("{name}.bias"), bias, Init::Zero, element),
            )
        })
    }

    pub fn quantized(
        graph: &Graph<'g>,
        name: &str,
        inputs: u32,
        outputs: u32,
        quantum: f32,
        init: Init,
        element: Element,
    ) -> Self {
        Self::declared(graph, inputs, outputs, element, |graph, weight, bias| {
            (
                graph.named_quantized_parameter(&format!("{name}.weight"), weight, init, quantum),
                graph.named_parameter(&format!("{name}.bias"), bias, Init::Zero, element),
            )
        })
    }

    pub fn block_quantized(
        graph: &Graph<'g>,
        name: &str,
        inputs: u32,
        outputs: u32,
        init: Init,
        element: Element,
        storage: Element,
    ) -> Self {
        Self::declared(graph, inputs, outputs, element, |graph, weight, bias| {
            (
                graph.named_block_quantized_parameter(
                    &format!("{name}.weight"),
                    weight,
                    init,
                    storage,
                ),
                graph.named_parameter(&format!("{name}.bias"), bias, Init::Zero, element),
            )
        })
    }

    fn declared(
        graph: &Graph<'g>,
        inputs: u32,
        outputs: u32,
        element: Element,
        declare: impl FnOnce(&Graph<'g>, Shape, Shape) -> (Value<'g>, Value<'g>),
    ) -> Self {
        assert!(
            inputs > 0 && outputs > 0,
            "a dense layer of {inputs} by {outputs} carries no weight",
        );
        assert!(
            !element.quantized(),
            "the product of a dense layer lands in the numbers its bias adds, and {} storage quantizes them; declare a quantized or a block quantized layer instead",
            element.name(),
        );
        let (weight, bias) = declare(
            graph,
            Shape::matrix(inputs, outputs),
            Shape::vector(outputs),
        );
        Self { weight, bias }
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
        name: &str,
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
            filter: graph.named_parameter(
                &format!("{name}.filter"),
                Shape::of([
                    outputs,
                    inputs / groups,
                    window.reach_rows(),
                    window.reach_columns(),
                ]),
                init,
                element,
            ),
            bias: graph.named_parameter(
                &format!("{name}.bias"),
                Shape::of([1, outputs, 1, 1]),
                Init::Zero,
                element,
            ),
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
    floor: Value<'g>,
}

impl<'g> LayerNorm<'g> {
    pub fn new(
        graph: &Graph<'g>,
        name: &str,
        columns: u32,
        init: Init,
        floor: f32,
        element: Element,
    ) -> Self {
        assert!(
            columns > 0,
            "a layer of {columns} columns normalizes nothing"
        );
        assert!(floor > 0.0, "a floor of {floor} divides by zero");
        Self {
            columns,
            scale: graph.named_parameter(
                &format!("{name}.scale"),
                Shape::vector(columns),
                init,
                element,
            ),
            shift: graph.named_parameter(
                &format!("{name}.shift"),
                Shape::vector(columns),
                Init::Zero,
                element,
            ),
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
        let mean = graph.mean_axis(input, MAX_RANK - 1);
        let centered = graph.sub(input, mean);
        let spread = graph.mean_axis(graph.mul(centered, centered), MAX_RANK - 1);
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
    pub fn new(
        graph: &Graph<'g>,
        name: &str,
        rows: u32,
        width: u32,
        init: Init,
        element: Element,
    ) -> Self {
        assert!(
            rows > 0 && width > 0,
            "an embedding of {rows} rows of {width} numbers holds nothing",
        );
        Self {
            table: graph.named_parameter(
                &format!("{name}.table"),
                Shape::matrix(rows, width),
                init,
                element,
            ),
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HeadShape {
    heads: u32,
    key_heads: u32,
    width: u32,
}

impl HeadShape {
    pub fn new(heads: u32, key_heads: u32, width: u32) -> Self {
        assert!(
            heads > 0 && key_heads > 0 && width > 0,
            "an attention of {heads} heads over {key_heads} key heads of {width} numbers carries no weight",
        );
        assert!(
            heads.is_multiple_of(key_heads),
            "an attention of {heads} query heads shares {key_heads} key heads, and every key head serves a whole group of queries",
        );
        Self {
            heads,
            key_heads,
            width,
        }
    }

    pub const fn heads(self) -> u32 {
        self.heads
    }

    pub const fn key_heads(self) -> u32 {
        self.key_heads
    }

    pub const fn width(self) -> u32 {
        self.width
    }
}

pub struct MultiHeadAttention<'g> {
    head: HeadShape,
    queries: Value<'g>,
    keys: Value<'g>,
    values: Value<'g>,
    output: Value<'g>,
    query_bias: Value<'g>,
    key_bias: Value<'g>,
    value_bias: Value<'g>,
    output_bias: Value<'g>,
    options: AttentionOptions<'g>,
    rotary: Option<f32>,
}

impl<'g> MultiHeadAttention<'g> {
    pub fn new(
        graph: &Graph<'g>,
        name: &str,
        head: HeadShape,
        init: Init,
        element: Element,
        options: AttentionOptions<'g>,
    ) -> Self {
        Self::declared(graph, name, head, init, element, options, None)
    }

    pub fn rotary(
        graph: &Graph<'g>,
        name: &str,
        head: HeadShape,
        init: Init,
        element: Element,
        options: AttentionOptions<'g>,
        base: f32,
    ) -> Self {
        assert!(
            base.is_finite() && base > 1.0,
            "an attention rotated by a base of {base} places every position on the same angle",
        );
        Self::declared(graph, name, head, init, element, options, Some(base))
    }

    fn declared(
        graph: &Graph<'g>,
        name: &str,
        head: HeadShape,
        init: Init,
        element: Element,
        options: AttentionOptions<'g>,
        rotary: Option<f32>,
    ) -> Self {
        let projection = |heads: u32, columns: u32, what: &str| {
            graph.named_parameter(
                &format!("{name}.{what}.weight"),
                Shape::of([heads, 1, columns, columns]),
                init,
                element,
            )
        };
        let shift = |heads: u32, columns: u32, what: &str| {
            graph.named_parameter(
                &format!("{name}.{what}.bias"),
                Shape::of([heads, 1, 1, columns]),
                Init::Zero,
                element,
            )
        };
        Self {
            head,
            queries: projection(head.heads(), head.width(), "query"),
            keys: projection(head.key_heads(), head.width(), "key"),
            values: projection(head.key_heads(), head.width(), "value"),
            output: projection(head.heads(), head.width(), "output"),
            query_bias: shift(head.heads(), head.width(), "query"),
            key_bias: shift(head.key_heads(), head.width(), "key"),
            value_bias: shift(head.key_heads(), head.width(), "value"),
            output_bias: shift(head.heads(), head.width(), "output"),
            options,
            rotary,
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, input: Value<'g>) -> Value<'g> {
        let shape = graph.shape(input);
        assert!(
            shape.dims()[0] == 1 || shape.dims()[0] == self.head.heads(),
            "an attention of {} heads reads a stream of {} heads, and one stream feeds every head or each head reads its own",
            self.head.heads(),
            shape.dims()[0],
        );
        assert!(
            shape.dims()[0] == 1 || shape.dims()[0] == self.head.key_heads(),
            "an attention of {} key heads reads a stream of {} heads, and one stream feeds every key head or each key head reads its own",
            self.head.key_heads(),
            shape.dims()[0],
        );
        assert_eq!(
            shape.dims()[3],
            self.head.width(),
            "an attention of width {} reads {} numbers per row",
            self.head.width(),
            shape.dims()[3],
        );
        let project = |source: Value<'g>, weight: Value<'g>, bias: Value<'g>| {
            graph.add(graph.matmul(source, weight), bias)
        };
        let query = project(input, self.queries, self.query_bias);
        let key = project(input, self.keys, self.key_bias);
        let (query, key) = match self.rotary {
            Some(base) => (
                graph.rope(query, self.options.origin, base),
                graph.rope(key, self.options.origin, base),
            ),
            None => (query, key),
        };
        let attended = graph.attention(
            query,
            key,
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

    pub fn head(&self) -> HeadShape {
        self.head
    }

    pub fn heads(&self) -> u32 {
        self.head.heads()
    }

    pub fn key_heads(&self) -> u32 {
        self.head.key_heads()
    }

    pub fn width(&self) -> u32 {
        self.head.width()
    }
}

pub struct Mlp<'g> {
    layers: Vec<Linear<'g>>,
}

impl<'g> Mlp<'g> {
    pub fn new(
        graph: &Graph<'g>,
        name: &str,
        widths: &[u32],
        init: Init,
        element: Element,
    ) -> Self {
        assert!(
            widths.len() >= 2,
            "a multilayer perceptron spans at least two widths",
        );
        let layers = widths
            .windows(2)
            .enumerate()
            .map(|(index, pair)| {
                Linear::new(
                    graph,
                    &format!("{name}.layers.{index}"),
                    pair[0],
                    pair[1],
                    init,
                    element,
                )
            })
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
    floor: Value<'g>,
}

impl<'g> RmsNorm<'g> {
    pub fn new(
        graph: &Graph<'g>,
        name: &str,
        columns: u32,
        init: Init,
        floor: f32,
        element: Element,
    ) -> Self {
        assert!(
            columns > 0,
            "a root mean square of {columns} columns scales nothing",
        );
        assert!(floor > 0.0, "a floor of {floor} divides by zero");
        Self {
            columns,
            scale: graph.named_parameter(
                &format!("{name}.scale"),
                Shape::vector(columns),
                init,
                element,
            ),
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
        let mean = graph.mean_axis(squares, MAX_RANK - 1);
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
        name: &str,
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
            scale: graph.named_parameter(
                &format!("{name}.scale"),
                Shape::of([1, channels, 1, 1]),
                init,
                element,
            ),
            shift: graph.named_parameter(
                &format!("{name}.shift"),
                Shape::of([1, channels, 1, 1]),
                Init::Zero,
                element,
            ),
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
