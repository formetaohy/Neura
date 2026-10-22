use neura_abi::{Element, MAX_RANK};
use neura_graph::{AttentionOptions, Fan, Graph, Init, Shape, Value, Window};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct AdapterBankShape {
    count: u32,
    inputs: u32,
    outputs: u32,
    rank: u32,
}

impl AdapterBankShape {
    pub fn new(count: u32, inputs: u32, outputs: u32, rank: u32) -> Self {
        assert!(
            count > 0 && inputs > 0 && outputs > 0 && rank > 0,
            "an adapter bank of {count} adapters over {inputs} by {outputs} and {rank} ranks carries no bypass",
        );
        Self {
            count,
            inputs,
            outputs,
            rank,
        }
    }
}

pub struct AdapterBank {
    down: Value,
    up: Value,
    scale: Value,
    inputs: u32,
    outputs: u32,
    rank: u32,
}

impl AdapterBank {
    pub fn new(
        graph: &Graph,
        name: &str,
        shape: AdapterBankShape,
        init: Init,
        element: Element,
        scale: f32,
    ) -> Self {
        let AdapterBankShape {
            count,
            inputs,
            outputs,
            rank,
        } = shape;
        assert!(
            scale > 0.0 && scale.is_finite(),
            "an adapter bank scaled by {scale} contributes nothing",
        );
        let down = inputs
            .checked_mul(rank)
            .expect("the down adapter of one bank row fits one tensor row");
        let up = rank
            .checked_mul(outputs)
            .expect("the up adapter of one bank row fits one tensor row");
        Self {
            down: graph.named_parameter(
                &format!("{name}.down"),
                Shape::of([1, 1, count, down]),
                init.spread(Fan {
                    inputs,
                    outputs: rank,
                }),
                element,
            ),
            up: graph.named_parameter(
                &format!("{name}.up"),
                Shape::of([1, 1, count, up]),
                Init::Zero,
                element,
            ),
            scale: graph.named_parameter(
                &format!("{name}.scale"),
                Shape::of([1, 1, count, 1]),
                Init::Constant(scale),
                Element::Single,
            ),
            inputs,
            outputs,
            rank,
        }
    }

    pub fn forward(&self, graph: &Graph, input: Value, base: Value, selection: Value) -> Value {
        let dims = graph.shape(input).dims();
        let (planes, rows) = (dims[0], dims[2]);
        assert_eq!(
            dims[1], 1,
            "an adapter walks the rows of {dims:?} in one plane",
        );
        assert_eq!(
            dims[3], self.inputs,
            "an adapter of {} inputs walks a row of {dims:?}",
            self.inputs,
        );
        assert_eq!(
            graph.shape(base).dims(),
            [planes, 1, rows, self.outputs],
            "an adapter of {} outputs adds to {planes} planes of {rows} rows",
            self.outputs,
        );
        assert_eq!(
            graph.shape(selection).dims(),
            [planes, rows, 1, 1],
            "an adapter bank selection names one index per row of {planes} planes of {rows} rows",
        );
        let left = graph.permute(input, [0, 2, 1, 3]);
        let down = graph.reshape(
            graph.gather(self.down, selection),
            Shape::of([planes, rows, self.inputs, self.rank]),
        );
        let up = graph.reshape(
            graph.gather(self.up, selection),
            Shape::of([planes, rows, self.rank, self.outputs]),
        );
        let bypass = graph.matmul(graph.matmul(left, down), up);
        let scaled = graph.mul(bypass, graph.gather(self.scale, selection));
        let wide = graph.add(graph.permute(base, [0, 2, 1, 3]), scaled);
        graph.identity(graph.permute(wide, [0, 2, 1, 3]))
    }

    pub fn forward_shared(
        &self,
        graph: &Graph,
        input: Value,
        base: Value,
        selection: Value,
    ) -> Value {
        let dims = graph.shape(input).dims();
        let (planes, rows) = (dims[0], dims[2]);
        assert_eq!(
            dims[1], 1,
            "an adapter walks the rows of {dims:?} in one plane",
        );
        assert_eq!(
            dims[3], self.inputs,
            "an adapter of {} inputs walks a row of {dims:?}",
            self.inputs,
        );
        assert_eq!(
            graph.shape(base).dims(),
            [planes, 1, rows, self.outputs],
            "an adapter of {} outputs adds to {planes} planes of {rows} rows",
            self.outputs,
        );
        assert_eq!(
            graph.shape(selection).dims(),
            [planes, 1, 1, 1],
            "a shared adapter bank selection names one index per plane of {planes} planes",
        );
        let down = graph.reshape(
            graph.gather(self.down, selection),
            Shape::of([planes, 1, self.inputs, self.rank]),
        );
        let up = graph.reshape(
            graph.gather(self.up, selection),
            Shape::of([planes, 1, self.rank, self.outputs]),
        );
        let scale = graph.broadcast_to(
            graph.gather(self.scale, selection),
            Shape::of([planes, rows, 1, 1]),
        );
        if rows == 1 {
            let wide = graph.reshape(input, Shape::of([planes, 1, self.inputs, 1]));
            let hidden = graph.sum_axis(graph.mul(wide, down), 2);
            let hidden = graph.reshape(hidden, Shape::of([planes, 1, self.rank, 1]));
            let bypass = graph.sum_axis(graph.mul(hidden, up), 2);
            return graph.add(base, graph.mul(bypass, scale));
        }
        let left = graph.permute(input, [0, 2, 1, 3]);
        let hidden = graph.matmul(left, down);
        let bypass = graph.matmul(hidden, up);
        let wide = graph.add(graph.permute(base, [0, 2, 1, 3]), graph.mul(bypass, scale));
        graph.identity(graph.permute(wide, [0, 2, 1, 3]))
    }

    pub fn down(&self) -> Value {
        self.down
    }

    pub fn up(&self) -> Value {
        self.up
    }

    pub fn scale(&self) -> Value {
        self.scale
    }

    pub fn parameters(&self) -> [Value; 3] {
        [self.down, self.up, self.scale]
    }
}

pub struct Linear {
    weight: Value,
    bias: Value,
}

impl Linear {
    pub fn new(
        graph: &Graph,
        name: &str,
        inputs: u32,
        outputs: u32,
        init: Init,
        element: Element,
    ) -> Self {
        Self::declared(graph, inputs, outputs, element, |graph, weight, bias| {
            (
                graph.named_parameter(
                    &format!("{name}.weight"),
                    weight,
                    init.spread(Fan { inputs, outputs }),
                    element,
                ),
                graph.named_parameter(&format!("{name}.bias"), bias, Init::Zero, element),
            )
        })
    }

    pub fn quantized(
        graph: &Graph,
        name: &str,
        inputs: u32,
        outputs: u32,
        quantum: f32,
        init: Init,
        element: Element,
    ) -> Self {
        Self::declared(graph, inputs, outputs, element, |graph, weight, bias| {
            (
                graph.named_quantized_parameter(
                    &format!("{name}.weight"),
                    weight,
                    init.spread(Fan { inputs, outputs }),
                    quantum,
                ),
                graph.named_parameter(&format!("{name}.bias"), bias, Init::Zero, element),
            )
        })
    }

    pub fn block_quantized(
        graph: &Graph,
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
                    init.spread(Fan { inputs, outputs }),
                    storage,
                ),
                graph.named_parameter(&format!("{name}.bias"), bias, Init::Zero, element),
            )
        })
    }

    fn declared(
        graph: &Graph,
        inputs: u32,
        outputs: u32,
        element: Element,
        declare: impl FnOnce(&Graph, Shape, Shape) -> (Value, Value),
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

pub struct Conv2d {
    filter: Value,
    bias: Value,
    window: Window,
}

impl Conv2d {
    pub fn new(
        graph: &Graph,
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
        let taps = window.reach_rows() * window.reach_columns();
        Self {
            filter: graph.named_parameter(
                &format!("{name}.filter"),
                Shape::of([
                    outputs,
                    inputs / groups,
                    window.reach_rows(),
                    window.reach_columns(),
                ]),
                init.spread(Fan {
                    inputs: (inputs / groups) * taps,
                    outputs: (outputs / groups) * taps,
                }),
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

    pub fn forward(&self, graph: &Graph, input: Value) -> Value {
        graph.add(graph.conv2d(input, self.filter, self.window), self.bias)
    }

    pub fn filter(&self) -> Value {
        self.filter
    }

    pub fn bias(&self) -> Value {
        self.bias
    }

    pub fn parameters(&self) -> [Value; 2] {
        [self.filter, self.bias]
    }
}

pub struct ConvTranspose2d {
    filter: Value,
    bias: Value,
    window: Window,
    groups: u32,
}

impl ConvTranspose2d {
    pub fn new(
        graph: &Graph,
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
            "a transposed convolution of {inputs} channels into {outputs} over {groups} groups carries no filter",
        );
        assert!(
            inputs.is_multiple_of(groups) && outputs.is_multiple_of(groups),
            "a transposed convolution of {inputs} channels into {outputs} cuts {groups} groups",
        );
        let taps = window.reach_rows() * window.reach_columns();
        Self {
            filter: graph.named_parameter(
                &format!("{name}.filter"),
                Shape::of([
                    inputs,
                    outputs / groups,
                    window.reach_rows(),
                    window.reach_columns(),
                ]),
                init.spread(Fan {
                    inputs: (outputs / groups) * taps,
                    outputs: inputs * taps,
                }),
                element,
            ),
            bias: graph.named_parameter(
                &format!("{name}.bias"),
                Shape::of([1, outputs, 1, 1]),
                Init::Zero,
                element,
            ),
            window,
            groups,
        }
    }

    pub fn forward(&self, graph: &Graph, input: Value) -> Value {
        graph.add(
            graph.conv2d_transpose(input, self.filter, self.window, self.groups),
            self.bias,
        )
    }

    pub fn filter(&self) -> Value {
        self.filter
    }

    pub fn bias(&self) -> Value {
        self.bias
    }

    pub fn parameters(&self) -> [Value; 2] {
        [self.filter, self.bias]
    }
}

pub struct LayerNorm {
    columns: u32,
    scale: Value,
    shift: Value,
    floor: Value,
}

impl LayerNorm {
    pub fn new(
        graph: &Graph,
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

    pub fn forward(&self, graph: &Graph, input: Value) -> Value {
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

    pub fn scale(&self) -> Value {
        self.scale
    }

    pub fn shift(&self) -> Value {
        self.shift
    }

    pub fn parameters(&self) -> [Value; 2] {
        [self.scale, self.shift]
    }
}

pub struct Embedding {
    table: Value,
}

impl Embedding {
    pub fn new(
        graph: &Graph,
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

    pub fn forward(&self, graph: &Graph, indices: Value) -> Value {
        graph.gather(self.table, indices)
    }

    pub fn table(&self) -> Value {
        self.table
    }

    pub fn parameters(&self) -> [Value; 1] {
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

pub struct MultiHeadAttention {
    head: HeadShape,
    queries: Value,
    keys: Value,
    values: Value,
    output: Value,
    query_bias: Value,
    key_bias: Value,
    value_bias: Value,
    output_bias: Value,
    options: AttentionOptions,
    rotary: Option<Value>,
}

impl MultiHeadAttention {
    pub fn new(
        graph: &Graph,
        name: &str,
        head: HeadShape,
        init: Init,
        element: Element,
        options: AttentionOptions,
    ) -> Self {
        Self::declared(graph, name, head, init, element, options, None)
    }

    pub fn rotary(
        graph: &Graph,
        name: &str,
        head: HeadShape,
        init: Init,
        element: Element,
        options: AttentionOptions,
        base: f32,
    ) -> Self {
        assert!(
            base.is_finite() && base > 1.0,
            "an attention rotated by a base of {base} places every position on the same angle",
        );
        Self::declared(
            graph,
            name,
            head,
            init,
            element,
            options,
            Some(graph.named_knob(&format!("{name}.base"), base)),
        )
    }

    fn declared(
        graph: &Graph,
        name: &str,
        head: HeadShape,
        init: Init,
        element: Element,
        options: AttentionOptions,
        rotary: Option<Value>,
    ) -> Self {
        let projection = |heads: u32, columns: u32, what: &str| {
            graph.named_parameter(
                &format!("{name}.{what}.weight"),
                Shape::of([heads, 1, columns, columns]),
                init.spread(Fan {
                    inputs: columns,
                    outputs: columns,
                }),
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

    pub fn forward(&self, graph: &Graph, input: Value) -> Value {
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
        let project = |source: Value, weight: Value, bias: Value| {
            graph.add(graph.matmul(source, weight), bias)
        };
        let query = project(input, self.queries, self.query_bias);
        let key = project(input, self.keys, self.key_bias);
        let (query, key) = match self.rotary {
            Some(base) => (
                graph.rope(query, self.options.origin, Some(base)),
                graph.rope(key, self.options.origin, Some(base)),
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

    pub fn parameters(&self) -> [Value; 8] {
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

    pub fn rotary_base(&self) -> Option<Value> {
        self.rotary
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

pub struct Mlp {
    layers: Vec<Linear>,
}

impl Mlp {
    pub fn new(graph: &Graph, name: &str, widths: &[u32], init: Init, element: Element) -> Self {
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

pub struct RmsNorm {
    columns: u32,
    scale: Value,
    floor: Value,
}

impl RmsNorm {
    pub fn new(
        graph: &Graph,
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

    pub fn forward(&self, graph: &Graph, input: Value) -> Value {
        assert_eq!(
            graph.shape(input).columns(),
            self.columns,
            "a root mean square of {} columns scales {:?}",
            self.columns,
            graph.shape(input).dims(),
        );
        graph.rms_norm(input, self.scale, self.floor)
    }

    pub fn scale(&self) -> Value {
        self.scale
    }

    pub fn parameters(&self) -> [Value; 1] {
        [self.scale]
    }
}

pub struct GroupNorm {
    channels: u32,
    groups: u32,
    scale: Value,
    shift: Value,
    floor: Value,
}

impl GroupNorm {
    pub fn new(
        graph: &Graph,
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

    pub fn forward(&self, graph: &Graph, input: Value) -> Value {
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

    pub fn scale(&self) -> Value {
        self.scale
    }

    pub fn shift(&self) -> Value {
        self.shift
    }

    pub fn parameters(&self) -> [Value; 2] {
        [self.scale, self.shift]
    }
}
