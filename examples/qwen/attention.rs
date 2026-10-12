use crate::config::Config;
use crate::mlp::Mlp;
use crate::ops::{flatten, heads, interleaved, partial_rope, rms, seat, tensor};
use neura::gguf::GgufFile;
use neura::{AttentionOptions, Graph, Shape, Value};

pub struct Attention<'g> {
    norm: Value<'g>,
    query: Value<'g>,
    key: Value<'g>,
    value: Value<'g>,
    output: Value<'g>,
    query_norm: Value<'g>,
    key_norm: Value<'g>,
    keys: Value<'g>,
    values: Value<'g>,
    mlp: Mlp<'g>,
    floor: Value<'g>,
    base: Value<'g>,
    scale: Value<'g>,
    rope: u32,
    heads: u32,
    key_heads: u32,
    width: u32,
    capacity: u32,
}

impl<'g> Attention<'g> {
    pub fn build(
        graph: &Graph<'g>,
        source: &GgufFile,
        config: &Config,
        layer: u32,
        context: u32,
    ) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        let width = config.head_width;
        let packed = config.heads * width * 2;
        let held = config.key_heads * width;
        let norm = tensor(graph, source, &at("attn_norm.weight"));
        let query = tensor(graph, source, &at("attn_q.weight"));
        let key = tensor(graph, source, &at("attn_k.weight"));
        let value = tensor(graph, source, &at("attn_v.weight"));
        let output = tensor(graph, source, &at("attn_output.weight"));
        let query_norm = tensor(graph, source, &at("attn_q_norm.weight"));
        let key_norm = tensor(graph, source, &at("attn_k_norm.weight"));
        assert_eq!(
            graph.shape(query).dims(),
            [1, 1, packed, config.hidden],
            "the query of layer {layer} walks {:?} where a query and a gate of {packed} numbers walk [1, 1, {packed}, {}]",
            graph.shape(query).dims(),
            config.hidden,
        );
        assert_eq!(
            graph.shape(key).dims(),
            [1, 1, held, config.hidden],
            "the key of layer {layer} walks {:?} where {held} numbers walk [1, 1, {held}, {}]",
            graph.shape(key).dims(),
            config.hidden,
        );
        assert_eq!(
            graph.shape(value).dims(),
            graph.shape(key).dims(),
            "the value of layer {layer} walks {:?} where its key walks {:?}",
            graph.shape(value).dims(),
            graph.shape(key).dims(),
        );
        assert_eq!(
            graph.shape(output).dims(),
            [1, 1, config.hidden, config.heads * width],
            "the output of layer {layer} walks {:?} where {} rows of {} numbers walk [1, 1, {}, {}]",
            graph.shape(output).dims(),
            config.hidden,
            config.heads * width,
            config.hidden,
            config.heads * width,
        );
        assert_eq!(
            graph.shape(query_norm).dims(),
            [1, 1, 1, width],
            "the query norm of layer {layer} walks {:?} where {width} numbers walk [1, 1, 1, {width}]",
            graph.shape(query_norm).dims(),
        );
        assert_eq!(
            graph.shape(key_norm).dims(),
            graph.shape(query_norm).dims(),
            "the key norm of layer {layer} walks {:?} where its query norm walks {:?}",
            graph.shape(key_norm).dims(),
            graph.shape(query_norm).dims(),
        );
        Self {
            norm,
            query,
            key,
            value,
            output,
            query_norm,
            key_norm,
            keys: graph.resident(
                Shape::of([config.key_heads, 1, context, width]),
                neura::Element::Single,
            ),
            values: graph.resident(
                Shape::of([config.key_heads, 1, context, width]),
                neura::Element::Single,
            ),
            mlp: Mlp::build(graph, source, layer, config.hidden, config.feed_forward),
            floor: graph.fill(Shape::scalar(), config.floor),
            base: graph.fill(Shape::scalar(), config.rope_base),
            scale: graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
            rope: config.rope_width,
            heads: config.heads,
            key_heads: config.key_heads,
            width,
            capacity: context,
        }
    }

    pub fn states(&self) -> [Value<'g>; 2] {
        [self.keys, self.values]
    }

    pub fn forward(&self, graph: &Graph<'g>, hidden: Value<'g>, position: Value<'g>) -> Value<'g> {
        let normed = rms(graph, hidden, self.norm, self.floor);
        let packed = graph.matmul(normed, graph.permute(self.query, [0, 1, 3, 2]));
        let query = interleaved(graph, packed, self.heads, self.width, 0);
        let gate = interleaved(graph, packed, self.heads, self.width, self.width);
        let key = heads(
            graph,
            graph.matmul(normed, graph.permute(self.key, [0, 1, 3, 2])),
            self.key_heads,
            self.width,
        );
        let value = heads(
            graph,
            graph.matmul(normed, graph.permute(self.value, [0, 1, 3, 2])),
            self.key_heads,
            self.width,
        );
        let query = partial_rope(
            graph,
            rms(graph, query, self.query_norm, self.floor),
            position,
            self.base,
            self.rope,
        );
        let key = partial_rope(
            graph,
            rms(graph, key, self.key_norm, self.floor),
            position,
            self.base,
            self.rope,
        );
        let row = seat(graph, position, self.key_heads, self.capacity);
        graph.write_into(self.keys, row, key);
        graph.write_into(self.values, row, value);
        let attended = graph.attention(
            query,
            self.keys,
            self.values,
            AttentionOptions {
                scale: Some(self.scale),
                causal: true,
                origin: Some(position),
                segments: None,
                reach: None,
                query_segments: None,
                bias: None,
            },
        );
        let gated = graph.mul(
            flatten(graph, attended),
            graph.sigmoid(flatten(graph, gate)),
        );
        let projected = graph.matmul(gated, graph.permute(self.output, [0, 1, 3, 2]));
        self.mlp
            .forward(graph, graph.add(hidden, projected), self.floor)
    }
}
