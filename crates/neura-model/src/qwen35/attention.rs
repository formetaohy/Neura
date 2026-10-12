use crate::qwen35::config::Config;
use crate::qwen35::mlp::Mlp;
use crate::qwen35::ops::interleaved;
use neura_decoder::{Cache, flatten, heads, matrix, partial, rms, vector};
use neura_gguf::Archive;
use neura_graph::{Graph, Shape, Value};

pub(crate) struct Attention<'g> {
    norm: Value<'g>,
    query: Value<'g>,
    key: Value<'g>,
    value: Value<'g>,
    output: Value<'g>,
    query_norm: Value<'g>,
    key_norm: Value<'g>,
    cache: Cache<'g>,
    mlp: Mlp<'g>,
    floor: Value<'g>,
    base: Value<'g>,
    scale: Value<'g>,
    rope: u32,
    heads: u32,
    key_heads: u32,
    width: u32,
}

impl<'g> Attention<'g> {
    pub(crate) fn new(
        graph: &Graph<'g>,
        source: &impl Archive,
        config: &Config,
        layer: u32,
        context: u32,
    ) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        let width = config.head_width;
        Self {
            norm: vector(graph, source, &at("attn_norm.weight"), config.hidden),
            query: matrix(
                graph,
                source,
                &at("attn_q.weight"),
                config.heads * width * 2,
                config.hidden,
            ),
            key: matrix(
                graph,
                source,
                &at("attn_k.weight"),
                config.key_heads * width,
                config.hidden,
            ),
            value: matrix(
                graph,
                source,
                &at("attn_v.weight"),
                config.key_heads * width,
                config.hidden,
            ),
            output: matrix(
                graph,
                source,
                &at("attn_output.weight"),
                config.hidden,
                config.heads * width,
            ),
            query_norm: vector(graph, source, &at("attn_q_norm.weight"), width),
            key_norm: vector(graph, source, &at("attn_k_norm.weight"), width),
            cache: Cache::new(graph, config.key_heads, width, context),
            mlp: Mlp::new(graph, source, layer, config.hidden, config.feed_forward),
            floor: graph.fill(Shape::scalar(), config.floor),
            base: graph.fill(Shape::scalar(), config.rope_base),
            scale: graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
            rope: config.rope_width,
            heads: config.heads,
            key_heads: config.key_heads,
            width,
        }
    }

    pub(crate) fn states(&self) -> [Value<'g>; 2] {
        self.cache.states()
    }

    pub(crate) fn forward(
        &self,
        graph: &Graph<'g>,
        hidden: Value<'g>,
        position: Value<'g>,
    ) -> Value<'g> {
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
        let query = partial(
            graph,
            rms(graph, query, self.query_norm, self.floor),
            position,
            self.base,
            self.rope,
        );
        let key = partial(
            graph,
            rms(graph, key, self.key_norm, self.floor),
            position,
            self.base,
            self.rope,
        );
        self.cache.write(graph, position, key, value);
        let attended = self.cache.attend(graph, query, position, self.scale);
        let gated = graph.mul(
            flatten(graph, attended),
            graph.sigmoid(flatten(graph, gate)),
        );
        let projected = graph.matmul(gated, graph.permute(self.output, [0, 1, 3, 2]));
        self.mlp
            .forward(graph, graph.add(hidden, projected), self.floor)
    }
}
