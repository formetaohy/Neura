use crate::config::Config;
use crate::mlp::Mlp;
use crate::model::{Planes, as_planes, as_rows};
use crate::ops::interleaved;
use neura::decoder::{Cache, heads, joined, matrix, partial, rms, vector};
use neura::gguf::Archive;
use neura::{Graph, Shape, Value};

pub(crate) struct Attention {
    norm: Value,
    query: Value,
    key: Value,
    value: Value,
    output: Value,
    query_norm: Value,
    key_norm: Value,
    cache: Cache,
    mlp: Mlp,
    floor: Value,
    base: Value,
    scale: Value,
    rope: u32,
    heads: u32,
    key_heads: u32,
    width: u32,
}

impl Attention {
    pub(crate) fn new(
        graph: &Graph,
        source: &impl Archive,
        config: &Config,
        layer: u32,
        planes: Planes,
    ) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        let width = config.head_width;
        let Planes {
            capacity,
            batch,
            rows,
            shared,
            prefix,
        } = planes;
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
            cache: match (shared, prefix) {
                (false, _) => Cache::new(graph, config.key_heads, width, capacity, batch, rows),
                (true, false) => {
                    Cache::with_shared(graph, config.key_heads, width, capacity, batch, rows)
                }
                (true, true) => {
                    Cache::shared(graph, config.key_heads, width, capacity, batch, rows)
                }
            },
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

    pub(crate) fn reset(&self, graph: &Graph, reset: Value, share: Option<Value>) {
        self.cache.reset(graph, reset, share);
    }

    pub(crate) fn clear_shared(&self, graph: &Graph, reset: Value) {
        self.cache.clear_shared(graph, reset);
    }

    pub(crate) fn forward(&self, graph: &Graph, hidden: Value, cursor: Value) -> Value {
        let normed = rms(graph, hidden, self.norm, self.floor);
        let packed = graph.matmul(normed, graph.permute(self.query, [0, 1, 3, 2]));
        let query = as_planes(graph, interleaved(graph, packed, self.heads, self.width, 0));
        let gate = as_planes(
            graph,
            interleaved(graph, packed, self.heads, self.width, self.width),
        );
        let key = as_planes(
            graph,
            heads(
                graph,
                graph.matmul(normed, graph.permute(self.key, [0, 1, 3, 2])),
                self.key_heads,
                self.width,
            ),
        );
        let value = as_planes(
            graph,
            heads(
                graph,
                graph.matmul(normed, graph.permute(self.value, [0, 1, 3, 2])),
                self.key_heads,
                self.width,
            ),
        );
        let query = partial(
            graph,
            rms(graph, query, self.query_norm, self.floor),
            cursor,
            self.base,
            self.rope,
        );
        let key = partial(
            graph,
            rms(graph, key, self.key_norm, self.floor),
            cursor,
            self.base,
            self.rope,
        );
        self.cache.write(graph, cursor, key, value);
        let attended = self.cache.attend(graph, query, cursor, self.scale);
        let gated = graph.mul(
            as_rows(graph, joined(graph, attended, self.heads)),
            graph.sigmoid(as_rows(graph, joined(graph, gate, self.heads))),
        );
        let projected = graph.matmul(gated, graph.permute(self.output, [0, 1, 3, 2]));
        self.mlp
            .forward(graph, graph.add(hidden, projected), self.floor)
    }
}
