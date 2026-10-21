use crate::config::Config;
use crate::mlp::Mlp;
use crate::model::{Planes, as_planes, as_rows};
use crate::ops::interleaved;
use crate::persona::Route;
use neura::decoder::{Cache, CacheShape, heads, joined, matrix, partial, rms, vector};
use neura::gguf::Archive;
use neura::{Adapter, AttentionOptions, Graph, Shape, Value};

pub(crate) struct Attention {
    norm: Value,
    query: Value,
    key: Value,
    value: Value,
    output: Value,
    query_norm: Value,
    key_norm: Value,
    cache: Option<Cache>,
    mlp: Mlp,
    route: Route,
    query_route: Option<Adapter>,
    key_route: Option<Adapter>,
    value_route: Option<Adapter>,
    output_route: Option<Adapter>,
    floor: Value,
    base: Value,
    scale: Value,
    rope: u32,
    heads: u32,
    key_heads: u32,
    width: u32,
    training: bool,
}

impl Attention {
    pub(crate) fn new(
        graph: &Graph,
        source: &impl Archive,
        config: &Config,
        layer: u32,
        planes: Planes,
        route: Route,
    ) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        let width = config.head_width;
        let Planes {
            capacity,
            window,
            bounded,
            batch,
            rows,
            shared,
            prefix,
            kv,
            training,
            persona: _,
            training_persona: _,
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
            cache: (!training).then(|| {
                let shape = CacheShape::new(config.key_heads, width, capacity, window, batch, rows);
                match (shared, prefix, bounded) {
                    (false, _, false) => Cache::new(graph, shape, kv),
                    (false, _, true) => Cache::windowed(graph, shape, kv),
                    (true, false, false) => Cache::with_shared(graph, shape, kv),
                    (true, false, true) => Cache::windowed_with_shared(graph, shape, kv),
                    (true, true, false) => Cache::shared(graph, shape, kv),
                    (true, true, true) => Cache::windowed_shared(graph, shape, kv),
                }
            }),
            mlp: Mlp::new(
                graph,
                source,
                layer,
                config.hidden,
                config.feed_forward,
                route,
            ),
            query_route: route.adapter(
                graph,
                &at("attn.q"),
                config.hidden,
                config.heads * width * 2,
            ),
            key_route: route.adapter(
                graph,
                &at("attn.k"),
                config.hidden,
                config.key_heads * width,
            ),
            value_route: route.adapter(
                graph,
                &at("attn.v"),
                config.hidden,
                config.key_heads * width,
            ),
            output_route: route.adapter(
                graph,
                &at("attn.output"),
                config.heads * width,
                config.hidden,
            ),
            route,
            floor: graph.fill(Shape::scalar(), config.floor),
            base: graph.fill(Shape::scalar(), config.rope_base),
            scale: graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
            rope: config.rope_width,
            heads: config.heads,
            key_heads: config.key_heads,
            width,
            training,
        }
    }

    pub(crate) fn reset(&self, graph: &Graph, reset: Value, share: Option<Value>) {
        self.cache
            .as_ref()
            .expect("an attention that trains holds no cache to reset")
            .reset(graph, reset, share);
    }

    pub(crate) fn clear_shared(&self, graph: &Graph, reset: Value) {
        self.cache
            .as_ref()
            .expect("an attention that trains holds no cache to clear")
            .clear_shared(graph, reset);
    }

    pub(crate) fn forward(
        &self,
        graph: &Graph,
        hidden: Value,
        cursor: Value,
        mask: Value,
    ) -> Value {
        let normed = rms(graph, hidden, self.norm, self.floor);
        let packed = self.route.apply(
            graph,
            self.query_route.as_ref(),
            normed,
            graph.matmul(normed, graph.permute(self.query, [0, 1, 3, 2])),
        );
        let planes = |value| match self.training {
            true => value,
            false => as_planes(graph, value),
        };
        let query = planes(interleaved(graph, packed, self.heads, self.width, 0));
        let gate = planes(interleaved(
            graph, packed, self.heads, self.width, self.width,
        ));
        let key = planes(heads(
            graph,
            self.route.apply(
                graph,
                self.key_route.as_ref(),
                normed,
                graph.matmul(normed, graph.permute(self.key, [0, 1, 3, 2])),
            ),
            self.key_heads,
            self.width,
        ));
        let value = planes(heads(
            graph,
            self.route.apply(
                graph,
                self.value_route.as_ref(),
                normed,
                graph.matmul(normed, graph.permute(self.value, [0, 1, 3, 2])),
            ),
            self.key_heads,
            self.width,
        ));
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
        let attended = match &self.cache {
            Some(cache) => {
                cache.write_masked(graph, cursor, key, value, mask);
                cache.attend(graph, query, cursor, self.scale)
            }
            None => graph.attention(
                query,
                key,
                value,
                AttentionOptions {
                    scale: Some(self.scale),
                    causal: true,
                    origin: None,
                    segments: None,
                    reach: None,
                    query_segments: None,
                    bias: None,
                },
            ),
        };
        let rows = |value| match self.training {
            true => joined(graph, value, self.heads),
            false => as_rows(graph, joined(graph, value, self.heads)),
        };
        let gated = graph.mul(rows(attended), graph.sigmoid(rows(gate)));
        let projected = self.route.apply(
            graph,
            self.output_route.as_ref(),
            gated,
            graph.matmul(gated, graph.permute(self.output, [0, 1, 3, 2])),
        );
        self.mlp
            .forward(graph, graph.add(hidden, projected), self.floor)
    }
}
