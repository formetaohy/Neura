use crate::persona::Route;
use neura::decoder::{matrix, rms, vector};
use neura::gguf::Archive;
use neura::{AdapterBank, Graph, Value};

pub(crate) struct Mlp {
    norm: Value,
    gate: Value,
    up: Value,
    down: Value,
    route: Route,
    gate_route: Option<AdapterBank>,
    up_route: Option<AdapterBank>,
    down_route: Option<AdapterBank>,
}

impl Mlp {
    pub(crate) fn new(
        graph: &Graph,
        source: &impl Archive,
        layer: u32,
        hidden: u32,
        feed_forward: u32,
        route: Route,
    ) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        Self {
            norm: vector(graph, source, &at("post_attention_norm.weight"), hidden),
            gate: matrix(graph, source, &at("ffn_gate.weight"), feed_forward, hidden),
            up: matrix(graph, source, &at("ffn_up.weight"), feed_forward, hidden),
            down: matrix(graph, source, &at("ffn_down.weight"), hidden, feed_forward),
            gate_route: route.adapter(graph, &at("mlp.gate"), hidden, feed_forward),
            up_route: route.adapter(graph, &at("mlp.up"), hidden, feed_forward),
            down_route: route.adapter(graph, &at("mlp.down"), feed_forward, hidden),
            route,
        }
    }

    pub(crate) fn forward(&self, graph: &Graph, value: Value, floor: Value) -> Value {
        let normed = rms(graph, value, self.norm, floor);
        let gate = self.route.apply(
            graph,
            self.gate_route.as_ref(),
            normed,
            graph.matmul(normed, graph.permute(self.gate, [0, 1, 3, 2])),
        );
        let up = self.route.apply(
            graph,
            self.up_route.as_ref(),
            normed,
            graph.matmul(normed, graph.permute(self.up, [0, 1, 3, 2])),
        );
        let held = graph.mul(graph.silu(gate), up);
        let product = self.route.apply(
            graph,
            self.down_route.as_ref(),
            held,
            graph.matmul(held, graph.permute(self.down, [0, 1, 3, 2])),
        );
        graph.add(value, product)
    }
}
