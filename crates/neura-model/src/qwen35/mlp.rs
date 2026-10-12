use neura_decoder::{matrix, rms, vector};
use neura_gguf::Archive;
use neura_graph::{Graph, Value};

pub(crate) struct Mlp {
    norm: Value,
    gate: Value,
    up: Value,
    down: Value,
}

impl Mlp {
    pub(crate) fn new(
        graph: &Graph,
        source: &impl Archive,
        layer: u32,
        hidden: u32,
        feed_forward: u32,
    ) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        Self {
            norm: vector(graph, source, &at("post_attention_norm.weight"), hidden),
            gate: matrix(graph, source, &at("ffn_gate.weight"), feed_forward, hidden),
            up: matrix(graph, source, &at("ffn_up.weight"), feed_forward, hidden),
            down: matrix(graph, source, &at("ffn_down.weight"), hidden, feed_forward),
        }
    }

    pub(crate) fn forward(&self, graph: &Graph, value: Value, floor: Value) -> Value {
        let normed = rms(graph, value, self.norm, floor);
        let left = graph.silu(graph.matmul(normed, graph.permute(self.gate, [0, 1, 3, 2])));
        let right = graph.matmul(normed, graph.permute(self.up, [0, 1, 3, 2]));
        let product = graph.matmul(
            graph.mul(left, right),
            graph.permute(self.down, [0, 1, 3, 2]),
        );
        graph.add(value, product)
    }
}
