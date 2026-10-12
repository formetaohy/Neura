use crate::ops::{rms, tensor};
use neura::gguf::GgufFile;
use neura::{Graph, Value};

pub struct Mlp<'g> {
    norm: Value<'g>,
    gate: Value<'g>,
    up: Value<'g>,
    down: Value<'g>,
}

impl<'g> Mlp<'g> {
    pub fn build(
        graph: &Graph<'g>,
        source: &GgufFile,
        layer: u32,
        hidden: u32,
        feed_forward: u32,
    ) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        let norm = tensor(graph, source, &at("post_attention_norm.weight"));
        let gate = tensor(graph, source, &at("ffn_gate.weight"));
        let up = tensor(graph, source, &at("ffn_up.weight"));
        let down = tensor(graph, source, &at("ffn_down.weight"));
        assert_eq!(
            graph.shape(gate).dims(),
            [1, 1, feed_forward, hidden],
            "the gate of layer {layer} walks {:?} where {feed_forward} rows of {hidden} numbers walk [1, 1, {feed_forward}, {hidden}]",
            graph.shape(gate).dims(),
        );
        assert_eq!(
            graph.shape(up).dims(),
            graph.shape(gate).dims(),
            "the up projection of layer {layer} walks {:?} where its gate walks {:?}",
            graph.shape(up).dims(),
            graph.shape(gate).dims(),
        );
        assert_eq!(
            graph.shape(down).dims(),
            [1, 1, hidden, feed_forward],
            "the down projection of layer {layer} walks {:?} where {hidden} rows of {feed_forward} numbers walk [1, 1, {hidden}, {feed_forward}]",
            graph.shape(down).dims(),
        );
        assert_eq!(
            graph.shape(norm).dims(),
            [1, 1, 1, hidden],
            "the post attention norm of layer {layer} walks {:?} where {hidden} numbers walk [1, 1, 1, {hidden}]",
            graph.shape(norm).dims(),
        );
        Self {
            norm,
            gate,
            up,
            down,
        }
    }

    pub fn forward(&self, graph: &Graph<'g>, value: Value<'g>, floor: Value<'g>) -> Value<'g> {
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
