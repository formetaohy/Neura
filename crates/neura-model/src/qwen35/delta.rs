use crate::qwen35::config::Config;
use crate::qwen35::mlp::Mlp;
use neura_abi::Element;
use neura_decoder::{flatten, heads, l2, matrix, rms, softplus, vector};
use neura_gguf::Archive;
use neura_graph::{Graph, Shape, Value};

pub(crate) struct Delta {
    norm: Value,
    qkv: Value,
    gate: Value,
    output: Value,
    conv: Value,
    alpha: Value,
    beta: Value,
    decay: Value,
    bias: Value,
    output_norm: Value,
    mlp: Mlp,
    conv_state: Value,
    state: Value,
    floor: Value,
    scale: Value,
    heads: u32,
    width: u32,
    kernel: u32,
}

impl Delta {
    pub(crate) fn new(graph: &Graph, source: &impl Archive, config: &Config, layer: u32) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        let width = config.linear_width;
        let held = config.linear_width_total();
        let mixed = config.mixed_width();
        Self {
            norm: vector(graph, source, &at("attn_norm.weight"), config.hidden),
            qkv: matrix(graph, source, &at("attn_qkv.weight"), mixed, config.hidden),
            gate: matrix(graph, source, &at("attn_gate.weight"), held, config.hidden),
            output: matrix(graph, source, &at("ssm_out.weight"), config.hidden, held),
            conv: matrix(
                graph,
                source,
                &at("ssm_conv1d.weight"),
                mixed,
                config.conv_kernel,
            ),
            alpha: matrix(
                graph,
                source,
                &at("ssm_alpha.weight"),
                config.linear_heads,
                config.hidden,
            ),
            beta: matrix(
                graph,
                source,
                &at("ssm_beta.weight"),
                config.linear_heads,
                config.hidden,
            ),
            decay: vector(graph, source, &at("ssm_a"), config.linear_heads),
            bias: vector(graph, source, &at("ssm_dt.bias"), config.linear_heads),
            output_norm: vector(graph, source, &at("ssm_norm.weight"), width),
            mlp: Mlp::new(graph, source, layer, config.hidden, config.feed_forward),
            conv_state: graph.resident(
                Shape::of([1, 1, config.conv_kernel - 1, mixed]),
                Element::Single,
            ),
            state: graph.resident(
                Shape::of([config.linear_heads, 1, width, width]),
                Element::Single,
            ),
            floor: graph.fill(Shape::scalar(), config.floor),
            scale: graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
            heads: config.linear_heads,
            width,
            kernel: config.conv_kernel,
        }
    }

    pub(crate) fn states(&self) -> [Value; 2] {
        [self.conv_state, self.state]
    }

    pub(crate) fn forward(&self, graph: &Graph, hidden: Value) -> Value {
        let normed = rms(graph, hidden, self.norm, self.floor);
        let mixed = graph.matmul(normed, graph.permute(self.qkv, [0, 1, 3, 2]));
        let window = graph.concat(
            &(0..self.kernel - 1)
                .map(|row| graph.slice(self.conv_state, 2, row, 1))
                .chain([mixed])
                .collect::<Vec<Value>>(),
            2,
        );
        let convolved = graph
            .silu(graph.sum_axis(graph.mul(window, graph.permute(self.conv, [0, 1, 3, 2])), 2));
        graph.copy_into(self.conv_state, graph.slice(window, 2, 1, self.kernel - 1));
        let held = self.heads * self.width;
        let query = graph.mul(
            l2(
                graph,
                heads(
                    graph,
                    graph.slice(convolved, 3, 0, held),
                    self.heads,
                    self.width,
                ),
                self.floor,
            ),
            self.scale,
        );
        let key = l2(
            graph,
            heads(
                graph,
                graph.slice(convolved, 3, held, held),
                self.heads,
                self.width,
            ),
            self.floor,
        );
        let value = heads(
            graph,
            graph.slice(convolved, 3, held * 2, held),
            self.heads,
            self.width,
        );
        let gate = graph.matmul(normed, graph.permute(self.gate, [0, 1, 3, 2]));
        let beta = graph.sigmoid(heads(
            graph,
            graph.matmul(normed, graph.permute(self.beta, [0, 1, 3, 2])),
            self.heads,
            1,
        ));
        let alpha = heads(
            graph,
            graph.matmul(normed, graph.permute(self.alpha, [0, 1, 3, 2])),
            self.heads,
            1,
        );
        let paced = softplus(
            graph,
            graph.add(alpha, heads(graph, self.bias, self.heads, 1)),
        );
        let decay = graph.exp(graph.mul(heads(graph, self.decay, self.heads, 1), paced));
        let decayed = graph.mul(self.state, decay);
        let memory = graph.matmul(key, decayed);
        let delta = graph.mul(graph.sub(value, memory), beta);
        let grown = graph.add(
            decayed,
            graph.matmul(graph.permute(key, [0, 1, 3, 2]), delta),
        );
        graph.copy_into(self.state, grown);
        let attended = graph.matmul(query, grown);
        let gated = graph.mul(
            rms(graph, attended, self.output_norm, self.floor),
            graph.silu(heads(graph, gate, self.heads, self.width)),
        );
        let projected = graph.matmul(
            flatten(graph, gated),
            graph.permute(self.output, [0, 1, 3, 2]),
        );
        self.mlp
            .forward(graph, graph.add(hidden, projected), self.floor)
    }
}
