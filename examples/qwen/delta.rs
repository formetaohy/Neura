use crate::config::Config;
use crate::mlp::Mlp;
use neura::decoder::{heads, joined, l2, matrix, rms, softplus, vector};
use neura::gguf::Archive;
use neura::{Element, Graph, Init, Shape, Value};

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
    rows: u32,
}

impl Delta {
    pub(crate) fn new(
        graph: &Graph,
        source: &impl Archive,
        config: &Config,
        layer: u32,
        rows: u32,
    ) -> Self {
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
            conv_state: graph.state(
                Shape::of([1, 1, config.conv_kernel - 1, mixed]),
                Init::Zero,
                Element::Single,
            ),
            state: graph.state(
                Shape::of([config.linear_heads, 1, width, width]),
                Init::Zero,
                Element::Single,
            ),
            floor: graph.fill(Shape::scalar(), config.floor),
            scale: graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
            heads: config.linear_heads,
            width,
            kernel: config.conv_kernel,
            rows,
        }
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
        let taps = graph.permute(self.conv, [0, 1, 3, 2]);
        let mut convolved = None;
        for tap in 0..self.kernel {
            let held = graph.mul(
                graph.slice(window, 2, tap, self.rows),
                graph.slice(taps, 2, tap, 1),
            );
            convolved = Some(match convolved {
                Some(sum) => graph.add(sum, held),
                None => held,
            });
        }
        let convolved = graph.silu(convolved.expect("a convolution walks the taps of its kernel"));
        graph.copy_into(
            self.conv_state,
            graph.slice(window, 2, self.rows, self.kernel - 1),
        );
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
        let mut state = self.state;
        let mut attended = Vec::with_capacity(self.rows as usize);
        for row in 0..self.rows {
            let query = graph.slice(query, 2, row, 1);
            let key = graph.slice(key, 2, row, 1);
            let value = graph.slice(value, 2, row, 1);
            let beta = graph.slice(beta, 2, row, 1);
            let decay = graph.slice(decay, 2, row, 1);
            let decayed = graph.mul(state, decay);
            let memory = graph.matmul(key, decayed);
            let delta = graph.mul(graph.sub(value, memory), beta);
            let grown = graph.add(
                decayed,
                graph.matmul(graph.permute(key, [0, 1, 3, 2]), delta),
            );
            attended.push(graph.matmul(query, grown));
            state = grown;
        }
        graph.copy_into(self.state, state);
        let attended = graph.concat(&attended, 2);
        let gated = graph.mul(
            rms(graph, attended, self.output_norm, self.floor),
            graph.silu(heads(graph, gate, self.heads, self.width)),
        );
        let projected = graph.matmul(
            joined(graph, gated, self.heads),
            graph.permute(self.output, [0, 1, 3, 2]),
        );
        self.mlp
            .forward(graph, graph.add(hidden, projected), self.floor)
    }
}
