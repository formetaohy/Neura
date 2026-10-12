use crate::config::Config;
use crate::mlp::Mlp;
use crate::ops::{flatten, heads, l2, rms, softplus, tensor};
use neura::gguf::GgufFile;
use neura::{Graph, Shape, Value};

pub struct Delta<'g> {
    norm: Value<'g>,
    qkv: Value<'g>,
    gate: Value<'g>,
    output: Value<'g>,
    conv: Value<'g>,
    alpha: Value<'g>,
    beta: Value<'g>,
    decay: Value<'g>,
    bias: Value<'g>,
    output_norm: Value<'g>,
    mlp: Mlp<'g>,
    conv_state: Value<'g>,
    state: Value<'g>,
    floor: Value<'g>,
    scale: Value<'g>,
    heads: u32,
    width: u32,
    kernel: u32,
}

impl<'g> Delta<'g> {
    pub fn build(graph: &Graph<'g>, source: &GgufFile, config: &Config, layer: u32) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        let width = config.linear_width;
        let held = config.linear_width_total();
        let mixed = config.mixed_width();
        let norm = tensor(graph, source, &at("attn_norm.weight"));
        let qkv = tensor(graph, source, &at("attn_qkv.weight"));
        let gate = tensor(graph, source, &at("attn_gate.weight"));
        let output = tensor(graph, source, &at("ssm_out.weight"));
        let conv = tensor(graph, source, &at("ssm_conv1d.weight"));
        let alpha = tensor(graph, source, &at("ssm_alpha.weight"));
        let beta = tensor(graph, source, &at("ssm_beta.weight"));
        let decay = tensor(graph, source, &at("ssm_a"));
        let bias = tensor(graph, source, &at("ssm_dt.bias"));
        let output_norm = tensor(graph, source, &at("ssm_norm.weight"));
        assert_eq!(
            graph.shape(qkv).dims(),
            [1, 1, mixed, config.hidden],
            "the projection of layer {layer} walks {:?} where a query, a key and a value of {mixed} numbers walk [1, 1, {mixed}, {}]",
            graph.shape(qkv).dims(),
            config.hidden,
        );
        assert_eq!(
            graph.shape(conv).dims(),
            [1, 1, mixed, config.conv_kernel],
            "the convolution of layer {layer} walks {:?} where {mixed} channels of {} taps walk [1, 1, {mixed}, {}]",
            graph.shape(conv).dims(),
            config.conv_kernel,
            config.conv_kernel,
        );
        assert_eq!(
            graph.shape(gate).dims(),
            [1, 1, held, config.hidden],
            "the gate of layer {layer} walks {:?} where {held} numbers walk [1, 1, {held}, {}]",
            graph.shape(gate).dims(),
            config.hidden,
        );
        assert_eq!(
            graph.shape(output).dims(),
            [1, 1, config.hidden, held],
            "the output of layer {layer} walks {:?} where {} rows of {held} numbers walk [1, 1, {}, {held}]",
            graph.shape(output).dims(),
            config.hidden,
            config.hidden,
        );
        assert_eq!(
            graph.shape(alpha).dims(),
            [1, 1, config.linear_heads, config.hidden],
            "the alpha of layer {layer} walks {:?} where {} heads walk [1, 1, {}, {}]",
            graph.shape(alpha).dims(),
            config.linear_heads,
            config.linear_heads,
            config.hidden,
        );
        assert_eq!(
            graph.shape(beta).dims(),
            graph.shape(alpha).dims(),
            "the beta of layer {layer} walks {:?} where its alpha walks {:?}",
            graph.shape(beta).dims(),
            graph.shape(alpha).dims(),
        );
        assert_eq!(
            graph.shape(decay).dims(),
            [1, 1, 1, config.linear_heads],
            "the decay of layer {layer} walks {:?} where {} heads walk [1, 1, 1, {}]",
            graph.shape(decay).dims(),
            config.linear_heads,
            config.linear_heads,
        );
        assert_eq!(
            graph.shape(bias).dims(),
            graph.shape(decay).dims(),
            "the step of layer {layer} walks {:?} where its decay walks {:?}",
            graph.shape(bias).dims(),
            graph.shape(decay).dims(),
        );
        assert_eq!(
            graph.shape(output_norm).dims(),
            [1, 1, 1, width],
            "the output norm of layer {layer} walks {:?} where {width} numbers walk [1, 1, 1, {width}]",
            graph.shape(output_norm).dims(),
        );
        Self {
            norm,
            qkv,
            gate,
            output,
            conv,
            alpha,
            beta,
            decay,
            bias,
            output_norm,
            mlp: Mlp::build(graph, source, layer, config.hidden, config.feed_forward),
            conv_state: graph.resident(
                Shape::of([1, 1, config.conv_kernel - 1, mixed]),
                neura::Element::Single,
            ),
            state: graph.resident(
                Shape::of([config.linear_heads, 1, width, width]),
                neura::Element::Single,
            ),
            floor: graph.fill(Shape::scalar(), config.floor),
            scale: graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
            heads: config.linear_heads,
            width,
            kernel: config.conv_kernel,
        }
    }

    pub fn states(&self) -> [Value<'g>; 2] {
        [self.conv_state, self.state]
    }

    pub fn forward(&self, graph: &Graph<'g>, hidden: Value<'g>) -> Value<'g> {
        let normed = rms(graph, hidden, self.norm, self.floor);
        let mixed = graph.matmul(normed, graph.permute(self.qkv, [0, 1, 3, 2]));
        let window = graph.concat(
            &(0..self.kernel - 1)
                .map(|row| graph.slice(self.conv_state, 2, row, 1))
                .chain([mixed])
                .collect::<Vec<Value<'g>>>(),
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
