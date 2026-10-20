use crate::config::Config;
use crate::mlp::Mlp;
use crate::model::{Planes, as_planes, as_rows};
use neura::decoder::{heads, joined, l2, matrix, rms, softplus, vector};
use neura::gguf::Archive;
use neura::{DeltaInput, DeltaRule};
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
    conv_shared: Option<Value>,
    rule: DeltaRule,
    floor: Value,
    scale: Value,
    heads: u32,
    width: u32,
    kernel: u32,
    rows: u32,
    prefix: bool,
}

impl Delta {
    pub(crate) fn new(
        graph: &Graph,
        source: &impl Archive,
        config: &Config,
        layer: u32,
        planes: Planes,
    ) -> Self {
        let at = |suffix: &str| format!("blk.{layer}.{suffix}");
        let width = config.linear_width;
        let held = config.linear_width_total();
        let mixed = config.mixed_width();
        let Planes {
            batch,
            rows,
            shared,
            prefix,
            ..
        } = planes;
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
                Shape::of([1, batch, config.conv_kernel - 1, mixed]),
                Init::Zero,
                Element::Single,
            ),
            conv_shared: shared.then(|| {
                graph.state(
                    Shape::of([1, 1, config.conv_kernel - 1, mixed]),
                    Init::Zero,
                    Element::Single,
                )
            }),
            rule: match (shared, prefix) {
                (false, _) => DeltaRule::new(graph, config.linear_heads, width, batch, rows),
                (true, false) => {
                    DeltaRule::with_shared(graph, config.linear_heads, width, batch, rows)
                }
                (true, true) => DeltaRule::shared(graph, config.linear_heads, width, batch, rows),
            },
            floor: graph.fill(Shape::scalar(), config.floor),
            scale: graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
            heads: config.linear_heads,
            width,
            kernel: config.conv_kernel,
            rows,
            prefix,
        }
    }

    pub(crate) fn reset(&self, graph: &Graph, reset: Value, share: Option<Value>) {
        self.rule.reset(graph, reset, share);
        let zero = graph.fill(Shape::scalar(), 0.0);
        match (self.conv_shared, share) {
            (Some(conv_shared), Some(share)) => {
                let shape = graph.shape(self.conv_state);
                let prefix = graph.broadcast_to(graph.slice(conv_shared, 1, 0, 1), shape);
                let blank = graph.select(reset, zero, self.conv_state);
                graph.copy_into(self.conv_state, graph.select(share, prefix, blank));
            }
            (None, None) => {
                graph.copy_into(self.conv_state, graph.select(reset, zero, self.conv_state));
            }
            _ => panic!(
                "a delta layer resets with the prefix it declares, and this layer declares {}",
                self.conv_shared.is_some(),
            ),
        }
    }

    pub(crate) fn clear_shared(&self, graph: &Graph, reset: Value) {
        self.rule.clear_shared(graph, reset);
        let conv_shared = self
            .conv_shared
            .expect("a delta layer that shares a prefix declares one");
        let zero = graph.fill(Shape::scalar(), 0.0);
        graph.copy_into(conv_shared, graph.select(reset, zero, conv_shared));
    }

    pub(crate) fn forward(&self, graph: &Graph, hidden: Value, mask: Value) -> Value {
        let conv_state = match self.prefix {
            true => self
                .conv_shared
                .expect("a delta layer that walks a shared prefix declares one"),
            false => self.conv_state,
        };
        let normed = rms(graph, hidden, self.norm, self.floor);
        let mixed = as_planes(
            graph,
            graph.matmul(normed, graph.permute(self.qkv, [0, 1, 3, 2])),
        );
        let window = graph.concat(
            &(0..self.kernel - 1)
                .map(|row| graph.slice(conv_state, 2, row, 1))
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
        let carried = graph.slice(window, 2, self.rows, self.kernel - 1);
        graph.copy_into(conv_state, graph.select(mask, carried, conv_state));
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
        let gate = as_planes(
            graph,
            heads(
                graph,
                graph.matmul(normed, graph.permute(self.gate, [0, 1, 3, 2])),
                self.heads,
                self.width,
            ),
        );
        let beta = graph.sigmoid(as_planes(
            graph,
            heads(
                graph,
                graph.matmul(normed, graph.permute(self.beta, [0, 1, 3, 2])),
                self.heads,
                1,
            ),
        ));
        let alpha = as_planes(
            graph,
            heads(
                graph,
                graph.matmul(normed, graph.permute(self.alpha, [0, 1, 3, 2])),
                self.heads,
                1,
            ),
        );
        let paced = softplus(
            graph,
            graph.add(alpha, heads(graph, self.bias, self.heads, 1)),
        );
        let log_decay = graph.mul(heads(graph, self.decay, self.heads, 1), paced);
        let rule_input = DeltaInput {
            query,
            key,
            value,
            beta,
            log_decay,
        };
        let attended = if self.rows == 1 && !self.prefix {
            self.rule.advance(graph, rule_input, Some(mask))
        } else {
            self.rule.forward(graph, rule_input, Some(mask))
        };
        let gated = graph.mul(
            rms(graph, attended, self.output_norm, self.floor),
            graph.silu(gate),
        );
        let projected = graph.matmul(
            as_rows(graph, joined(graph, gated, self.heads)),
            graph.permute(self.output, [0, 1, 3, 2]),
        );
        self.mlp
            .forward(graph, graph.add(hidden, projected), self.floor)
    }
}
