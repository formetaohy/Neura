use crate::config::Config;
use crate::mlp::Mlp;
use crate::model::{Planes, as_planes, as_rows};
use crate::persona::Route;
use neura::decoder::{heads, joined, l2, matrix, rms, softplus, vector};
use neura::gguf::Archive;
use neura::{AdapterBank, DeltaScan, Element, Graph, Init, Shape, Value};
use neura::{DeltaInput, DeltaRule};

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
    conv_state: Option<Value>,
    conv_shared: Option<Value>,
    rule: Option<DeltaRule>,
    scan: Option<DeltaScan>,
    route: Route,
    qkv_route: Option<AdapterBank>,
    gate_route: Option<AdapterBank>,
    output_route: Option<AdapterBank>,
    floor: Value,
    scale: Value,
    heads: u32,
    width: u32,
    kernel: u32,
    mixed: u32,
    rows: u32,
    batch: u32,
    prefix: bool,
    training: bool,
}

impl Delta {
    pub(crate) fn new(
        graph: &Graph,
        source: &impl Archive,
        config: &Config,
        layer: u32,
        planes: Planes,
        route: Route,
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
            training,
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
            mlp: Mlp::new(
                graph,
                source,
                layer,
                config.hidden,
                config.feed_forward,
                route,
            ),
            conv_state: (!training).then(|| {
                graph.state(
                    Shape::of([1, batch, config.conv_kernel - 1, mixed]),
                    Init::Zero,
                    Element::Single,
                )
            }),
            conv_shared: (!training && shared).then(|| {
                graph.state(
                    Shape::of([1, 1, config.conv_kernel - 1, mixed]),
                    Init::Zero,
                    Element::Single,
                )
            }),
            rule: (!training).then(|| match (shared, prefix) {
                (false, _) => DeltaRule::new(graph, config.linear_heads, width, batch, rows),
                (true, false) => {
                    DeltaRule::with_shared(graph, config.linear_heads, width, batch, rows)
                }
                (true, true) => DeltaRule::shared(graph, config.linear_heads, width, batch, rows),
            }),
            scan: training.then(|| DeltaScan::new(config.linear_heads, width, batch, rows)),
            qkv_route: route.adapter(graph, &at("delta.qkv"), config.hidden, mixed),
            gate_route: route.adapter(graph, &at("delta.gate"), config.hidden, held),
            output_route: route.adapter(graph, &at("delta.output"), held, config.hidden),
            route,
            floor: graph.fill(Shape::scalar(), config.floor),
            scale: graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
            heads: config.linear_heads,
            width,
            kernel: config.conv_kernel,
            mixed,
            rows,
            batch,
            prefix,
            training,
        }
    }

    pub(crate) fn reset(&self, graph: &Graph, reset: Value, share: Option<Value>) {
        let rule = self
            .rule
            .as_ref()
            .expect("a delta layer that trains holds no state to reset");
        let conv_state = self
            .conv_state
            .expect("a delta layer that trains holds no convolution to reset");
        rule.reset(graph, reset, share);
        let zero = graph.fill(Shape::scalar(), 0.0);
        match (self.conv_shared, share) {
            (Some(conv_shared), Some(share)) => {
                let shape = graph.shape(conv_state);
                let prefix = graph.broadcast_to(graph.slice(conv_shared, 1, 0, 1), shape);
                let blank = graph.select(reset, zero, conv_state);
                graph.copy_into(conv_state, graph.select(share, prefix, blank));
            }
            (None, None) => {
                graph.copy_into(conv_state, graph.select(reset, zero, conv_state));
            }
            _ => panic!(
                "a delta layer resets with the prefix it declares, and this layer declares {}",
                self.conv_shared.is_some(),
            ),
        }
    }

    pub(crate) fn clear_shared(&self, graph: &Graph, reset: Value) {
        let rule = self
            .rule
            .as_ref()
            .expect("a delta layer that trains holds no state to clear");
        rule.clear_shared(graph, reset);
        let conv_shared = self
            .conv_shared
            .expect("a delta layer that shares a prefix declares one");
        let zero = graph.fill(Shape::scalar(), 0.0);
        graph.copy_into(conv_shared, graph.select(reset, zero, conv_shared));
    }

    pub(crate) fn forward(&self, graph: &Graph, hidden: Value, mask: Value) -> Value {
        let conv_state = match (self.training, self.prefix) {
            (true, _) => graph.fill(Shape::of([1, self.batch, self.kernel - 1, self.mixed]), 0.0),
            (false, true) => self
                .conv_shared
                .expect("a delta layer that walks a shared prefix declares one"),
            (false, false) => self
                .conv_state
                .expect("a delta layer that walks its planes declares a state"),
        };
        let normed = rms(graph, hidden, self.norm, self.floor);
        let planes = |value| match self.training {
            true => value,
            false => as_planes(graph, value),
        };
        let mixed = planes(self.route.apply(
            graph,
            self.qkv_route.as_ref(),
            normed,
            graph.matmul(normed, graph.permute(self.qkv, [0, 1, 3, 2])),
        ));
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
        if !self.training {
            let carried = graph.slice(window, 2, self.rows, self.kernel - 1);
            graph.copy_into(conv_state, graph.select(mask, carried, conv_state));
        }
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
        let gate = planes(heads(
            graph,
            self.route.apply(
                graph,
                self.gate_route.as_ref(),
                normed,
                graph.matmul(normed, graph.permute(self.gate, [0, 1, 3, 2])),
            ),
            self.heads,
            self.width,
        ));
        let beta = graph.sigmoid(planes(heads(
            graph,
            graph.matmul(normed, graph.permute(self.beta, [0, 1, 3, 2])),
            self.heads,
            1,
        )));
        let alpha = planes(heads(
            graph,
            graph.matmul(normed, graph.permute(self.alpha, [0, 1, 3, 2])),
            self.heads,
            1,
        ));
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
        let attended = match &self.scan {
            Some(scan) => {
                let state = graph.fill(
                    Shape::of([self.heads, self.batch, self.width, self.width]),
                    0.0,
                );
                scan.forward(graph, state, rule_input).attended
            }
            None => self
                .rule
                .as_ref()
                .expect("a delta layer that answers holds a rule")
                .forward(graph, rule_input, Some(mask)),
        };
        let gated = graph.mul(
            rms(graph, attended, self.output_norm, self.floor),
            graph.silu(gate),
        );
        let wide = match self.training {
            true => joined(graph, gated, self.heads),
            false => as_rows(graph, joined(graph, gated, self.heads)),
        };
        let projected = self.route.apply(
            graph,
            self.output_route.as_ref(),
            wide,
            graph.matmul(wide, graph.permute(self.output, [0, 1, 3, 2])),
        );
        self.mlp
            .forward(graph, graph.add(hidden, projected), self.floor)
    }
}
