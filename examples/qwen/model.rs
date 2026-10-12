use crate::config::Config;
use neura::gguf::GgufFile;
use neura::{AttentionOptions, Graph, Init, Shape, Value};

pub struct Model<'g> {
    embedding: Value<'g>,
    output_norm: Value<'g>,
    floor: Value<'g>,
    steps: Vec<Layer<'g>>,
}

enum Layer<'g> {
    Attend(Attend<'g>),
    Recur(Recur<'g>),
}

struct Attend<'g> {
    norm: Value<'g>,
    query: Value<'g>,
    key: Value<'g>,
    value: Value<'g>,
    output: Value<'g>,
    query_norm: Value<'g>,
    key_norm: Value<'g>,
    post: Value<'g>,
    gate: Value<'g>,
    up: Value<'g>,
    down: Value<'g>,
    keys: Value<'g>,
    values: Value<'g>,
    scale: Value<'g>,
    base: Value<'g>,
    rope: u32,
    heads: u32,
    key_heads: u32,
    width: u32,
}

struct Recur<'g> {
    norm: Value<'g>,
    qkv: Value<'g>,
    gate: Value<'g>,
    output: Value<'g>,
    conv: Value<'g>,
    alpha: Value<'g>,
    beta: Value<'g>,
    decay: Value<'g>,
    time: Value<'g>,
    scale: Value<'g>,
    post: Value<'g>,
    up: Value<'g>,
    down: Value<'g>,
    ffn_gate: Value<'g>,
    conv_state: Value<'g>,
    state: Value<'g>,
}

impl<'g> Model<'g> {
    pub fn new(graph: &Graph<'g>, source: &GgufFile, config: &Config, context: u32) -> Self {
        let width = config.head_width;
        let value_width = config.linear_heads * config.linear_width;
        let mixed = config.mixed_width();
        let floor = graph.fill(Shape::scalar(), config.floor);
        let base = graph.fill(Shape::scalar(), config.rope_base);
        let score = graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt());
        let steps = (0..config.layers)
            .map(|layer| {
                let at = |suffix: &str| format!("blk.{layer}.{suffix}");
                let norm = tensor(graph, source, &at("attn_norm.weight"));
                let post = tensor(graph, source, &at("post_attention_norm.weight"));
                let up = tensor(graph, source, &at("ffn_up.weight"));
                let down = tensor(graph, source, &at("ffn_down.weight"));
                assert_eq!(
                    graph.shape(up).dims()[2],
                    config.feed_forward,
                    "the feed forward of layer {layer} spans {} numbers where the metadata declares {}",
                    graph.shape(up).dims()[2],
                    config.feed_forward,
                );
                if config.attends(layer) {
                    Layer::Attend(Attend {
                        norm,
                        post,
                        up,
                        down,
                        query: tensor(graph, source, &at("attn_q.weight")),
                        key: tensor(graph, source, &at("attn_k.weight")),
                        value: tensor(graph, source, &at("attn_v.weight")),
                        output: tensor(graph, source, &at("attn_output.weight")),
                        query_norm: tensor(graph, source, &at("attn_q_norm.weight")),
                        key_norm: tensor(graph, source, &at("attn_k_norm.weight")),
                        gate: tensor(graph, source, &at("ffn_gate.weight")),
                        keys: graph.resident(
                            Shape::of([config.key_heads, 1, context, width]),
                            neura::Element::Single,
                        ),
                        values: graph.resident(
                            Shape::of([config.key_heads, 1, context, width]),
                            neura::Element::Single,
                        ),
                        scale: score,
                        base,
                        rope: config.rope_width,
                        heads: config.heads,
                        key_heads: config.key_heads,
                        width,
                    })
                } else {
                    Layer::Recur(Recur {
                        norm,
                        post,
                        up,
                        down,
                        qkv: tensor(graph, source, &at("attn_qkv.weight")),
                        gate: tensor(graph, source, &at("attn_gate.weight")),
                        output: tensor(graph, source, &at("ssm_out.weight")),
                        conv: tensor(graph, source, &at("ssm_conv1d.weight")),
                        alpha: tensor(graph, source, &at("ssm_alpha.weight")),
                        beta: tensor(graph, source, &at("ssm_beta.weight")),
                        decay: tensor(graph, source, &at("ssm_a")),
                        time: tensor(graph, source, &at("ssm_dt.bias")),
                        scale: tensor(graph, source, &at("ssm_norm.weight")),
                        ffn_gate: tensor(graph, source, &at("ffn_gate.weight")),
                        conv_state: graph.resident(
                            Shape::of([1, 1, config.conv_kernel - 1, mixed]),
                            neura::Element::Single,
                        ),
                        state: graph.resident(
                            Shape::of([
                                config.linear_heads,
                                1,
                                config.linear_width,
                                value_width / config.linear_heads,
                            ]),
                            neura::Element::Single,
                        ),
                    })
                }
            })
            .collect::<Vec<Layer<'g>>>();
        let _ = value_width;
        let embedding = tensor(graph, source, "token_embd.weight");
        assert_eq!(
            graph.shape(embedding).dims()[3],
            config.hidden,
            "the token embeddings hold {} numbers per token where the metadata declares {}",
            graph.shape(embedding).dims()[3],
            config.hidden,
        );
        assert_eq!(
            graph.shape(embedding).dims()[2],
            config.vocabulary,
            "the token embeddings hold {} tokens where the metadata declares {}",
            graph.shape(embedding).dims()[2],
            config.vocabulary,
        );
        Self {
            embedding,
            output_norm: tensor(graph, source, "output_norm.weight"),
            floor,
            steps,
        }
    }

    pub fn states(&self) -> Vec<Value<'g>> {
        let mut states = Vec::new();
        for layer in &self.steps {
            match layer {
                Layer::Attend(attend) => {
                    states.push(attend.keys);
                    states.push(attend.values);
                }
                Layer::Recur(recur) => {
                    states.push(recur.conv_state);
                    states.push(recur.state);
                }
            }
        }
        states
    }

    pub fn step(&self, graph: &Graph<'g>, token: Value<'g>, position: Value<'g>) -> Value<'g> {
        let mut hidden = graph.gather(self.embedding, token);
        for layer in &self.steps {
            hidden = match layer {
                Layer::Attend(attend) => self.attend(graph, attend, hidden, position),
                Layer::Recur(recur) => self.recur(graph, recur, hidden, position),
            };
        }
        let normed = rms(graph, hidden, self.output_norm, self.floor);
        let head = graph.permute(self.embedding, [0, 1, 3, 2]);
        graph.matmul(normed, head)
    }

    fn attend(
        &self,
        graph: &Graph<'g>,
        attend: &Attend<'g>,
        hidden: Value<'g>,
        position: Value<'g>,
    ) -> Value<'g> {
        let width = attend.width;
        let heads = attend.heads;
        let key_heads = attend.key_heads;
        let normed = rms(graph, hidden, attend.norm, self.floor);
        let packed = graph.matmul(normed, graph.permute(attend.query, [0, 1, 3, 2]));
        let query = graph.concat(
            &(0..heads)
                .map(|head| graph.slice(packed, 3, head * width * 2, width))
                .collect::<Vec<Value<'g>>>(),
            0,
        );
        let gate = graph.concat(
            &(0..heads)
                .map(|head| graph.slice(packed, 3, head * width * 2 + width, width))
                .collect::<Vec<Value<'g>>>(),
            0,
        );
        let key = heads_of(
            graph,
            graph.matmul(normed, graph.permute(attend.key, [0, 1, 3, 2])),
            key_heads,
            width,
        );
        let value = heads_of(
            graph,
            graph.matmul(normed, graph.permute(attend.value, [0, 1, 3, 2])),
            key_heads,
            width,
        );
        let query = partial_rope(
            graph,
            rms(graph, query, attend.query_norm, self.floor),
            position,
            attend.base,
            attend.rope,
        );
        let key = partial_rope(
            graph,
            rms(graph, key, attend.key_norm, self.floor),
            position,
            attend.base,
            attend.rope,
        );
        let seat = per_plane(graph, position, key_heads);
        graph.write_into(attend.keys, seat, key);
        graph.write_into(attend.values, seat, value);
        let attended = graph.attention(
            query,
            attend.keys,
            attend.values,
            AttentionOptions {
                scale: Some(attend.scale),
                causal: true,
                origin: Some(position),
                segments: None,
                reach: None,
                query_segments: None,
                bias: None,
            },
        );
        let flat = graph.mul(
            flatten(graph, attended),
            graph.sigmoid(flatten(graph, gate)),
        );
        let projected = graph.matmul(flat, graph.permute(attend.output, [0, 1, 3, 2]));
        let hidden = graph.add(hidden, projected);
        self.feed_forward(
            graph,
            attend.post,
            attend.gate,
            attend.up,
            attend.down,
            hidden,
        )
    }

    fn recur(
        &self,
        graph: &Graph<'g>,
        recur: &Recur<'g>,
        hidden: Value<'g>,
        _position: Value<'g>,
    ) -> Value<'g> {
        let heads = self.linear_heads(graph, recur);
        let width = graph.shape(recur.scale).dims()[3];
        let normed = rms(graph, hidden, recur.norm, self.floor);
        let mixed = graph.matmul(normed, graph.permute(recur.qkv, [0, 1, 3, 2]));
        let width_kernel = graph.shape(recur.conv_state).dims()[2];
        let window = graph.concat(
            &(0..width_kernel)
                .map(|row| graph.slice(recur.conv_state, 2, row, 1))
                .chain([mixed])
                .collect::<Vec<Value<'g>>>(),
            2,
        );
        let convolved = graph.silu(graph.sum_axis(
            graph.mul(window, graph.permute(recur.conv, [0, 1, 3, 2])),
            2,
        ));
        graph.copy_into(recur.conv_state, graph.slice(window, 2, 1, width_kernel));
        let query = graph.slice(convolved, 3, 0, heads * width);
        let key = graph.slice(convolved, 3, heads * width, heads * width);
        let value = graph.slice(convolved, 3, heads * width * 2, heads * width);
        let query = l2(graph, heads_of(graph, query, heads, width), self.floor);
        let key = l2(graph, heads_of(graph, key, heads, width), self.floor);
        let scale = graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt());
        let query = graph.mul(query, scale);
        let gate = graph.matmul(normed, graph.permute(recur.gate, [0, 1, 3, 2]));
        let beta = per_head(
            graph,
            graph.sigmoid(graph.matmul(normed, graph.permute(recur.beta, [0, 1, 3, 2]))),
            heads,
        );
        let alpha = per_head(
            graph,
            graph.matmul(normed, graph.permute(recur.alpha, [0, 1, 3, 2])),
            heads,
        );
        let paced = softplus(graph, graph.add(alpha, per_head(graph, recur.time, heads)));
        let decay =
            graph.exp(graph.neg(graph.mul(graph.exp(per_head(graph, recur.decay, heads)), paced)));
        let state = recur.state;
        let decayed = graph.mul(state, decay);
        let memory = graph.matmul(key, decayed);
        let delta = graph.mul(
            graph.sub(per_head_heads(graph, value, heads, width), memory),
            beta,
        );
        let grown = graph.add(
            decayed,
            graph.matmul(graph.permute(key, [0, 1, 3, 2]), delta),
        );
        graph.copy_into(state, grown);
        let attended = graph.matmul(query, grown);
        let normed = rms(graph, attended, recur.scale, self.floor);
        let gated = graph.mul(normed, graph.silu(heads_of(graph, gate, heads, width)));
        let flat = flatten(graph, gated);
        let projected = graph.matmul(flat, graph.permute(recur.output, [0, 1, 3, 2]));
        let hidden = graph.add(hidden, projected);
        self.feed_forward(
            graph,
            recur.post,
            recur.ffn_gate,
            recur.up,
            recur.down,
            hidden,
        )
    }

    fn linear_heads(&self, graph: &Graph<'g>, recur: &Recur<'g>) -> u32 {
        graph.shape(recur.state).dims()[0]
    }

    fn feed_forward(
        &self,
        graph: &Graph<'g>,
        norm: Value<'g>,
        gate: Value<'g>,
        up: Value<'g>,
        down: Value<'g>,
        hidden: Value<'g>,
    ) -> Value<'g> {
        let normed = rms(graph, hidden, norm, self.floor);
        let left = graph.silu(graph.matmul(normed, graph.permute(gate, [0, 1, 3, 2])));
        let right = graph.matmul(normed, graph.permute(up, [0, 1, 3, 2]));
        let product = graph.matmul(graph.mul(left, right), graph.permute(down, [0, 1, 3, 2]));
        graph.add(hidden, product)
    }
}

fn tensor<'g>(graph: &Graph<'g>, source: &GgufFile, name: &str) -> Value<'g> {
    let held = source
        .tensor(name)
        .unwrap_or_else(|| panic!("this gguf holds no tensor named {name}"));
    let shape = match held.rank {
        1 => Shape::vector(held.shape[0] as u32),
        2 => Shape::matrix(held.shape[1] as u32, held.shape[0] as u32),
        rank => panic!("the tensor {name} walks {rank} dimensions and holds no row layout"),
    };
    if held.element.block().is_some() {
        graph.named_block_quantized_parameter(name, shape, Init::Zero, held.element)
    } else {
        graph.named_parameter(name, shape, Init::Zero, held.element)
    }
}

fn heads_of<'g>(graph: &Graph<'g>, value: Value<'g>, heads: u32, width: u32) -> Value<'g> {
    let parts = (0..heads)
        .map(|head| graph.slice(value, 3, head * width, width))
        .collect::<Vec<Value<'g>>>();
    graph.concat(&parts, 0)
}

fn per_head_heads<'g>(graph: &Graph<'g>, value: Value<'g>, heads: u32, width: u32) -> Value<'g> {
    heads_of(graph, value, heads, width)
}

fn per_plane<'g>(graph: &Graph<'g>, value: Value<'g>, planes: u32) -> Value<'g> {
    let parts = (0..planes).map(|_| value).collect::<Vec<Value<'g>>>();
    graph.concat(&parts, 0)
}

fn per_head<'g>(graph: &Graph<'g>, value: Value<'g>, heads: u32) -> Value<'g> {
    let parts = (0..heads)
        .map(|head| graph.slice(value, 3, head, 1))
        .collect::<Vec<Value<'g>>>();
    graph.concat(&parts, 0)
}

fn flatten<'g>(graph: &Graph<'g>, value: Value<'g>) -> Value<'g> {
    let elements = graph.shape(value).elements();
    graph.reshape(value, Shape::of([1, 1, 1, elements]))
}

fn rms<'g>(graph: &Graph<'g>, value: Value<'g>, weight: Value<'g>, floor: Value<'g>) -> Value<'g> {
    let squares = graph.mul(value, value);
    let mean = graph.mean_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(mean, floor)));
    graph.mul(graph.mul(value, scale), weight)
}

fn l2<'g>(graph: &Graph<'g>, value: Value<'g>, floor: Value<'g>) -> Value<'g> {
    let squares = graph.mul(value, value);
    let total = graph.sum_axis(squares, 3);
    let scale = graph.recip(graph.sqrt(graph.add(total, floor)));
    graph.mul(value, scale)
}

fn softplus<'g>(graph: &Graph<'g>, value: Value<'g>) -> Value<'g> {
    let threshold = graph.fill(Shape::scalar(), 20.0);
    let raised = graph.log(graph.add(graph.fill(Shape::scalar(), 1.0), graph.exp(value)));
    graph.select(graph.greater(value, threshold), value, raised)
}

fn partial_rope<'g>(
    graph: &Graph<'g>,
    value: Value<'g>,
    position: Value<'g>,
    base: Value<'g>,
    width: u32,
) -> Value<'g> {
    let written = graph.shape(value).dims()[3];
    let front = graph.slice(value, 3, 0, width);
    let turned = graph.rope(front, Some(position), Some(base));
    if written == width {
        return turned;
    }
    let rest = graph.slice(value, 3, width, written - width);
    graph.concat(&[turned, rest], 3)
}
