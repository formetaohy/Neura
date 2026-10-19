use crate::attention::Attention;
use crate::config::Config;
use crate::delta::Delta;
use neura::decoder::{matrix, rms, vector};
use neura::gguf::Archive;
use neura::{Graph, Shape, Value};

pub(crate) fn hidden_mask(graph: &Graph, mask: Value) -> Value {
    graph.permute(mask, [0, 2, 1, 3])
}

pub(crate) fn as_planes(graph: &Graph, value: Value) -> Value {
    graph.permute(value, [0, 2, 1, 3])
}

pub(crate) fn as_rows(graph: &Graph, value: Value) -> Value {
    graph.permute(value, [0, 2, 1, 3])
}

pub struct Model {
    embedding: Value,
    output_norm: Value,
    floor: Value,
    layers: Vec<Layer>,
}

enum Layer {
    Attend(Attention),
    Delta(Delta),
}

impl Model {
    pub fn new(
        graph: &Graph,
        source: &impl Archive,
        config: &Config,
        capacity: u32,
        batch: u32,
    ) -> Self {
        let layers = (0..config.layers)
            .map(|layer| {
                if config.attends(layer) {
                    Layer::Attend(Attention::new(
                        graph, source, config, layer, capacity, batch,
                    ))
                } else {
                    Layer::Delta(Delta::new(graph, source, config, layer, batch))
                }
            })
            .collect::<Vec<Layer>>();
        Self {
            embedding: matrix(
                graph,
                source,
                "token_embd.weight",
                config.vocabulary,
                config.hidden,
            ),
            output_norm: vector(graph, source, "output_norm.weight", config.hidden),
            floor: graph.fill(Shape::scalar(), config.floor),
            layers,
        }
    }

    pub fn reset(&self, graph: &Graph, reset: Value) {
        for layer in &self.layers {
            match layer {
                Layer::Attend(attend) => attend.reset(graph, reset),
                Layer::Delta(delta) => delta.reset(graph, reset),
            }
        }
    }

    pub fn step(&self, graph: &Graph, token: Value, cursor: Value, mask: Value) -> Value {
        let row_mask = hidden_mask(graph, mask);
        let mut hidden = graph.mul(graph.gather(self.embedding, token), row_mask);
        for layer in &self.layers {
            hidden = match layer {
                Layer::Attend(attend) => attend.forward(graph, hidden, cursor),
                Layer::Delta(delta) => delta.forward(graph, hidden, mask),
            };
            hidden = graph.mul(hidden, row_mask);
        }
        let normed = rms(graph, hidden, self.output_norm, self.floor);
        graph.matmul(normed, graph.permute(self.embedding, [0, 1, 3, 2]))
    }
}
