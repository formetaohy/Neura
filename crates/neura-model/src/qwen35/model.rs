use crate::qwen35::attention::Attention;
use crate::qwen35::config::Config;
use crate::qwen35::delta::Delta;
use neura_decoder::{matrix, rms, vector};
use neura_gguf::Archive;
use neura_graph::{Graph, Shape, Value};

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
    pub fn new(graph: &Graph, source: &impl Archive, config: &Config, context: u32) -> Self {
        let layers = (0..config.layers)
            .map(|layer| {
                if config.attends(layer) {
                    Layer::Attend(Attention::new(graph, source, config, layer, context))
                } else {
                    Layer::Delta(Delta::new(graph, source, config, layer))
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

    pub fn states(&self) -> Vec<Value> {
        let mut states = Vec::new();
        for layer in &self.layers {
            match layer {
                Layer::Attend(attend) => states.extend(attend.states()),
                Layer::Delta(delta) => states.extend(delta.states()),
            }
        }
        states
    }

    pub fn step(&self, graph: &Graph, token: Value, position: Value) -> Value {
        let mut hidden = graph.gather(self.embedding, token);
        for layer in &self.layers {
            hidden = match layer {
                Layer::Attend(attend) => attend.forward(graph, hidden, position),
                Layer::Delta(delta) => delta.forward(graph, hidden),
            };
        }
        let normed = rms(graph, hidden, self.output_norm, self.floor);
        graph.matmul(normed, graph.permute(self.embedding, [0, 1, 3, 2]))
    }
}
