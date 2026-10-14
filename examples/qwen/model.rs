use crate::attention::Attention;
use crate::config::Config;
use crate::delta::Delta;
use neura::decoder::{matrix, rms, vector};
use neura::gguf::Archive;
use neura::{Graph, Shape, Value};

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
        context: u32,
        rows: u32,
    ) -> Self {
        let layers = (0..config.layers)
            .map(|layer| {
                if config.attends(layer) {
                    Layer::Attend(Attention::new(graph, source, config, layer, context, rows))
                } else {
                    Layer::Delta(Delta::new(graph, source, config, layer, rows))
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

    pub fn step(&self, graph: &Graph, token: Value, cursor: Value) -> Value {
        let mut hidden = graph.gather(self.embedding, token);
        for layer in &self.layers {
            hidden = match layer {
                Layer::Attend(attend) => attend.forward(graph, hidden, cursor),
                Layer::Delta(delta) => delta.forward(graph, hidden),
            };
        }
        let normed = rms(graph, hidden, self.output_norm, self.floor);
        graph.matmul(normed, graph.permute(self.embedding, [0, 1, 3, 2]))
    }
}
