use crate::attention::Attention;
use crate::config::Config;
use crate::delta::Delta;
use neura::decoder::{matrix, rms, vector};
use neura::gguf::Archive;
use neura::{Element, Graph, Shape, Value};

pub(crate) fn as_planes(graph: &Graph, value: Value) -> Value {
    graph.permute(value, [0, 2, 1, 3])
}

pub(crate) fn as_rows(graph: &Graph, value: Value) -> Value {
    graph.permute(value, [0, 2, 1, 3])
}

#[derive(Clone, Copy)]
pub(crate) struct Planes {
    pub capacity: u32,
    pub window: u32,
    pub bounded: bool,
    pub batch: u32,
    pub rows: u32,
    pub shared: bool,
    pub prefix: bool,
    pub kv: Element,
}

pub struct Model {
    embedding: Value,
    output_norm: Value,
    floor: Value,
    layers: Vec<Layer>,
}

enum Layer {
    Attend(Box<Attention>),
    Delta(Box<Delta>),
}

impl Model {
    pub fn new(graph: &Graph, source: &impl Archive, config: &Config, planes: Planes) -> Self {
        let layers = (0..config.layers)
            .map(|layer| {
                if config.attends(layer) {
                    Layer::Attend(Box::new(Attention::new(
                        graph, source, config, layer, planes,
                    )))
                } else {
                    Layer::Delta(Box::new(Delta::new(graph, source, config, layer, planes)))
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

    pub fn reset(&self, graph: &Graph, reset: Value, share: Option<Value>) {
        for layer in &self.layers {
            match layer {
                Layer::Attend(attend) => attend.reset(graph, reset, share),
                Layer::Delta(delta) => delta.reset(graph, reset, share),
            }
        }
    }

    pub fn clear_shared(&self, graph: &Graph, reset: Value) {
        for layer in &self.layers {
            match layer {
                Layer::Attend(attend) => attend.clear_shared(graph, reset),
                Layer::Delta(delta) => delta.clear_shared(graph, reset),
            }
        }
    }

    pub fn step(&self, graph: &Graph, token: Value, cursor: Value, mask: Value) -> Value {
        let hidden = self.layers(graph, token, cursor, mask);
        let normed = rms(graph, hidden, self.output_norm, self.floor);
        graph.matmul(normed, graph.permute(self.embedding, [0, 1, 3, 2]))
    }

    pub fn prefill(&self, graph: &Graph, token: Value, cursor: Value, mask: Value) -> Value {
        let hidden = self.layers(graph, token, cursor, mask);
        graph.retain(hidden);
        hidden
    }

    fn layers(&self, graph: &Graph, token: Value, cursor: Value, mask: Value) -> Value {
        let plane_mask = graph.permute(mask, [0, 2, 1, 3]);
        let mut hidden = graph.mul(graph.gather(self.embedding, token), mask);
        for layer in &self.layers {
            hidden = match layer {
                Layer::Attend(attend) => attend.forward(graph, hidden, cursor, plane_mask),
                Layer::Delta(delta) => delta.forward(graph, hidden, plane_mask),
            };
            hidden = graph.mul(hidden, mask);
        }
        hidden
    }
}
