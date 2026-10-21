use neura::gguf::GgufFile;
use neura::{
    Adapter, AdapterShape, CheckpointFile, Element, Graph, Init, Overlay, Placed, Shape, Source,
    Value,
};
use std::path::Path;

const PREFIX: &str = "adapter.";
const SPREAD: f32 = 0.1;

#[derive(Clone, Copy)]
pub(crate) struct Plan {
    count: u32,
    rank: u32,
}

impl Plan {
    pub(crate) fn new(count: u32, rank: u32) -> Self {
        assert!(
            count > 0 && rank > 0,
            "a persona plan of {count} personas of rank {rank} carries no bypass",
        );
        Self { count, rank }
    }

    pub(crate) fn count(self) -> u32 {
        self.count
    }

    pub(crate) fn rank(self) -> u32 {
        self.rank
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Route {
    plan: Option<Plan>,
    selection: Option<Value>,
}

impl Route {
    pub(crate) fn new(graph: &Graph, batch: u32, plan: Option<Plan>) -> Self {
        let selection =
            plan.map(|_| graph.state(Shape::of([1, 1, batch, 1]), Init::Zero, Element::Single));
        Self { plan, selection }
    }

    pub(crate) fn selection(self) -> Option<Value> {
        self.selection
    }

    pub(crate) fn adapter(
        self,
        graph: &Graph,
        name: &str,
        inputs: u32,
        outputs: u32,
    ) -> Option<Adapter> {
        let plan = self.plan?;
        Some(Adapter::new(
            graph,
            &format!("{PREFIX}{name}"),
            AdapterShape::new(plan.count, inputs, outputs, plan.rank),
            Init::Zero,
            Element::Single,
            1.0,
        ))
    }

    pub(crate) fn apply(
        self,
        graph: &Graph,
        adapter: Option<&Adapter>,
        input: Value,
        base: Value,
    ) -> Value {
        let (Some(adapter), Some(selection)) = (adapter, self.selection) else {
            return base;
        };
        let wide_input = graph.permute(input, [2, 0, 1, 3]);
        let wide_base = graph.permute(base, [2, 0, 1, 3]);
        let picked = graph.permute(selection, [2, 0, 1, 3]);
        let routed = adapter.forward_shared(graph, wide_input, wide_base, picked);
        graph.permute(routed, [1, 2, 0, 3])
    }
}

pub(crate) enum Table {
    Random { seed: u64 },
    Trained(CheckpointFile),
}

impl Table {
    pub(crate) fn open(path: Option<&str>, seed: u64) -> Self {
        match path {
            None | Some("random") => Self::Random { seed },
            Some(path) => Self::Trained(CheckpointFile::open(Path::new(path))),
        }
    }

    pub(crate) fn source<'t>(&'t self, graph: &Graph) -> TableSource<'t> {
        let tensors = match self {
            Self::Random { .. } => graph
                .snapshot()
                .values()
                .iter()
                .filter_map(|held| {
                    let name = held.name.as_deref()?;
                    name.starts_with(PREFIX)
                        .then(|| (name.to_owned(), u64::from(held.shape.elements())))
                })
                .collect(),
            Self::Trained(_) => Vec::new(),
        };
        TableSource {
            table: self,
            tensors,
        }
    }
}

pub(crate) struct TableSource<'t> {
    table: &'t Table,
    tensors: Vec<(String, u64)>,
}

impl Source for TableSource<'_> {
    fn placed(&self, name: &str) -> Option<Placed> {
        match self.table {
            Table::Random { .. } => {
                self.tensors
                    .iter()
                    .find(|(held, _)| held == name)
                    .map(|(_, elements)| Placed {
                        element: Element::Single,
                        elements: *elements,
                        scale: 1.0,
                        payload_bytes: *elements * 4,
                        table_bytes: 0,
                    })
            }
            Table::Trained(file) => file.placed(name),
        }
    }

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        match self.table {
            Table::Random { seed } => {
                let elements = self
                    .tensors
                    .iter()
                    .find(|(held, _)| held == name)
                    .map(|(_, elements)| *elements)
                    .unwrap_or_else(|| {
                        panic!("this untrained persona table holds no tensor named {name}")
                    });
                sink(&tensor(name, elements, *seed));
            }
            Table::Trained(file) => file.payload(name, sink),
        }
    }

    fn table(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        match self.table {
            Table::Random { .. } => {
                panic!(
                    "this untrained persona table quantizes no tensor, and {name} asks for quanta"
                )
            }
            Table::Trained(file) => file.table(name, sink),
        }
    }
}

pub(crate) fn load(
    runtime: &neura::Runtime,
    graph: &Graph,
    source: &GgufFile,
    table: Option<&Table>,
) -> neura::Weights {
    match table {
        Some(table) => runtime.load(graph, &Overlay::new(source, table.source(graph))),
        None => runtime.load(graph, source),
    }
}

fn tensor(name: &str, elements: u64, seed: u64) -> Vec<u8> {
    let held = name.ends_with(".scale");
    let mut state = seed ^ hash(name);
    let mut bytes = Vec::with_capacity(elements as usize * 4);
    for _ in 0..elements {
        let value = match held {
            true => 1.0,
            false => {
                state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
                let mut mixed = state;
                mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                mixed ^= mixed >> 31;
                ((mixed >> 40) as f32 / 16_777_216.0 - 0.5) * SPREAD
            }
        };
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

fn hash(name: &str) -> u64 {
    name.bytes().fold(0xcbf2_9ce4_8422_2325, |held, byte| {
        (held ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3)
    })
}
