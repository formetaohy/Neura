use neura_abi::Element;
use neura_gguf::Gguf;
use neura_graph::{Graph, Init, Shape, Value};
use neura_precision::{pack, unpack};
use neura_profile::{MatmulStrategy, Profile};
use neura_runtime::{Program, Runtime};

use crate::support;
use crate::support::gguf::build;
use crate::support::input::random;
use support::{assert_close, open};

const TOLERANCE: f32 = 1e-3;

#[derive(Clone, Copy)]
enum Facing {
    Reduction,
    Output,
}

struct Narrow {
    runtime: Runtime,
    graph: Graph,
    left: Value,
    stored: Value,
    out: Value,
    payload: Option<Vec<u8>>,
    values: Vec<f32>,
    inputs: Vec<f32>,
    facing: Facing,
    element: Element,
    depth: u32,
    columns: u32,
}

impl Narrow {
    fn of(element: Element, facing: Facing, depth: u32, columns: u32) -> Self {
        let runtime = open();
        let values = random(columns * depth, 23);
        let graph = Graph::new();
        let left = graph.input(Shape::matrix(1, depth), Element::Single);
        let shape = match facing {
            Facing::Reduction => Shape::matrix(columns, depth),
            Facing::Output => Shape::matrix(depth, columns),
        };
        let (stored, payload) = match (facing, element.block()) {
            (Facing::Output, _) => (
                graph.named_parameter(element.name(), shape, Init::Zero, element),
                None,
            ),
            (Facing::Reduction, Some(_)) => (
                graph.named_block_quantized_parameter(element.name(), shape, Init::Zero, element),
                Some(pack(element, 1.0, &values)),
            ),
            (Facing::Reduction, None) => (
                graph.named_parameter(element.name(), shape, Init::Zero, element),
                None,
            ),
        };
        let right = match facing {
            Facing::Reduction => graph.permute(stored, [0, 1, 3, 2]),
            Facing::Output => stored,
        };
        let out = graph.matmul(left, right);
        graph.retain(out);
        Self {
            runtime,
            graph,
            left,
            stored,
            out,
            payload,
            values,
            inputs: random(depth, 29),
            facing,
            element,
            depth,
            columns,
        }
    }

    fn program(&self, profile: Option<Profile>) -> Program {
        let weights = match &self.payload {
            Some(payload) => {
                let shape = [u64::from(self.columns), u64::from(self.depth)];
                let gguf = Gguf::decode(&build(32, &[(self.element, &shape, payload)]));
                self.runtime.load(&self.graph, &gguf)
            }
            None => self.runtime.weights(&self.graph),
        };
        match profile {
            Some(profile) => self.runtime.compile_with(&self.graph, &weights, profile),
            None => self.runtime.compile(&self.graph, &weights),
        }
    }

    fn reference(&self) -> Vec<f32> {
        let weight = match &self.payload {
            Some(payload) => unpack(self.element, self.values.len(), payload),
            None => self.values.clone(),
        };
        (0..self.columns as usize)
            .map(|column| {
                self.inputs
                    .iter()
                    .enumerate()
                    .map(|(step, value)| match self.facing {
                        Facing::Reduction => weight[column * self.depth as usize + step] * value,
                        Facing::Output => weight[step * self.columns as usize + column] * value,
                    })
                    .sum()
            })
            .collect()
    }

    fn run(&self, profile: Option<Profile>) -> (Program, Vec<f32>) {
        let program = self.program(profile);
        if self.payload.is_none() {
            self.runtime.write(&program, self.stored, &self.values);
        }
        self.runtime.write(&program, self.left, &self.inputs);
        self.runtime.run(&program);
        let produced = self.runtime.read(&program, self.out);
        assert_close(&produced, &self.reference(), TOLERANCE);
        (program, produced)
    }
}

fn strategies(program: &Program) -> Vec<MatmulStrategy> {
    program
        .matmul_geometries()
        .into_iter()
        .map(|(tile, _)| tile.strategy())
        .collect()
}

#[test]
fn a_narrow_product_walks_a_weight_its_reduction_stores() {
    let narrow = Narrow::of(Element::Single, Facing::Reduction, 128, 256);
    let (program, produced) = narrow.run(None);
    assert_eq!(produced.len(), 256);
    assert_eq!(
        strategies(&program),
        vec![MatmulStrategy::Folded],
        "a weight of one row reads its reduction through the lanes of a folded tile",
    );
}

#[test]
fn a_narrow_product_walks_a_weight_its_columns_stores() {
    let narrow = Narrow::of(Element::Single, Facing::Output, 128, 256);
    let (program, produced) = narrow.run(None);
    assert_eq!(produced.len(), 256);
    assert_eq!(
        strategies(&program),
        vec![MatmulStrategy::Streamed],
        "a weight of one row reads its columns through the lanes of a streamed tile",
    );
}

#[test]
fn a_narrow_product_walks_the_quantized_weight_a_gguf_stores_by_its_reduction() {
    for element in [Element::Q4K, Element::Q6K, Element::Q8_0] {
        let narrow = Narrow::of(element, Facing::Reduction, 256, 64);
        let smallest = narrow.runtime.profiles()[0];
        let (program, produced) = narrow.run(Some(smallest));
        assert_eq!(produced.len(), 64);
        assert_eq!(
            strategies(&program),
            vec![MatmulStrategy::Folded],
            "a {} weight the gguf stores by its reduction walks the folded tile",
            element.name(),
        );
    }
}

#[test]
fn a_tuned_narrow_product_measures_the_storage_its_weight_holds() {
    let narrow = Narrow::of(Element::Q4K, Facing::Reduction, 256, 64);
    let payload = narrow.payload.as_ref().expect("a quantized weight");
    let shape = [u64::from(narrow.columns), u64::from(narrow.depth)];
    let gguf = Gguf::decode(&build(32, &[(Element::Q4K, &shape, payload)]));
    let weights = narrow.runtime.load(&narrow.graph, &gguf);
    let program = narrow.runtime.tune(&narrow.graph, &weights);
    narrow.runtime.write(&program, narrow.left, &narrow.inputs);
    narrow.runtime.run(&program);
    assert_close(
        &narrow.runtime.read(&program, narrow.out),
        &narrow.reference(),
        TOLERANCE,
    );
    assert_eq!(
        strategies(&program),
        vec![MatmulStrategy::Folded],
        "a tuned plan keeps the tile the weight's own storage asks for",
    );
}

struct Segmented {
    runtime: Runtime,
    graph: Graph,
    left: Value,
    stored: Value,
    lengths: Value,
    out: Value,
    inputs: Vec<f32>,
    weights: Vec<f32>,
    depth: u32,
    columns: u32,
}

impl Segmented {
    fn of(planes: u32, bound: u32, depth: u32, columns: u32) -> Self {
        let runtime = open();
        let graph = Graph::new();
        let lengths = graph.input(Shape::vector(planes), Element::Single);
        let ragged = graph.ragged(bound, lengths);
        let left = graph.input(
            Shape::of([1, 1, bound, depth]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let stored = graph.input(Shape::of([planes, 1, columns, depth]), Element::Single);
        let out = graph.grouped_matmul(left, graph.permute(stored, [0, 1, 3, 2]), ragged.offsets);
        graph.retain(out);
        Self {
            runtime,
            graph,
            left,
            stored,
            lengths,
            out,
            inputs: random(bound * depth, 11),
            weights: random(planes * columns * depth, 13),
            depth,
            columns,
        }
    }

    fn reference(&self, lengths: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        let mut row = 0usize;
        for (segment, length) in lengths.iter().enumerate() {
            let plane = &self.weights[(segment as u32 * self.columns * self.depth) as usize..]
                [..(self.columns * self.depth) as usize];
            for _ in 0..*length as u32 {
                let left = &self.inputs[row * self.depth as usize..][..self.depth as usize];
                for column in 0..self.columns as usize {
                    let weight = &plane[column * self.depth as usize..][..self.depth as usize];
                    out.push(
                        left.iter()
                            .zip(weight)
                            .map(|(left, weight)| left * weight)
                            .sum(),
                    );
                }
                row += 1;
            }
        }
        out
    }

    fn run(&self, lengths: &[f32]) -> (Program, Vec<f32>) {
        let weights = self.runtime.weights(&self.graph);
        let program = self.runtime.compile(&self.graph, &weights);
        self.runtime.write(&program, self.lengths, lengths);
        self.runtime.write(&program, self.left, &self.inputs);
        self.runtime.write(&program, self.stored, &self.weights);
        self.runtime.run(&program);
        let produced = self.runtime.read(&program, self.out);
        assert_close(&produced, &self.reference(lengths), TOLERANCE);
        (program, produced)
    }
}

#[test]
fn a_narrow_product_streams_a_weight_whose_left_walks_elsewhere() {
    let runtime = open();
    let graph = Graph::new();
    let stored = graph.input(Shape::matrix(512, 256), Element::Single);
    let packed = graph.input(Shape::matrix(256, 2), Element::Single);
    let out = graph.matmul(
        graph.permute(packed, [0, 1, 3, 2]),
        graph.permute(stored, [0, 1, 3, 2]),
    );
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, stored, &random(512 * 256, 23));
    runtime.write(&program, packed, &random(256 * 2, 31));
    runtime.run(&program);
    assert_eq!(
        strategies(&program),
        vec![MatmulStrategy::Streamed],
        "a folded tile stages the reduction of its left operand, and a plan hands it a left that walks elsewhere to the streamed tile",
    );
}

#[test]
fn a_narrow_segmented_product_weighs_the_rows_a_reduction_stored_weight_packs() {
    for lengths in [
        [2.0f32, 0.0].as_slice(),
        [1.0, 1.0].as_slice(),
        [0.0, 2.0].as_slice(),
    ] {
        let segmented = Segmented::of(2, 2, 8, 4);
        let (program, produced) = segmented.run(lengths);
        let live: f32 = lengths.iter().sum();
        assert_eq!(produced.len(), live as usize * 4);
        assert_eq!(
            strategies(&program),
            vec![MatmulStrategy::Folded],
            "a segmented product of two rows reads its reduction through the lanes of a folded tile",
        );
    }
}
