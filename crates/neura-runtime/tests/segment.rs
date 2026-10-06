use neura_abi::Element;
use neura_graph::{Graph, Shape, Value};
use neura_profile::{CooperativeMatrix, CooperativeTile, MatmulStrategy, MatmulTile, Profile};
use neura_runtime::{Backends, Product, Program, Runtime};

#[path = "support/backend.rs"]
mod backend;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

fn data(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..count)
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn expected(
    lengths: &[f32],
    packed: &[f32],
    weights: &[f32],
    depth: u32,
    columns: u32,
) -> Vec<f32> {
    let mut out = Vec::new();
    let mut row = 0usize;
    for (segment, length) in lengths.iter().enumerate() {
        let right =
            &weights[(segment as u32 * depth * columns) as usize..][..(depth * columns) as usize];
        for _ in 0..*length as u32 {
            let left = &packed[row * depth as usize..][..depth as usize];
            for column in 0..columns as usize {
                let mut total = 0.0f32;
                for step in 0..depth as usize {
                    total += left[step] * right[step * columns as usize + column];
                }
                out.push(total);
            }
            row += 1;
        }
    }
    out
}

struct Grouped {
    runtime: Runtime,
    graph: Graph<'static>,
    left: Value<'static>,
    weights: Value<'static>,
    lengths: Value<'static>,
    out: Value<'static>,
    offsets: Value<'static>,
    depth: u32,
    columns: u32,
}

impl Grouped {
    fn of(planes: u32, bound: u32, depth: u32, columns: u32) -> Self {
        Self::over(open(), planes, bound, depth, columns)
    }

    fn over(runtime: Runtime, planes: u32, bound: u32, depth: u32, columns: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(planes), Element::Single);
        let ragged = graph.ragged(bound, lengths);
        let left = graph.input(
            Shape::of([1, 1, bound, depth]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let weights = graph.input(Shape::of([planes, 1, depth, columns]), Element::Single);
        let out = graph.grouped_matmul(left, weights, ragged.offsets);
        graph.retain(out);
        graph.retain(ragged.offsets);
        Self {
            runtime,
            graph,
            left,
            weights,
            lengths,
            out,
            offsets: ragged.offsets,
            depth,
            columns,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn chosen(&self, profile: Profile, chosen: &[(Product, MatmulTile)]) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime
            .compile_chosen(&self.graph, &weights, profile, chosen)
    }

    fn step(&self, program: &Program, lengths: &[f32]) -> Vec<f32> {
        self.step_with(program, lengths, 1e-4)
    }

    fn step_with(&self, program: &Program, lengths: &[f32], tolerance: f32) -> Vec<f32> {
        let bound = self.graph.shape(self.left).dims()[2];
        let packed = data(bound * self.depth, 7);
        let weights = data(lengths.len() as u32 * self.depth * self.columns, 13);
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.left, &packed);
        self.runtime.write(program, self.weights, &weights);
        self.runtime.run(program);
        let produced = self.runtime.read(program, self.out);
        let scanned = self.runtime.read(program, self.offsets);
        assert_close(
            &produced,
            &expected(lengths, &packed, &weights, self.depth, self.columns),
            tolerance,
        );
        let mut running = 0.0f32;
        let mut offsets = vec![0.0f32; lengths.len() + 1];
        for (plane, length) in lengths.iter().enumerate() {
            running += length;
            offsets[plane + 1] = running;
        }
        assert_close(&scanned, &offsets, 1e-5);
        produced
    }
}

#[test]
fn a_grouped_product_weighs_the_rows_every_segment_packs() {
    let grouped = Grouped::of(3, 12, 4, 5);
    let program = grouped.compile();
    let produced = grouped.step(&program, &[3.0, 0.0, 7.0]);
    assert_eq!(produced.len(), 10 * 5);
}

#[test]
fn a_grouped_product_walks_a_segment_longer_than_its_tile() {
    let grouped = Grouped::of(4, 40, 8, 3);
    let program = grouped.compile();
    let produced = grouped.step(&program, &[17.0, 1.0, 9.0, 13.0]);
    assert_eq!(produced.len(), 40 * 3);
}

#[test]
fn a_grouped_product_walks_a_segment_the_tiles_a_short_bound_leaves() {
    let grouped = Grouped::of(1, 12, 8, 4);
    let tile = MatmulTile::new(MatmulStrategy::Streamed, 4, 4, 8, 1, 4);
    let profile = Profile::of(&[tile]);
    let program = grouped.chosen(profile, &[(Product::of(1, 12, 4, 8), tile)]);
    let produced = grouped.step(&program, &[5.0]);
    assert_eq!(
        produced.len(),
        5 * 4,
        "a segment walks one tile at a time, and a bound of two tiles leaves the second at the rows a binding holds",
    );
}

#[test]
fn a_grouped_product_walks_a_segment_the_default_tiles_leave_short() {
    let grouped = Grouped::of(1, 100, 8, 4);
    let program = grouped.compile();
    let produced = grouped.step(&program, &[50.0]);
    assert_eq!(produced.len(), 50 * 4);
}

#[test]
fn a_grouped_product_of_no_rows_holds_no_element() {
    let grouped = Grouped::of(3, 8, 4, 2);
    let program = grouped.compile();
    let produced = grouped.step(&program, &[0.0, 0.0, 0.0]);
    assert!(produced.is_empty(), "{produced:?} holds no row");
}

#[test]
fn a_grouped_product_splits_the_depth_of_a_long_pair() {
    let grouped = Grouped::of(2, 16, 128, 2);
    let program = grouped.compile();
    let produced = grouped.step(&program, &[5.0, 3.0]);
    assert_eq!(
        produced.len(),
        8 * 2,
        "the partials of a split depth walk the rows a binding holds",
    );
}

struct Planes {
    runtime: Runtime,
    graph: Graph<'static>,
    left: Value<'static>,
    weights: Value<'static>,
    mask: Value<'static>,
    out: Value<'static>,
    offsets: Value<'static>,
    depth: u32,
    columns: u32,
    bound: u32,
    planes: u32,
}

impl Planes {
    fn of(planes: u32, bound: u32, depth: u32, columns: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let live = graph.free(planes);
        let mask = graph.input(
            Shape::of([planes, 1, bound, 1]).freed(&[(0, live)]),
            Element::Single,
        );
        let lengths = graph.sum_axis(mask, 2);
        let ragged = graph.ragged(bound, lengths);
        let left = graph.input(
            Shape::of([1, 1, bound, depth]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let weights = graph.input(Shape::of([planes, 1, depth, columns]), Element::Single);
        let out = graph.grouped_matmul(left, weights, ragged.offsets);
        graph.retain(out);
        graph.retain(ragged.offsets);
        Self {
            runtime: open(),
            graph,
            left,
            weights,
            mask,
            out,
            offsets: ragged.offsets,
            depth,
            columns,
            bound,
            planes,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(&self, program: &Program, lengths: &[f32]) -> Vec<f32> {
        let packed = data(self.bound * self.depth, 7);
        let weights = data(self.planes * self.depth * self.columns, 13);
        let mut mask = vec![0.0f32; lengths.len() * self.bound as usize];
        for (plane, length) in lengths.iter().enumerate() {
            for at in 0..*length as u32 {
                mask[plane * self.bound as usize + at as usize] = 1.0;
            }
        }
        self.runtime.bind(program, &[lengths.len() as u32]);
        self.runtime.write(program, self.mask, &mask);
        self.runtime.write(program, self.left, &packed);
        self.runtime.write(program, self.weights, &weights);
        self.runtime.run(program);
        let produced = self.runtime.read(program, self.out);
        assert_close(
            &produced,
            &expected(lengths, &packed, &weights, self.depth, self.columns),
            1e-4,
        );
        let scanned = self.runtime.read(program, self.offsets);
        let mut running = 0.0f32;
        let mut offsets = vec![0.0f32; self.planes as usize + 1];
        for (plane, length) in lengths.iter().enumerate() {
            running += length;
            offsets[plane + 1] = running;
        }
        assert!(
            scanned.len() > lengths.len(),
            "the offsets table keeps its bound"
        );
        assert_close(
            &scanned[..lengths.len() + 1],
            &offsets[..lengths.len() + 1],
            1e-5,
        );
        produced
    }
}

#[test]
fn a_grouped_product_walks_the_lengths_a_device_counts() {
    let planes = Planes::of(3, 10, 4, 2);
    let program = planes.compile();
    let produced = planes.step(&program, &[4.0, 0.0, 6.0]);
    assert_eq!(produced.len(), 10 * 2);
}

#[test]
fn a_grouped_product_walks_the_segments_a_binding_holds() {
    let planes = Planes::of(4, 24, 4, 3);
    let program = planes.compile();
    let wide = planes.step(&program, &[9.0, 4.0, 2.0, 6.0]);
    assert_eq!(wide.len(), 21 * 3);
    let narrow = planes.step(&program, &[9.0, 4.0]);
    assert_eq!(
        narrow.len(),
        13 * 3,
        "the grouped product walks the rows of the segments a binding holds",
    );
    assert_close(&narrow, &wide[..13 * 3], 1e-4);
}

struct Fused {
    runtime: Runtime,
    graph: Graph<'static>,
    left: Value<'static>,
    weights: Value<'static>,
    bias: Value<'static>,
    lengths: Value<'static>,
    out: Value<'static>,
    depth: u32,
    columns: u32,
}

impl Fused {
    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn of(planes: u32, bound: u32, depth: u32, columns: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(planes), Element::Single);
        let ragged = graph.ragged(bound, lengths);
        let left = graph.input(
            Shape::of([1, 1, bound, depth]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let weights = graph.input(Shape::of([planes, 1, depth, columns]), Element::Single);
        let bias = graph.input(
            Shape::of([1, 1, bound, columns]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let product = graph.grouped_matmul(left, weights, ragged.offsets);
        let out = graph.add(product, bias);
        graph.retain(out);
        Self {
            runtime: open(),
            graph,
            left,
            weights,
            bias,
            lengths,
            out,
            depth,
            columns,
        }
    }

    fn step(&self, program: &Program, lengths: &[f32]) -> Vec<f32> {
        let bound = self.graph.shape(self.left).dims()[2];
        let packed = data(bound * self.depth, 7);
        let weights = data(lengths.len() as u32 * self.depth * self.columns, 13);
        let bias = data(bound * self.columns, 23);
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.left, &packed);
        self.runtime.write(program, self.weights, &weights);
        self.runtime.write(program, self.bias, &bias);
        self.runtime.run(program);
        let produced = self.runtime.read(program, self.out);
        let mut expected = expected(lengths, &packed, &weights, self.depth, self.columns);
        for (at, value) in expected.iter_mut().enumerate() {
            *value += bias
                [(at / self.columns as usize) * self.columns as usize + at % self.columns as usize];
        }
        assert_close(&produced, &expected, 1e-4);
        produced
    }
}

#[test]
fn a_grouped_product_weighs_the_rows_a_chain_operand_holds() {
    let fused = Fused::of(3, 12, 4, 5);
    let program = fused.compile();
    let produced = fused.step(&program, &[3.0, 0.0, 7.0]);
    assert_eq!(produced.len(), 10 * 5);
}

#[test]
fn a_grouped_product_weighs_the_rows_of_every_backend() {
    for backends in Backends::PLATFORM {
        let grouped = Grouped::over(backend::open_with(backends), 3, 12, 4, 5);
        let program = grouped.compile();
        let produced = grouped.step(&program, &[3.0, 0.0, 7.0]);
        assert_eq!(
            produced.len(),
            10 * 5,
            "{backends:?} walked the rows a ragged axis packs",
        );
    }
}

#[test]
fn a_grouped_product_weighs_the_rows_of_a_cooperative_tile() {
    let runtime = backend::open_with(Backends::VULKAN);
    let Some(matrix) = runtime.capability().cooperative_matrix else {
        return;
    };
    let fragment =
        CooperativeMatrix::new(matrix.subgroup, matrix.rows, matrix.columns, matrix.depth);
    let subgroups = 256 / matrix.subgroup;
    let tile = MatmulTile::cooperative(CooperativeTile::new(
        fragment,
        (subgroups, 1),
        (2, 2),
        fragment.depth(),
    ));
    let streamed = MatmulTile::new(MatmulStrategy::Streamed, 1, 256, 8, 1, 256);
    let profile = Profile::of(&[streamed, tile]);
    let grouped = Grouped::over(runtime, 3, 12, 4, 5);
    let program = grouped.chosen(profile, &[(Product::of(1, 12, 5, 4), tile)]);
    let produced = grouped.step_with(&program, &[3.0, 0.0, 7.0], 2e-2);
    assert_eq!(
        produced.len(),
        10 * 5,
        "a cooperative tile weighed the rows a ragged axis packs",
    );
}
