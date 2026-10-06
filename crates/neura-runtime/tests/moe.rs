use neura_abi::Element;
use neura_graph::{Graph, Shape, Value};
use neura_runtime::{Program, Runtime};

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

struct Packed<'a> {
    lengths: &'a [f32],
    left: &'a [f32],
    weights: &'a [f32],
    weight: &'a [f32],
    depth: u32,
    columns: u32,
}

fn reference(packed: Packed<'_>) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let Packed {
        lengths,
        left,
        weights,
        weight,
        depth,
        columns,
    } = packed;
    let depth = depth as usize;
    let columns = columns as usize;
    let live: usize = lengths.iter().map(|length| *length as usize).sum();
    let mut out = vec![0.0f32; live * columns];
    let mut left_gradient = vec![0.0f32; live * depth];
    let mut weight_gradient = vec![0.0f32; lengths.len() * depth * columns];
    let mut row = 0usize;
    for (segment, length) in lengths.iter().enumerate() {
        let filter = &weights[segment * depth * columns..][..depth * columns];
        for _ in 0..*length as usize {
            let source = &left[row * depth..][..depth];
            let incoming = &weight[row * columns..][..columns];
            for column in 0..columns {
                let incoming = incoming[column];
                let mut total = 0.0f32;
                for step in 0..depth {
                    total += source[step] * filter[step * columns + column];
                    left_gradient[row * depth + step] += filter[step * columns + column] * incoming;
                    weight_gradient[(segment * depth + step) * columns + column] +=
                        source[step] * incoming;
                }
                out[row * columns + column] = total;
            }
            row += 1;
        }
    }
    (out, left_gradient, weight_gradient)
}

struct Segmented {
    runtime: Runtime,
    graph: Graph<'static>,
    lengths: Value<'static>,
    counted: Option<Value<'static>>,
    left: Value<'static>,
    weights: Value<'static>,
    weight: Value<'static>,
    out: Value<'static>,
    offsets: Value<'static>,
    left_gradient: Value<'static>,
    weight_gradient: Value<'static>,
    depth: u32,
    columns: u32,
    bound: u32,
    planes: u32,
}

impl Segmented {
    fn of(planes: u32, bound: u32, depth: u32, columns: u32) -> Self {
        Self::build(planes, bound, depth, columns, false)
    }

    fn counted(planes: u32, bound: u32, depth: u32, columns: u32) -> Self {
        Self::build(planes, bound, depth, columns, true)
    }

    fn build(planes: u32, bound: u32, depth: u32, columns: u32, counted: bool) -> Self {
        let graph: Graph<'static> = Graph::new();
        let mask = counted.then(|| graph.input(Shape::of([planes, 1, bound, 1]), Element::Single));
        let lengths = match mask {
            Some(mask) => graph.sum_axis(mask, 2),
            None => graph.input(Shape::vector(planes), Element::Single),
        };
        let ragged = graph.ragged(bound, lengths);
        let left = graph.gradient_input(
            Shape::of([1, 1, bound, depth]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let weights = graph.gradient_input(Shape::of([planes, 1, depth, columns]), Element::Single);
        let out = graph.grouped_matmul(left, weights, ragged.offsets);
        let weight = graph.gradient_input(
            Shape::of([1, 1, bound, columns]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let loss = graph.sum(graph.mul(out, weight));
        let collected = graph.backward(loss);
        let left_gradient = collected.of(left);
        let weight_gradient = collected.of(weights);
        graph.retain(out);
        graph.retain(left_gradient);
        graph.retain(weight_gradient);
        graph.retain(ragged.offsets);
        Self {
            runtime: open(),
            graph,
            lengths,
            counted: mask,
            left,
            weights,
            weight,
            out,
            offsets: ragged.offsets,
            left_gradient,
            weight_gradient,
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

    fn step(&self, program: &Program, lengths: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let left = data(self.bound * self.depth, 7);
        let weights = data(self.planes * self.depth * self.columns, 13);
        let weight = data(self.bound * self.columns, 29);
        match self.counted {
            Some(mask) => {
                let bound = self.graph.shape(self.left).dims()[2];
                let mut flags = vec![0.0f32; self.planes as usize * bound as usize];
                for (plane, length) in lengths.iter().enumerate() {
                    flags[plane * bound as usize..][..*length as usize].fill(1.0);
                }
                self.runtime.write(program, mask, &flags);
            }
            None => self.runtime.write(program, self.lengths, lengths),
        }
        self.runtime.write(program, self.left, &left);
        self.runtime.write(program, self.weights, &weights);
        self.runtime.write(program, self.weight, &weight);
        self.runtime.run(program);
        let produced = self.runtime.read_many(
            program,
            &[self.out, self.left_gradient, self.weight_gradient],
        );
        let (out, left_gradient, weight_gradient) = reference(Packed {
            lengths,
            left: &left,
            weights: &weights,
            weight: &weight,
            depth: self.depth,
            columns: self.columns,
        });
        assert_close(&produced[0], &out, 1e-4);
        assert_close(&produced[1], &left_gradient, 1e-4);
        assert_close(&produced[2], &weight_gradient, 1e-4);
        let mut running = 0.0f32;
        let mut offsets = vec![0.0f32; lengths.len() + 1];
        for (plane, length) in lengths.iter().enumerate() {
            running += length;
            offsets[plane + 1] = running;
        }
        assert_close(
            &self.runtime.read(program, self.offsets)[..lengths.len() + 1],
            &offsets,
            1e-5,
        );
        (
            produced[0].clone(),
            produced[1].clone(),
            produced[2].clone(),
        )
    }
}

#[test]
fn a_segmented_product_trains_the_rows_it_packs_and_the_weights_it_weighs() {
    for lengths in [
        [3.0f32, 0.0, 5.0, 2.0].as_slice(),
        [4.0, 4.0, 4.0, 4.0].as_slice(),
        [0.0, 0.0, 0.0, 7.0].as_slice(),
        [1.0, 0.0, 0.0, 0.0].as_slice(),
        [9.0, 1.0, 5.0, 1.0].as_slice(),
    ] {
        let segmented = Segmented::of(4, 16, 8, 6);
        let program = segmented.compile();
        segmented.step(&program, lengths);
    }
}

#[test]
fn a_segmented_product_of_no_rows_trains_nothing() {
    let segmented = Segmented::of(3, 8, 4, 2);
    let program = segmented.compile();
    let (out, left_gradient, weight_gradient) = segmented.step(&program, &[0.0, 0.0, 0.0]);
    assert!(out.is_empty(), "{out:?} holds no row");
    assert!(left_gradient.is_empty(), "{left_gradient:?} holds no row");
    assert!(
        weight_gradient.iter().all(|value| *value == 0.0),
        "an expert no token reaches learns nothing",
    );
}

#[test]
fn a_segment_longer_than_a_workgroup_trains_every_row_of_it() {
    let segmented = Segmented::of(2, 512, 8, 4);
    let program = segmented.compile();
    let (out, left_gradient, _) = segmented.step(&program, &[300.0, 70.0]);
    assert_eq!(out.len(), 370 * 4);
    assert_eq!(left_gradient.len(), 370 * 8);
}

#[test]
fn a_weight_tile_wider_than_a_segment_weighs_every_row_of_every_tile() {
    let segmented = Segmented::of(2, 64, 128, 96);
    let program = segmented.compile();
    let (out, left_gradient, weight_gradient) = segmented.step(&program, &[40.0, 20.0]);
    assert_eq!(out.len(), 60 * 96);
    assert_eq!(left_gradient.len(), 60 * 128);
    assert_eq!(weight_gradient.len(), 2 * 128 * 96);
}

#[test]
fn a_device_count_of_the_rows_weighs_every_segment_it_closes() {
    let segmented = Segmented::counted(4, 32, 16, 8);
    let program = segmented.compile();
    for lengths in [
        [14.0f32, 0.0, 9.0, 8.0].as_slice(),
        [1.0, 1.0, 1.0, 29.0].as_slice(),
        [0.0, 0.0, 0.0, 0.0].as_slice(),
    ] {
        let (_, _, weight_gradient) = segmented.step(&program, lengths);
        assert_eq!(weight_gradient.len(), 4 * 16 * 8);
    }
}

struct Mixture {
    runtime: Runtime,
    graph: Graph<'static>,
    lengths: Value<'static>,
    batched: Value<'static>,
    weights: Value<'static>,
    weight: Value<'static>,
    out: Value<'static>,
    offsets: Value<'static>,
    batched_gradient: Value<'static>,
    weight_gradient: Value<'static>,
    depth: u32,
    columns: u32,
    capacity: u32,
    bound: u32,
    planes: u32,
}

impl Mixture {
    fn of(planes: u32, capacity: u32, bound: u32, depth: u32, columns: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(planes), Element::Single);
        let ragged = graph.ragged(bound, lengths);
        let mapped = graph.rows(ragged);
        let rows = Shape::of([1, 1, bound, 1]).freed(&[(2, ragged.extent)]);
        let batched =
            graph.gradient_input(Shape::of([planes, 1, capacity, depth]), Element::Single);
        let source = graph.add(
            graph.mul(mapped.plane, graph.fill(rows, capacity as f32)),
            mapped.position,
        );
        let packed = graph.gather(batched, source);
        let weights = graph.gradient_input(Shape::of([planes, 1, depth, columns]), Element::Single);
        let out = graph.grouped_matmul(packed, weights, ragged.offsets);
        let weight = graph.gradient_input(
            Shape::of([1, 1, bound, columns]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let loss = graph.sum(graph.mul(out, weight));
        let collected = graph.backward(loss);
        let batched_gradient = collected.of(batched);
        let weight_gradient = collected.of(weights);
        graph.retain(out);
        graph.retain(batched_gradient);
        graph.retain(weight_gradient);
        graph.retain(ragged.offsets);
        Self {
            runtime: open(),
            graph,
            lengths,
            batched,
            weights,
            weight,
            out,
            offsets: ragged.offsets,
            batched_gradient,
            weight_gradient,
            depth,
            columns,
            capacity,
            bound,
            planes,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(&self, program: &Program, lengths: &[f32]) {
        let (planes, capacity) = (self.planes, self.capacity);
        let batched = data(planes * capacity * self.depth, 3);
        let weights = data(planes * self.depth * self.columns, 13);
        let weight = data(self.bound * self.columns, 29);
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.batched, &batched);
        self.runtime.write(program, self.weights, &weights);
        self.runtime.write(program, self.weight, &weight);
        self.runtime.run(program);
        let produced = self.runtime.read_many(
            program,
            &[self.out, self.weight_gradient, self.batched_gradient],
        );
        let mut packed = vec![0.0f32; (self.bound * self.depth) as usize];
        let mut row = 0u32;
        for (plane, length) in lengths.iter().enumerate() {
            for position in 0..*length as u32 {
                let from = ((plane as u32 * capacity + position) * self.depth) as usize;
                let to = (row * self.depth) as usize;
                packed[to..to + self.depth as usize]
                    .copy_from_slice(&batched[from..from + self.depth as usize]);
                row += 1;
            }
        }
        let (out, packed_gradient, weight_gradient) = reference(Packed {
            lengths,
            left: &packed,
            weights: &weights,
            weight: &weight,
            depth: self.depth,
            columns: self.columns,
        });
        assert_close(&produced[0], &out, 1e-4);
        assert_close(&produced[1], &weight_gradient, 1e-4);
        let mut batched_gradient = vec![0.0f32; (planes * capacity * self.depth) as usize];
        let mut row = 0u32;
        for (plane, length) in lengths.iter().enumerate() {
            for position in 0..*length as u32 {
                let from = (row * self.depth) as usize;
                let to = ((plane as u32 * capacity + position) * self.depth) as usize;
                for step in 0..self.depth as usize {
                    batched_gradient[to + step] += packed_gradient[from + step];
                }
                row += 1;
            }
        }
        assert_close(&produced[2], &batched_gradient, 1e-4);
        let mut running = 0.0f32;
        let mut offsets = vec![0.0f32; lengths.len() + 1];
        for (plane, length) in lengths.iter().enumerate() {
            running += length;
            offsets[plane + 1] = running;
        }
        assert_close(
            &self.runtime.read(program, self.offsets)[..lengths.len() + 1],
            &offsets,
            1e-5,
        );
    }
}

#[test]
fn a_packed_mixture_trains_the_tokens_it_packs_and_the_experts_it_weighs() {
    let mixture = Mixture::of(3, 6, 12, 4, 2);
    let program = mixture.compile();
    for lengths in [
        [2.0f32, 0.0, 5.0].as_slice(),
        [6.0, 1.0, 3.0].as_slice(),
        [0.0, 11.0, 0.0].as_slice(),
        [0.0, 0.0, 0.0].as_slice(),
    ] {
        mixture.step(&program, lengths);
    }
}
