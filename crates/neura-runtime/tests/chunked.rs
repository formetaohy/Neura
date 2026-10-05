use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Shape, Value};
use neura_runtime::{Backends, Program, Runtime};

#[path = "support/backend.rs"]
mod backend;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const WIDTH: u32 = 4;
const PLANES: u32 = 4;
const KEYS: u32 = 16;
const QUERIES: u32 = 12;
const SCALE: f32 = 0.5;

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

fn offsets(lengths: &[f32]) -> Vec<u32> {
    let mut offsets = Vec::with_capacity(lengths.len() + 1);
    let mut running = 0u32;
    offsets.push(0);
    for length in lengths {
        running += *length as u32;
        offsets.push(running);
    }
    offsets
}

struct Batch<'a> {
    keys: &'a [f32],
    values: &'a [f32],
    queries: &'a [f32],
    weight: &'a [f32],
}

struct Reference {
    out: Vec<f32>,
    query_grad: Vec<f32>,
    key_grad: Vec<f32>,
    value_grad: Vec<f32>,
}

fn reference(
    query_lengths: &[f32],
    key_lengths: &[f32],
    cursors: &[f32],
    batch: Batch<'_>,
    reach: u32,
    causal: bool,
    width: u32,
) -> Reference {
    let Batch {
        keys,
        values,
        queries,
        weight,
    } = batch;
    let query_offsets = offsets(query_lengths);
    let key_offsets = offsets(key_lengths);
    let query_live = query_offsets.last().copied().unwrap_or(0) as usize;
    let key_live = key_offsets.last().copied().unwrap_or(0) as usize;
    let width = width as usize;
    let mut out = vec![0.0f32; query_live * width];
    let mut query_grad = vec![0.0f32; query_live * width];
    let mut key_grad = vec![0.0f32; key_live * width];
    let mut value_grad = vec![0.0f32; key_live * width];
    for (plane, key_length) in key_lengths.iter().enumerate() {
        let keys_in_plane = *key_length as usize;
        let queries_in_plane = query_lengths[plane] as usize;
        let key_start = key_offsets[plane] as usize;
        let query_start = query_offsets[plane] as usize;
        let ringed = !cursors.is_empty();
        let base = if ringed {
            cursors[plane] as usize
        } else {
            keys_in_plane - queries_in_plane
        };
        let written = base + queries_in_plane;
        for row in 0..queries_in_plane {
            let at = (query_start + row) * width;
            let query = &queries[at..at + width];
            let dout = &weight[at..at + width];
            let position = base + row;
            let mut attended = Vec::new();
            let mut scores = Vec::new();
            for key in 0..keys_in_plane {
                let slot = if ringed {
                    ring_position(key, keys_in_plane, written)
                } else {
                    key
                };
                if causal {
                    if slot > position {
                        continue;
                    }
                    if reach > 0 && (position - slot) as u32 >= reach {
                        continue;
                    }
                }
                let packed = (key_start + key) * width;
                scores.push(
                    query
                        .iter()
                        .zip(&keys[packed..packed + width])
                        .map(|(left, right)| left * right)
                        .sum::<f32>()
                        * SCALE,
                );
                attended.push(packed);
            }
            if attended.is_empty() {
                continue;
            }
            let peak = scores.iter().copied().fold(f32::MIN, f32::max);
            let weights = scores
                .iter()
                .map(|score| (score - peak).exp())
                .collect::<Vec<f32>>();
            let total = weights.iter().sum::<f32>();
            let probabilities = weights
                .iter()
                .map(|weight| weight / total)
                .collect::<Vec<f32>>();
            for depth in 0..width {
                out[at + depth] = attended
                    .iter()
                    .zip(&probabilities)
                    .map(|(packed, probability)| probability * values[packed + depth])
                    .sum::<f32>();
            }
            let row_dot = (0..width)
                .map(|depth| dout[depth] * out[at + depth])
                .sum::<f32>();
            for (packed, probability) in attended.iter().zip(&probabilities) {
                let weighted = (0..width)
                    .map(|depth| dout[depth] * values[packed + depth])
                    .sum::<f32>();
                let scored = probability * (weighted - row_dot) * SCALE;
                for depth in 0..width {
                    value_grad[packed + depth] += probability * dout[depth];
                    key_grad[packed + depth] += scored * query[depth];
                    query_grad[at + depth] += scored * keys[packed + depth];
                }
            }
        }
    }
    Reference {
        out,
        query_grad,
        key_grad,
        value_grad,
    }
}

fn ring_position(slot: usize, capacity: usize, written: usize) -> usize {
    if written <= capacity {
        return slot;
    }
    slot + capacity * ((written - 1 - slot) / capacity)
}

#[derive(Clone, Copy)]
struct Case {
    planes: u32,
    queries: u32,
    keys: u32,
    width: u32,
    reach: u32,
    causal: bool,
    binding: bool,
    ringed: bool,
}

impl Case {
    fn of(planes: u32, queries: u32, keys: u32, width: u32) -> Self {
        Self {
            planes,
            queries,
            keys,
            width,
            reach: 0,
            causal: true,
            binding: false,
            ringed: false,
        }
    }

    fn windowed(mut self, reach: u32) -> Self {
        self.reach = reach;
        self
    }

    fn unmasked(mut self) -> Self {
        self.causal = false;
        self
    }

    fn bound(mut self) -> Self {
        self.binding = true;
        self
    }

    fn ringed(mut self) -> Self {
        self.ringed = true;
        self
    }
}

struct Chunked {
    runtime: Runtime,
    graph: Graph<'static>,
    query_lengths: Value<'static>,
    key_lengths: Value<'static>,
    keys: Value<'static>,
    values: Value<'static>,
    queries: Value<'static>,
    weight: Value<'static>,
    cursors: Option<Value<'static>>,
    query_offsets: Value<'static>,
    key_offsets: Value<'static>,
    out: Value<'static>,
    gradients: [Value<'static>; 3],
    case: Case,
}

impl Chunked {
    fn over(backends: Backends, case: Case) -> Self {
        Self::build(backend::open_with(backends), case)
    }

    fn build(runtime: Runtime, case: Case) -> Self {
        let Case {
            planes,
            queries,
            keys,
            width,
            reach,
            causal,
            binding,
            ringed,
        } = case;
        let graph: Graph<'static> = Graph::new();
        let live = binding.then(|| graph.free(planes));
        let lengths = |bound: u32| {
            let shape = match live {
                Some(live) => Shape::of([1, bound]).freed(&[(3, live)]),
                None => Shape::vector(bound),
            };
            graph.input(shape, Element::Single)
        };
        let query_lengths = lengths(planes);
        let key_lengths = lengths(planes);
        let cursors = ringed.then(|| lengths(planes));
        let query_axis = graph.ragged(queries, query_lengths);
        let key_axis = graph.ragged(keys, key_lengths);
        let packed_queries = Shape::of([1, 1, queries, width]).freed(&[(2, query_axis.extent)]);
        let packed_keys = Shape::of([1, 1, keys, width]).freed(&[(2, key_axis.extent)]);
        let query = graph.gradient_input(packed_queries, Element::Single);
        let key = graph.gradient_input(packed_keys, Element::Single);
        let value = graph.gradient_input(packed_keys, Element::Single);
        let out = graph.attention(
            query,
            key,
            value,
            AttentionOptions {
                scale: SCALE,
                causal,
                origin: cursors,
                segments: Some(key_axis.offsets),
                reach: (reach > 0).then_some(reach),
                query_segments: Some(query_axis.offsets),
            },
        );
        let weight = graph.input(packed_queries, Element::Single);
        let loss = graph.sum(graph.mul(out, weight));
        let collected = graph.backward(loss);
        let gradients = [collected.of(query), collected.of(key), collected.of(value)];
        graph.retain(out);
        for gradient in gradients {
            graph.retain(gradient);
        }
        graph.retain(query_axis.offsets);
        graph.retain(key_axis.offsets);
        Self {
            runtime,
            graph,
            query_lengths,
            key_lengths,
            keys: key,
            values: value,
            queries: query,
            weight,
            cursors,
            query_offsets: query_axis.offsets,
            key_offsets: key_axis.offsets,
            out,
            gradients,
            case,
        }
    }

    fn compile(&self) -> Program<'_> {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn compile_narrow(&self) -> Program<'_> {
        let runtime = &self.runtime;
        let narrow = runtime
            .profiles()
            .into_iter()
            .min_by_key(|profile| profile.workgroup())
            .expect("a device offers a workgroup a program compiles for");
        let weights = runtime.weights(&self.graph);
        runtime.compile_chosen(&self.graph, &weights, narrow, &[])
    }

    fn bind(&self, program: &Program<'_>, planes: u32) {
        assert!(self.case.binding, "only a bound plane axis binds a length");
        self.runtime.bind(program, &[planes]);
    }

    fn step(
        &self,
        program: &Program<'_>,
        query_lengths: &[f32],
        key_lengths: &[f32],
    ) -> Vec<Vec<f32>> {
        self.walk(program, query_lengths, key_lengths, &[])
    }

    fn step_ring(
        &self,
        program: &Program<'_>,
        query_lengths: &[f32],
        key_lengths: &[f32],
        cursors: &[f32],
    ) -> Vec<Vec<f32>> {
        assert!(
            self.case.ringed,
            "only a ring weighs the slots its window wrapped",
        );
        self.walk(program, query_lengths, key_lengths, cursors)
    }

    fn walk(
        &self,
        program: &Program<'_>,
        query_lengths: &[f32],
        key_lengths: &[f32],
        cursors: &[f32],
    ) -> Vec<Vec<f32>> {
        let shape = |value| self.graph.shape(value).dims();
        let query_bound = shape(self.queries)[2];
        let key_bound = shape(self.keys)[2];
        let width = shape(self.keys)[3];
        let keys = data(key_bound * width, 43);
        let values = data(key_bound * width, 71);
        let queries = data(query_bound * width, 17);
        let weight = data(query_bound * width, 89);
        self.runtime
            .write(program, self.query_lengths, query_lengths);
        self.runtime.write(program, self.key_lengths, key_lengths);
        if let Some(cursor) = self.cursors {
            self.runtime.write(program, cursor, cursors);
        }
        self.runtime.write(program, self.keys, &keys);
        self.runtime.write(program, self.values, &values);
        self.runtime.write(program, self.queries, &queries);
        self.runtime.write(program, self.weight, &weight);
        self.runtime.run(program);
        let produced = self.runtime.read_many(
            program,
            &[
                self.out,
                self.gradients[0],
                self.gradients[1],
                self.gradients[2],
            ],
        );
        let scanned = self
            .runtime
            .read_many(program, &[self.query_offsets, self.key_offsets]);
        for (scanned, lengths) in [(&scanned[0], query_lengths), (&scanned[1], key_lengths)] {
            assert_eq!(
                scanned[..lengths.len() + 1],
                offsets(lengths)
                    .into_iter()
                    .map(|offset| offset as f32)
                    .collect::<Vec<f32>>(),
            );
        }
        let expected = reference(
            query_lengths,
            key_lengths,
            cursors,
            Batch {
                keys: &keys,
                values: &values,
                queries: &queries,
                weight: &weight,
            },
            self.case.reach,
            self.case.causal,
            width,
        );
        assert_close(&produced[0], &expected.out, 1e-5);
        assert_close(&produced[1], &expected.query_grad, 1e-5);
        assert_close(&produced[2], &expected.key_grad, 1e-5);
        assert_close(&produced[3], &expected.value_grad, 1e-5);
        produced
    }
}

#[test]
fn a_chunked_prefill_weighs_the_rows_each_plane_packs_against_its_cached_keys() {
    let chunked = Chunked::build(open(), Case::of(PLANES, QUERIES, KEYS, WIDTH));
    let program = chunked.compile();
    for (queries, keys) in [
        ([3.0f32, 2.0, 1.0, 0.0], [7.0f32, 5.0, 3.0, 1.0]),
        ([4.0, 4.0, 4.0, 0.0], [4.0, 4.0, 4.0, 0.0]),
        ([0.0, 0.0, 0.0, 0.0], [6.0, 5.0, 3.0, 2.0]),
        ([2.0, 0.0, 0.0, 0.0], [2.0, 4.0, 4.0, 4.0]),
        ([1.0, 2.0, 3.0, 1.0], [6.0, 5.0, 4.0, 1.0]),
    ] {
        chunked.step(&program, &queries, &keys);
    }
}

#[test]
fn a_chunked_prefill_without_a_mask_weighs_every_cached_key_of_its_plane() {
    let chunked = Chunked::build(open(), Case::of(PLANES, QUERIES, KEYS, WIDTH).unmasked());
    let program = chunked.compile();
    chunked.step(&program, &[3.0, 2.0, 1.0, 0.0], &[7.0, 5.0, 3.0, 1.0]);
    chunked.step(&program, &[2.0, 4.0, 4.0, 0.0], &[4.0, 6.0, 4.0, 2.0]);
}

#[test]
fn a_windowed_chunked_prefill_reaches_back_from_the_position_its_rows_pack() {
    for reach in [1, 2, 3, 5] {
        let case = Case::of(PLANES, QUERIES, KEYS, WIDTH).windowed(reach);
        let chunked = Chunked::build(open(), case);
        let program = chunked.compile();
        chunked.step(&program, &[3.0, 2.0, 1.0, 0.0], &[7.0, 5.0, 3.0, 1.0]);
        chunked.step(&program, &[2.0, 4.0, 4.0, 1.0], &[6.0, 4.0, 5.0, 1.0]);
    }
}

#[test]
fn a_chunked_prefill_walks_only_the_planes_a_binding_holds() {
    let chunked = Chunked::build(open(), Case::of(PLANES, QUERIES, KEYS, WIDTH).bound());
    let program = chunked.compile();
    let queries = [3.0f32, 2.0, 1.0, 1.0];
    let keys = [7.0f32, 5.0, 3.0, 1.0];
    chunked.bind(&program, PLANES);
    let wide = chunked.step(&program, &queries, &keys);
    for planes in [PLANES - 1, 1] {
        chunked.bind(&program, planes);
        let narrow = chunked.step(
            &program,
            &queries[..planes as usize],
            &keys[..planes as usize],
        );
        for (narrow, wide) in narrow.iter().zip(&wide) {
            assert_eq!(
                narrow,
                &wide[..narrow.len()],
                "the planes a binding holds walk the rows their own offsets close",
            );
        }
    }
}

#[test]
fn a_chunked_prefill_of_a_plane_longer_than_a_workgroup_trains_every_row() {
    let chunked = Chunked::build(open(), Case::of(PLANES, 512, 512, WIDTH));
    let program = chunked.compile_narrow();
    chunked.step(&program, &[130.0, 70.0, 0.0, 5.0], &[300.0, 70.0, 1.0, 5.0]);
}

#[test]
fn every_platform_backend_trains_a_chunked_prefill() {
    for backends in Backends::PLATFORM {
        let chunked = Chunked::over(backends, Case::of(PLANES, QUERIES, KEYS, WIDTH));
        let program = chunked.compile();
        chunked.step(&program, &[3.0, 2.0, 1.0, 0.0], &[7.0, 5.0, 3.0, 1.0]);
    }
}

#[test]
fn a_chunked_prefill_weighs_a_ring_by_the_count_its_slots_wrapped() {
    for backends in Backends::PLATFORM {
        let case = Case::of(PLANES, PLANES * QUERIES, PLANES * KEYS, WIDTH)
            .windowed(KEYS)
            .ringed();
        let chunked = Chunked::over(backends, case);
        let program = chunked.compile();
        for (queries, keys, cursors) in [
            (
                [12.0f32, 4.0, 5.0, 3.0],
                [16.0f32, 16.0, 16.0, 16.0],
                [8.0f32, 12.0, 4.0, 21.0],
            ),
            (
                [2.0, 2.0, 2.0, 2.0],
                [16.0, 9.0, 16.0, 4.0],
                [20.0, 7.0, 33.0, 2.0],
            ),
        ] {
            chunked.step_ring(&program, &queries, &keys, &cursors);
        }
    }
}

#[test]
fn a_windowed_ring_chunked_prefill_reaches_back_from_the_slots_it_wrapped() {
    for reach in [1, 2, 5, KEYS] {
        let case = Case::of(PLANES, PLANES * QUERIES, PLANES * KEYS, WIDTH)
            .windowed(reach)
            .ringed();
        let chunked = Chunked::build(open(), case);
        let program = chunked.compile();
        chunked.step_ring(
            &program,
            &[6.0, 6.0, 6.0, 6.0],
            &[16.0, 12.0, 16.0, 3.0],
            &[18.0, 10.0, 26.0, 2.0],
        );
    }
}
