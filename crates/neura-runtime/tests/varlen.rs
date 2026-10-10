use neura_abi::Element;
use neura_gpu::Backends;
use neura_graph::{AttentionOptions, Graph, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/backend.rs"]
mod backend;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const WIDTH: u32 = 4;
const PLANES: u32 = 4;
const BOUND: u32 = 16;
const TALL: u32 = 512;
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

fn reference(lengths: &[f32], batch: Batch<'_>, reach: u32, causal: bool, width: u32) -> Reference {
    let Batch {
        keys,
        values,
        queries,
        weight,
    } = batch;
    let offsets = offsets(lengths);
    let live = offsets.last().copied().unwrap_or(0) as usize;
    let width = width as usize;
    let mut out = vec![0.0f32; live * width];
    let mut query_grad = vec![0.0f32; live * width];
    let mut key_grad = vec![0.0f32; live * width];
    let mut value_grad = vec![0.0f32; live * width];
    for (plane, length) in lengths.iter().enumerate() {
        let count = *length as usize;
        let start = offsets[plane] as usize;
        for row in 0..count {
            let at = (start + row) * width;
            let query = &queries[at..at + width];
            let dout = &weight[at..at + width];
            let mut attended = Vec::new();
            let mut scores = Vec::new();
            let last = if causal { row } else { count - 1 };
            for key in 0..=last {
                if reach > 0 && (row - key) as u32 >= reach {
                    continue;
                }
                let packed = (start + key) * width;
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

#[derive(Clone, Copy)]
struct Case {
    planes: u32,
    bound: u32,
    width: u32,
    reach: u32,
    causal: bool,
    binding: bool,
    weighed: bool,
}

impl Case {
    fn of(planes: u32, bound: u32, width: u32) -> Self {
        Self {
            planes,
            bound,
            width,
            reach: 0,
            causal: true,
            binding: false,
            weighed: false,
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

    fn weighed(mut self) -> Self {
        self.weighed = true;
        self
    }
}

struct Packed {
    runtime: Runtime,
    graph: Graph<'static>,
    lengths: Value<'static>,
    keys: Value<'static>,
    values: Value<'static>,
    queries: Value<'static>,
    weight: Value<'static>,
    offsets: Value<'static>,
    out: Value<'static>,
    gradients: [Value<'static>; 3],
    factor: Option<Value<'static>>,
    case: Case,
}

impl Packed {
    fn over(backends: Backends, case: Case) -> Self {
        Self::build(backend::open_with(backends), case)
    }

    fn build(runtime: Runtime, case: Case) -> Self {
        let Case {
            planes,
            bound,
            width,
            reach,
            causal,
            binding,
            weighed,
        } = case;
        let graph: Graph<'static> = Graph::new();
        let live = binding.then(|| graph.free(planes));
        let lengths = graph.input(
            match live {
                Some(live) => Shape::of([1, planes]).freed(&[(3, live)]),
                None => Shape::vector(planes),
            },
            Element::Single,
        );
        let ragged = graph.ragged(bound, lengths);
        let packed = Shape::of([1, 1, bound, width]).freed(&[(2, ragged.extent)]);
        let queries = graph.gradient_input(packed, Element::Single);
        let keys = graph.gradient_input(packed, Element::Single);
        let values = graph.gradient_input(packed, Element::Single);
        let out = graph.attention(
            queries,
            keys,
            values,
            AttentionOptions {
                scale: Some(graph.knob(SCALE)),
                causal,
                origin: None,
                segments: Some(ragged.offsets),
                reach: (reach > 0).then_some(reach),
                query_segments: None,
            },
        );
        let weight = graph.input(packed, Element::Single);
        let loss = graph.sum(graph.mul(out, weight));
        let collected = graph.backward(loss);
        let factor = weighed.then(|| graph.input(packed, Element::Single));
        let query_grad = match factor {
            Some(factor) => graph.mul(collected.of(queries), factor),
            None => collected.of(queries),
        };
        let gradients = [query_grad, collected.of(keys), collected.of(values)];
        graph.retain(out);
        for gradient in gradients {
            graph.retain(gradient);
        }
        graph.retain(ragged.offsets);
        Self {
            runtime,
            graph,
            lengths,
            keys,
            values,
            queries,
            weight,
            offsets: ragged.offsets,
            out,
            gradients,
            factor,
            case,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn compile_narrow(&self) -> Program {
        let runtime = &self.runtime;
        let narrow = runtime
            .profiles()
            .into_iter()
            .min_by_key(|profile| profile.workgroup())
            .expect("a device offers a workgroup a program compiles for");
        let weights = runtime.weights(&self.graph);
        runtime.compile_chosen(&self.graph, &weights, narrow, &[])
    }

    fn bind(&self, program: &Program, planes: u32) {
        assert!(self.case.binding, "only a bound plane axis binds a length");
        self.runtime.bind(program, &[planes]);
    }

    fn step(&self, program: &Program, lengths: &[f32]) -> Vec<Vec<f32>> {
        let bound = self.graph.shape(self.keys).dims()[2];
        let width = self.graph.shape(self.keys).dims()[3];
        let keys = data(bound * width, 43);
        let values = data(bound * width, 71);
        let queries = data(bound * width, 17);
        let weight = data(bound * width, 89);
        let factor = data(bound * width, 113);
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.keys, &keys);
        self.runtime.write(program, self.values, &values);
        self.runtime.write(program, self.queries, &queries);
        self.runtime.write(program, self.weight, &weight);
        if let Some(value) = self.factor {
            self.runtime.write(program, value, &factor);
        }
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
        let scanned = self.runtime.read(program, self.offsets);
        assert_eq!(
            scanned[..lengths.len() + 1],
            offsets(lengths)
                .into_iter()
                .map(|offset| offset as f32)
                .collect::<Vec<f32>>(),
        );
        let expected = reference(
            lengths,
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
        let weighed = expected
            .query_grad
            .iter()
            .zip(&factor)
            .map(|(gradient, factor)| gradient * factor)
            .collect::<Vec<f32>>();
        let query_grad = if self.factor.is_some() {
            &weighed
        } else {
            &expected.query_grad
        };
        assert_close(&produced[0], &expected.out, 1e-5);
        assert_close(&produced[1], query_grad, 1e-5);
        assert_close(&produced[2], &expected.key_grad, 1e-5);
        assert_close(&produced[3], &expected.value_grad, 1e-5);
        produced
    }
}

#[test]
fn a_packed_query_walks_the_offsets_its_keys_close() {
    let packed = Packed::over(Backends::PLATFORM, Case::of(PLANES, BOUND, WIDTH));
    let program = packed.compile();
    for lengths in [
        [3.0f32, 0.0, 5.0, 2.0].as_slice(),
        [4.0, 4.0, 4.0, 4.0].as_slice(),
        [0.0, 0.0, 0.0, 0.0].as_slice(),
        [1.0, 0.0, 0.0, 0.0].as_slice(),
        [0.0, 0.0, 0.0, 7.0].as_slice(),
        [6.0, 5.0, 3.0, 2.0].as_slice(),
    ] {
        packed.step(&program, lengths);
    }
}

#[test]
fn a_packed_query_without_a_mask_weighs_every_key_of_its_plane() {
    let packed = Packed::build(open(), Case::of(PLANES, BOUND, WIDTH).unmasked());
    let program = packed.compile();
    packed.step(&program, &[3.0, 0.0, 5.0, 2.0]);
    packed.step(&program, &[6.0, 5.0, 3.0, 2.0]);
}

#[test]
fn a_windowed_packed_query_reaches_back_within_its_plane() {
    for reach in [1, 2, 3] {
        let case = Case::of(PLANES, BOUND, WIDTH).windowed(reach);
        let packed = Packed::build(open(), case);
        let program = packed.compile();
        packed.step(&program, &[3.0, 0.0, 5.0, 2.0]);
        packed.step(&program, &[6.0, 1.0, 2.0, 0.0]);
    }
}

#[test]
fn a_packed_query_gradient_reads_the_operand_it_folds_at_the_row_it_walks() {
    let packed = Packed::build(open(), Case::of(PLANES, BOUND, WIDTH).weighed());
    let program = packed.compile();
    packed.step(&program, &[3.0, 0.0, 5.0, 2.0]);
    packed.step(&program, &[6.0, 5.0, 3.0, 2.0]);
}

#[test]
fn a_packed_query_of_a_plane_longer_than_a_workgroup_trains_every_row() {
    let packed = Packed::build(open(), Case::of(PLANES, TALL, WIDTH));
    let program = packed.compile_narrow();
    packed.step(&program, &[300.0, 70.0, 0.0, 5.0]);
}

#[test]
fn a_packed_query_walks_only_the_planes_a_binding_holds() {
    let packed = Packed::build(open(), Case::of(PLANES, BOUND, WIDTH).bound());
    let program = packed.compile();
    let lengths = [3.0f32, 2.0, 5.0, 4.0];
    packed.bind(&program, PLANES);
    let wide = packed.step(&program, &lengths);
    for planes in [PLANES - 1, 1] {
        packed.bind(&program, planes);
        let narrow = packed.step(&program, &lengths[..planes as usize]);
        for (narrow, wide) in narrow.iter().zip(&wide) {
            assert_eq!(
                narrow,
                &wide[..narrow.len()],
                "the planes a binding holds walk the rows their own offsets close",
            );
        }
    }
}
