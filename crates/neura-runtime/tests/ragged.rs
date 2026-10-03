use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const WIDTH: u32 = 4;
const PLANES: u32 = 4;
const BOUND: u32 = 16;
const SCALE: f32 = 0.5;

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

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

fn packed(lengths: &[f32], cache: &[f32], width: u32) -> Vec<f32> {
    let mut packed = Vec::new();
    let offsets = offsets(lengths);
    for plane in 0..lengths.len() {
        let start = offsets[plane] as usize * width as usize;
        let end = offsets[plane + 1] as usize * width as usize;
        packed.extend_from_slice(&cache[start..end]);
    }
    packed
}

fn expected(
    lengths: &[f32],
    cache: &[f32],
    query: &[f32],
    cursors: &[f32],
    width: u32,
) -> Vec<f32> {
    let offsets = offsets(lengths);
    let planes = lengths.len();
    let mut out = vec![0.0f32; planes * width as usize];
    for plane in 0..planes {
        let keys = lengths[plane] as usize;
        let start = offsets[plane] as usize;
        let row = &query[plane * width as usize..][..width as usize];
        let mut scores = Vec::new();
        let mut attended = Vec::new();
        for at in 0..keys {
            if at as f32 > cursors[plane] {
                break;
            }
            let key = &cache[(start + at) * width as usize..][..width as usize];
            scores.push(
                row.iter()
                    .zip(key)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    * SCALE,
            );
            attended.push(start + at);
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
        for depth in 0..width as usize {
            let mut value = 0.0f32;
            for (at, weight) in attended.iter().zip(&weights) {
                value += weight * cache[at * width as usize + depth];
            }
            out[plane * width as usize + depth] = value / total;
        }
    }
    out
}

struct Ragged {
    runtime: Runtime,
    graph: Graph<'static>,
    lengths: Value<'static>,
    cursor: Value<'static>,
    query: Value<'static>,
    cache: Value<'static>,
    out: Value<'static>,
    offsets: Value<'static>,
}

impl Ragged {
    fn of(planes: u32, bound: u32, width: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(planes), Element::Single);
        let ragged = graph.ragged(bound, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, bound, width]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let query = graph.input(Shape::of([1, planes, 1, width]), Element::Single);
        let cursor = graph.input(Shape::of([1, planes, 1, 1]), Element::Single);
        let out = graph.attention(
            query,
            cache,
            cache,
            AttentionOptions {
                scale: SCALE,
                causal: true,
                origin: Some(cursor),
                segments: Some(ragged.offsets),
            },
        );
        graph.retain(out);
        graph.retain(ragged.offsets);
        Self {
            runtime: open(),
            graph,
            lengths,
            cursor,
            query,
            cache,
            out,
            offsets: ragged.offsets,
        }
    }

    fn compile(&self) -> Program<'_> {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(&self, program: &Program<'_>, lengths: &[f32], width: u32) -> (Vec<f32>, Vec<f32>) {
        let bound = self.graph.shape(self.cache).dims()[2];
        let cache = data(bound * width, 17);
        let query = data(lengths.len() as u32 * width, 29);
        let cursors = lengths
            .iter()
            .map(|length| (length - 1.0).max(0.0))
            .collect::<Vec<f32>>();
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.cache, &cache);
        self.runtime.write(program, self.query, &query);
        self.runtime.write(program, self.cursor, &cursors);
        self.runtime.run(program);
        let produced = self.runtime.read(program, self.out);
        let scanned = self.runtime.read(program, self.offsets);
        let packed = packed(lengths, &cache, width);
        let expected = expected(lengths, &packed, &query, &cursors, width);
        assert_close(&produced, &expected, 1e-5);
        (produced, scanned)
    }
}

#[test]
fn a_device_prefix_places_the_planes_of_a_packed_cache() {
    let ragged = Ragged::of(PLANES, BOUND, WIDTH);
    let program = ragged.compile();
    let (_, scanned) = ragged.step(&program, &[3.0, 0.0, 5.0, 2.0], WIDTH);
    assert_eq!(scanned, [0.0, 3.0, 3.0, 8.0, 10.0]);
}

#[test]
fn a_prefix_walks_every_chunk_of_a_long_plane_axis() {
    let planes = 3000;
    let bound = 4608;
    let widths = [1, 2];
    for width in widths {
        let ragged = Ragged::of(planes, bound, width);
        let program = ragged.compile();
        let lengths = (0..planes)
            .map(|plane| (plane % 4) as f32)
            .collect::<Vec<f32>>();
        let (_, scanned) = ragged.step(&program, &lengths, width);
        assert_eq!(
            scanned,
            offsets(&lengths)
                .into_iter()
                .map(|offset| offset as f32)
                .collect::<Vec<f32>>(),
        );
    }
}

#[test]
fn device_counts_rule_the_key_spans() {
    let graph: Graph<'static> = Graph::new();
    let mask = graph.input(Shape::of([PLANES, 1, BOUND, 1]), Element::Single);
    let counts = graph.sum_axis(mask, 2);
    let ragged = graph.ragged(BOUND, counts);
    let cache = graph.resident(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(Shape::of([1, PLANES, 1, WIDTH]), Element::Single);
    let cursor = graph.input(Shape::of([1, PLANES, 1, 1]), Element::Single);
    let out = graph.attention(
        query,
        cache,
        cache,
        AttentionOptions {
            scale: SCALE,
            causal: true,
            origin: Some(cursor),
            segments: Some(ragged.offsets),
        },
    );
    graph.retain(out);
    graph.retain(ragged.offsets);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let mut mask_data = vec![0.0f32; (PLANES * BOUND) as usize];
    let mut lengths = Vec::new();
    for plane in 0..PLANES {
        let length = (plane * 3) % 5;
        lengths.push(length as f32);
        for at in 0..length {
            mask_data[(plane * BOUND + at) as usize] = 1.0;
        }
    }
    let cache_data = data(BOUND * WIDTH, 17);
    let query_data = data(PLANES * WIDTH, 29);
    let cursors = lengths
        .iter()
        .map(|length| (length - 1.0).max(0.0))
        .collect::<Vec<f32>>();
    runtime.write(&program, mask, &mask_data);
    runtime.write(&program, cache, &cache_data);
    runtime.write(&program, query, &query_data);
    runtime.write(&program, cursor, &cursors);
    runtime.run(&program);
    let produced = runtime.read(&program, out);
    let scanned = runtime.read(&program, ragged.offsets);
    assert_eq!(
        scanned,
        offsets(&lengths)
            .into_iter()
            .map(|offset| offset as f32)
            .collect::<Vec<f32>>(),
    );
    let packed = packed(&lengths, &cache_data, WIDTH);
    assert_close(
        &produced,
        &expected(&lengths, &packed, &query_data, &cursors, WIDTH),
        1e-5,
    );
}

#[test]
fn an_extent_beyond_the_bound_a_graph_declares_is_refused() {
    let ragged = Ragged::of(PLANES, BOUND, WIDTH);
    let program = ragged.compile();
    let lengths = [3.0f32, 0.0, 5.0, 20.0];
    let bound = ragged.graph.shape(ragged.cache).dims()[2];
    ragged.runtime.write(&program, ragged.lengths, &lengths);
    ragged
        .runtime
        .write(&program, ragged.cache, &data(bound * WIDTH, 17));
    ragged
        .runtime
        .write(&program, ragged.query, &data(PLANES * WIDTH, 29));
    ragged
        .runtime
        .write(&program, ragged.cursor, &[2.0, 0.0, 4.0, 19.0]);
    ragged.runtime.run(&program);
    assert!(refuses(|| {
        ragged.runtime.read(&program, ragged.out);
    }));
}

#[test]
fn a_segmented_attention_reaches_no_gradient() {
    assert!(refuses(|| {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.parameter(
            Shape::vector(PLANES),
            neura_graph::Init::Zero,
            Element::Single,
        );
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let query = graph.parameter(
            Shape::of([1, PLANES, 1, WIDTH]),
            neura_graph::Init::Zero,
            Element::Single,
        );
        graph.attention(
            query,
            cache,
            cache,
            AttentionOptions {
                scale: SCALE,
                causal: false,
                origin: None,
                segments: Some(ragged.offsets),
            },
        );
    }));
}

#[test]
fn a_segmented_attention_names_an_offset_per_plane() {
    assert!(refuses(|| {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(PLANES), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let query = graph.input(Shape::of([1, PLANES + 1, 1, WIDTH]), Element::Single);
        graph.attention(
            query,
            cache,
            cache,
            AttentionOptions {
                scale: SCALE,
                causal: false,
                origin: None,
                segments: Some(ragged.offsets),
            },
        );
    }));
}

#[test]
fn a_causal_segmented_attention_walks_a_cursor() {
    assert!(refuses(|| {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(PLANES), Element::Single);
        let ragged = graph.ragged(BOUND, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let query = graph.input(Shape::of([1, PLANES, 1, WIDTH]), Element::Single);
        graph.attention(
            query,
            cache,
            cache,
            AttentionOptions {
                scale: SCALE,
                causal: true,
                origin: None,
                segments: Some(ragged.offsets),
            },
        );
    }));
}

#[test]
fn a_ragged_extent_outruns_the_numbers_a_device_sums_exactly() {
    assert!(refuses(|| {
        let graph: Graph<'static> = Graph::new();
        let lengths = graph.input(Shape::vector(PLANES), Element::Single);
        graph.ragged((1 << 24) + 1, lengths);
    }));
    let graph: Graph<'static> = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    graph.ragged(1 << 24, lengths);
}

#[test]
fn a_ragged_axis_refuses_a_length_the_host_holds_as_a_view() {
    assert!(refuses(|| {
        let graph: Graph<'static> = Graph::new();
        let observations = graph.input(Shape::matrix(PLANES, PLANES), Element::Single);
        let lengths = graph.permute(observations, [0, 1, 3, 2]);
        graph.ragged(BOUND, lengths);
    }));
}

#[test]
fn a_segmented_attention_walks_an_axis_a_device_authors() {
    assert!(refuses(|| {
        let graph: Graph<'static> = Graph::new();
        let offsets = graph.input(Shape::vector(PLANES + 1), Element::Single);
        let cache = graph.input(
            Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, graph.free(BOUND))]),
            Element::Single,
        );
        let query = graph.input(Shape::of([1, PLANES, 1, WIDTH]), Element::Single);
        graph.attention(
            query,
            cache,
            cache,
            AttentionOptions {
                scale: SCALE,
                causal: false,
                origin: None,
                segments: Some(offsets),
            },
        );
    }));
}
