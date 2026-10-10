use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const WIDTH: u32 = 4;
const PLANES: u32 = 4;
const BOUND: u32 = 16;
const WIDE_BOUND: u32 = 1024;
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
    expected_with(lengths, cache, query, cursors, width, 0)
}

fn expected_with(
    lengths: &[f32],
    cache: &[f32],
    query: &[f32],
    cursors: &[f32],
    width: u32,
    reach: u32,
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
            if reach > 0 && cursors[plane] - at as f32 >= reach as f32 {
                continue;
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
    reach: u32,
}

impl Ragged {
    fn of(planes: u32, bound: u32, width: u32) -> Self {
        Self::windowed(planes, bound, width, 0)
    }

    fn windowed(planes: u32, bound: u32, width: u32, reach: u32) -> Self {
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
                scale: Some(graph.knob(SCALE)),
                causal: true,
                origin: Some(cursor),
                segments: Some(ragged.offsets),
                reach: (reach > 0).then_some(reach),
                query_segments: None,
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
            reach,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(&self, program: &Program, lengths: &[f32], width: u32) -> (Vec<f32>, Vec<f32>) {
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
        let expected = expected_with(lengths, &packed, &query, &cursors, width, self.reach);
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
            scale: Some(graph.knob(SCALE)),
            causal: true,
            origin: Some(cursor),
            segments: Some(ragged.offsets),
            reach: None,
            query_segments: None,
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
                scale: Some(graph.knob(SCALE)),
                causal: false,
                origin: None,
                segments: Some(ragged.offsets),
                reach: None,
                query_segments: None,
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
                scale: Some(graph.knob(SCALE)),
                causal: true,
                origin: None,
                segments: Some(ragged.offsets),
                reach: None,
                query_segments: None,
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
                scale: Some(graph.knob(SCALE)),
                causal: false,
                origin: None,
                segments: Some(offsets),
                reach: None,
                query_segments: None,
            },
        );
    }));
}

struct Planes {
    runtime: Runtime,
    graph: Graph<'static>,
    lengths: Value<'static>,
    cursor: Value<'static>,
    query: Value<'static>,
    cache: Value<'static>,
    out: Value<'static>,
    offsets: Value<'static>,
    total: Value<'static>,
}

impl Planes {
    fn of(planes: u32, bound: u32, width: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let live = graph.free(planes);
        let lengths = graph.input(Shape::of([1, planes]).freed(&[(3, live)]), Element::Single);
        let ragged = graph.ragged(bound, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, bound, width]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let query = graph.input(
            Shape::of([1, planes, 1, width]).freed(&[(1, live)]),
            Element::Single,
        );
        let cursor = graph.input(Shape::of([1, planes, 1, 1]), Element::Single);
        let out = graph.attention(
            query,
            cache,
            cache,
            AttentionOptions {
                scale: Some(graph.knob(SCALE)),
                causal: true,
                origin: Some(cursor),
                segments: Some(ragged.offsets),
                reach: None,
                query_segments: None,
            },
        );
        let total = graph.sum(cache);
        graph.retain(out);
        graph.retain(ragged.offsets);
        graph.retain(total);
        Self {
            runtime: open(),
            graph,
            lengths,
            cursor,
            query,
            cache,
            out,
            offsets: ragged.offsets,
            total,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn run(&self, program: &Program, lengths: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        let width = self.graph.shape(self.cache).dims()[3];
        let bound = self.graph.shape(self.cache).dims()[2];
        let planes = self.graph.shape(self.cursor).dims()[1];
        let cache = (0..bound * width)
            .map(|at| (at % 7) as f32 + 1.0)
            .collect::<Vec<f32>>();
        let query = data(lengths.len() as u32 * width, 29);
        let cursors = lengths
            .iter()
            .map(|length| (length - 1.0).max(0.0))
            .chain(std::iter::repeat_n(
                0.0,
                (planes - lengths.len() as u32) as usize,
            ))
            .collect::<Vec<f32>>();
        self.runtime.bind(program, &[lengths.len() as u32]);
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.cache, &cache);
        self.runtime.write(program, self.query, &query);
        self.runtime.write(program, self.cursor, &cursors);
        self.runtime.run(program);
        let produced = self.runtime.read(program, self.out);
        let scanned = self.runtime.read(program, self.offsets);
        let summed = self.runtime.read(program, self.total);
        let packed = packed(lengths, &cache, width);
        assert_close(
            &produced,
            &expected(lengths, &packed, &query, &cursors, width),
            1e-5,
        );
        let walked = self.runtime.read(program, self.cache).len();
        assert_eq!(
            walked,
            packed.len(),
            "the packed cache walks the tokens a binding holds",
        );
        assert_close(&summed, &[packed.iter().sum()], 1e-5);
        (produced, scanned, summed[0])
    }
}

#[test]
fn a_ragged_axis_walks_only_the_planes_a_binding_holds() {
    let planes = Planes::of(PLANES, WIDE_BOUND, WIDTH);
    let program = planes.compile();
    let (_, warm, _) = planes.run(&program, &[3.0, 2.0, 300.0, 300.0]);
    assert_eq!(warm, [0.0, 3.0, 5.0, 305.0, 605.0]);
    let (produced, scanned, _) = planes.run(&program, &[3.0, 2.0]);
    assert_eq!(produced.len(), 2 * WIDTH as usize);
    assert_eq!(
        scanned[..3],
        [0.0, 3.0, 5.0],
        "the prefix closes the offsets at the planes a binding holds, not at the bound its graph declares",
    );
}

#[test]
fn a_ragged_axis_of_no_planes_closes_its_offsets_at_nothing() {
    let planes = Planes::of(PLANES, WIDE_BOUND, WIDTH);
    let program = planes.compile();
    planes.run(&program, &[3.0, 2.0, 300.0, 300.0]);
    let (produced, scanned, summed) = planes.run(&program, &[]);
    assert!(produced.is_empty());
    assert_eq!(scanned[0], 0.0);
    assert_eq!(summed, 0.0);
}

#[test]
fn a_ragged_axis_of_one_plane_walks_the_plane_a_binding_holds() {
    let planes = Planes::of(1, WIDE_BOUND, WIDTH);
    let program = planes.compile();
    let (produced, scanned, _) = planes.run(&program, &[5.0]);
    assert_eq!(produced.len(), WIDTH as usize);
    assert_eq!(
        scanned[..2],
        [0.0, 5.0],
        "the prefix closes one offset for the one plane the binding holds",
    );
    let (produced, scanned, summed) = planes.run(&program, &[]);
    assert!(produced.is_empty());
    assert_eq!(scanned[0], 0.0);
    assert_eq!(summed, 0.0);
}

struct CountedPlanes {
    runtime: Runtime,
    graph: Graph<'static>,
    count: Value<'static>,
    lengths: Value<'static>,
    cursor: Value<'static>,
    query: Value<'static>,
    cache: Value<'static>,
    out: Value<'static>,
    offsets: Value<'static>,
    total: Value<'static>,
}

impl CountedPlanes {
    fn of(planes: u32, bound: u32, width: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let count = graph.input(Shape::scalar(), Element::Single);
        let live = graph.counted(planes, count);
        let lengths = graph.input(
            Shape::of([1, 1, 1, planes]).freed(&[(3, live)]),
            Element::Single,
        );
        let ragged = graph.ragged(bound, lengths);
        let cache = graph.resident(
            Shape::of([1, 1, bound, width]).freed(&[(2, ragged.extent)]),
            Element::Single,
        );
        let query = graph.input(
            Shape::of([1, planes, 1, width]).freed(&[(1, live)]),
            Element::Single,
        );
        let cursor = graph.input(Shape::of([1, planes, 1, 1]), Element::Single);
        let out = graph.attention(
            query,
            cache,
            cache,
            AttentionOptions {
                scale: Some(graph.knob(SCALE)),
                causal: true,
                origin: Some(cursor),
                segments: Some(ragged.offsets),
                reach: None,
                query_segments: None,
            },
        );
        let total = graph.sum(cache);
        graph.retain(out);
        graph.retain(ragged.offsets);
        graph.retain(total);
        Self {
            runtime: open(),
            graph,
            count,
            lengths,
            cursor,
            query,
            cache,
            out,
            offsets: ragged.offsets,
            total,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn run(&self, program: &Program, planes: u32, lengths: &[f32]) -> (Vec<f32>, Vec<f32>, f32) {
        let width = self.graph.shape(self.cache).dims()[3];
        let bound = self.graph.shape(self.cache).dims()[2];
        let cursor_planes = self.graph.shape(self.cursor).dims()[1];
        let cache = (0..bound * width)
            .map(|at| (at % 7) as f32 + 1.0)
            .collect::<Vec<f32>>();
        let query = data(cursor_planes * width, 29);
        let cursors = lengths
            .iter()
            .map(|length| (length - 1.0).max(0.0))
            .collect::<Vec<f32>>();
        self.runtime.write(program, self.count, &[planes as f32]);
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.cache, &cache);
        self.runtime.write(program, self.query, &query);
        self.runtime.write(program, self.cursor, &cursors);
        self.runtime.run(program);
        let produced = self.runtime.read(program, self.out);
        let scanned = self.runtime.read(program, self.offsets);
        let summed = self.runtime.read(program, self.total);
        let live = &lengths[..planes as usize];
        let packed = packed(live, &cache, width);
        assert_close(
            &produced,
            &expected(live, &packed, &query, &cursors, width),
            1e-5,
        );
        assert_eq!(
            self.runtime.read(program, self.cache).len(),
            packed.len(),
            "a count that rules the planes walks only the tokens those planes hold",
        );
        assert_close(&summed, &[packed.iter().sum()], 1e-5);
        (produced, scanned, summed[0])
    }
}

#[test]
fn a_ragged_axis_a_device_count_narrows_closes_only_the_planes_it_walks() {
    let planes = CountedPlanes::of(PLANES, WIDE_BOUND, WIDTH);
    let program = planes.compile();
    let (wide, scanned, _) = planes.run(&program, PLANES, &[3.0, 2.0, 300.0, 300.0]);
    assert_eq!(scanned, [0.0, 3.0, 5.0, 305.0, 605.0]);
    let (narrow, scanned, _) = planes.run(&program, 2, &[3.0, 2.0, 300.0, 300.0]);
    assert_eq!(
        narrow,
        wide[..narrow.len()],
        "the planes a device count holds walk the tokens their lengths name",
    );
    assert_eq!(
        scanned[..3],
        [0.0, 3.0, 5.0],
        "the prefix closes the offsets at the planes a device count holds",
    );
}

#[test]
fn a_device_count_of_the_planes_narrows_what_the_prefix_closes() {
    let graph: Graph<'static> = Graph::new();
    let flags = graph.input(Shape::of([1, 1, 1, PLANES]), Element::Single);
    let count = graph.sum(flags);
    let live = graph.counted(PLANES, count);
    let lengths = graph.input(
        Shape::of([1, 1, 1, PLANES]).freed(&[(3, live)]),
        Element::Single,
    );
    let ragged = graph.ragged(WIDE_BOUND, lengths);
    let cache = graph.resident(
        Shape::of([1, 1, WIDE_BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(
        Shape::of([1, PLANES, 1, WIDTH]).freed(&[(1, live)]),
        Element::Single,
    );
    let cursor = graph.input(Shape::of([1, PLANES, 1, 1]), Element::Single);
    let out = graph.attention(
        query,
        cache,
        cache,
        AttentionOptions {
            scale: Some(graph.knob(SCALE)),
            causal: true,
            origin: Some(cursor),
            segments: Some(ragged.offsets),
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    graph.retain(ragged.offsets);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let lengths_data = [3.0f32, 2.0, 5.0, 1.0];
    let cache_data = (0..WIDE_BOUND * WIDTH)
        .map(|at| (at % 7) as f32 + 1.0)
        .collect::<Vec<f32>>();
    let query_data = data(PLANES * WIDTH, 31);
    let cursor_data = lengths_data
        .iter()
        .map(|length| length - 1.0)
        .collect::<Vec<f32>>();
    runtime.write(&program, lengths, &lengths_data);
    runtime.write(&program, cache, &cache_data);
    runtime.write(&program, query, &query_data);
    runtime.write(&program, cursor, &cursor_data);
    for (plane_flags, live_planes) in [
        ([1.0f32, 1.0, 1.0, 1.0], PLANES),
        ([1.0, 1.0, 0.0, 0.0], 2),
        ([0.0, 0.0, 0.0, 0.0], 0),
    ] {
        runtime.write(&program, flags, &plane_flags);
        runtime.run(&program);
        let produced = runtime.read(&program, out);
        let scanned = runtime.read(&program, ragged.offsets);
        let live_lengths = &lengths_data[..live_planes as usize];
        let packed_cache = packed(live_lengths, &cache_data, WIDTH);
        assert_close(
            &produced,
            &expected(
                live_lengths,
                &packed_cache,
                &query_data,
                &cursor_data,
                WIDTH,
            ),
            1e-5,
        );
        assert_eq!(
            runtime.read(&program, cache).len(),
            packed_cache.len(),
            "the packed cache walks the tokens the planes a device count holds name",
        );
        assert_eq!(
            scanned[..live_lengths.len() + 1],
            offsets(live_lengths)
                .into_iter()
                .map(|offset| offset as f32)
                .collect::<Vec<f32>>(),
        );
    }
}

#[test]
fn a_windowed_segmented_attention_weighs_the_keys_a_plane_reaches() {
    for reach in [1, 2, 3] {
        let ragged = Ragged::windowed(PLANES, BOUND, WIDTH, reach);
        let program = ragged.compile();
        ragged.step(&program, &[3.0, 0.0, 5.0, 2.0], WIDTH);
        ragged.step(&program, &[6.0, 1.0, 2.0, 0.0], WIDTH);
    }
}

struct Packed<'a> {
    lengths: &'a [f32],
    keys: &'a [f32],
    values: &'a [f32],
    queries: &'a [f32],
    cursors: &'a [f32],
    gradient: &'a [f32],
    rows: u32,
    reach: u32,
}

fn row_map(lengths: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut plane = Vec::new();
    let mut position = Vec::new();
    for (walked, length) in lengths.iter().enumerate() {
        for row in 0..*length as u32 {
            plane.push(walked as f32);
            position.push(row as f32);
        }
    }
    (plane, position)
}

fn trainable_reference(packed: Packed<'_>, width: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let Packed {
        lengths,
        keys,
        values,
        queries,
        cursors,
        gradient,
        rows,
        reach,
    } = packed;
    let offsets = offsets(lengths);
    let planes = lengths.len();
    let width = width as usize;
    let rows = rows as usize;
    let live = *offsets
        .last()
        .expect("a ragged axis closes at least one plane") as usize;
    let mut out = vec![0.0f32; planes * rows * width];
    let mut query_grad = vec![0.0f32; planes * rows * width];
    let mut key_grad = vec![0.0f32; live * width];
    let mut value_grad = vec![0.0f32; live * width];
    for plane in 0..planes {
        let count = lengths[plane] as usize;
        let start = offsets[plane] as usize;
        let written = cursors[plane] as usize + rows;
        for row in 0..rows {
            let at = (plane * rows + row) * width;
            let query = &queries[at..][..width];
            let dout = &gradient[at..][..width];
            let position = cursors[plane] as usize + row;
            let mut logits = Vec::with_capacity(count);
            let mut attended = Vec::with_capacity(count);
            for key in 0..count {
                let slot = if reach > 0 && written > count {
                    key + count * ((written - 1 - key) / count)
                } else {
                    key
                };
                if slot > position {
                    continue;
                }
                if reach > 0 && position - slot >= reach as usize {
                    continue;
                }
                let packed = (start + key) * width;
                logits.push(
                    query
                        .iter()
                        .zip(&keys[packed..][..width])
                        .map(|(left, right)| left * right)
                        .sum::<f32>()
                        * SCALE,
                );
                attended.push(packed);
            }
            if attended.is_empty() {
                continue;
            }
            let peak = logits.iter().copied().fold(f32::MIN, f32::max);
            let weights = logits
                .iter()
                .map(|logit| (logit - peak).exp())
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
    (out, query_grad, key_grad, value_grad)
}

struct Trainable {
    runtime: Runtime,
    graph: Graph<'static>,
    lengths: Value<'static>,
    cursor: Value<'static>,
    queries: Value<'static>,
    keys: Value<'static>,
    values: Value<'static>,
    weight: Value<'static>,
    offsets: Value<'static>,
    out: Value<'static>,
    rows: [Value<'static>; 2],
    gradients: [Value<'static>; 3],
}

impl Trainable {
    fn windowed(planes: u32, bound: u32, width: u32, reach: u32) -> Self {
        Self::shaped(planes, 1, 1, bound, width, reach)
    }

    fn shaped(planes: u32, heads: u32, rows: u32, bound: u32, width: u32, reach: u32) -> Self {
        Self::build(planes, heads, rows, bound, width, reach, false)
    }

    fn bound(planes: u32, bound: u32, width: u32, reach: u32, binding: bool) -> Self {
        Self::build(planes, 1, 1, bound, width, reach, binding)
    }

    fn build(
        planes: u32,
        heads: u32,
        rows: u32,
        bound: u32,
        width: u32,
        reach: u32,
        binding: bool,
    ) -> Self {
        assert_eq!(planes % heads, 0, "a head carries a whole number of planes");
        let batch = planes / heads;
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
        let mapped = graph.rows(ragged);
        let packed = Shape::of([1, 1, bound, width]).freed(&[(2, ragged.extent)]);
        let planed = |dims: [u32; 4]| match live {
            Some(live) => Shape::of(dims).freed(&[(1, live)]),
            None => Shape::of(dims),
        };
        let queries = graph.gradient_input(planed([heads, batch, rows, width]), Element::Single);
        let keys = graph.gradient_input(packed, Element::Single);
        let values = graph.gradient_input(packed, Element::Single);
        let cursor = graph.input(Shape::of([heads, batch, 1, 1]), Element::Single);
        let out = graph.attention(
            queries,
            keys,
            values,
            AttentionOptions {
                scale: Some(graph.knob(SCALE)),
                causal: true,
                origin: Some(cursor),
                segments: Some(ragged.offsets),
                reach: (reach > 0).then_some(reach),
                query_segments: None,
            },
        );
        let weight = graph.input(planed([heads, batch, rows, width]), Element::Single);
        let loss = graph.sum(graph.mul(out, weight));
        let collected = graph.backward(loss);
        let gradients = [
            collected.of(queries),
            collected.of(keys),
            collected.of(values),
        ];
        for gradient in gradients {
            graph.retain(gradient);
        }
        graph.retain(mapped.plane);
        graph.retain(mapped.position);
        graph.retain(ragged.offsets);
        Self {
            runtime: open(),
            graph,
            lengths,
            cursor,
            queries,
            keys,
            values,
            weight,
            offsets: ragged.offsets,
            out,
            rows: [mapped.plane, mapped.position],
            gradients,
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
        self.runtime.bind(program, &[planes]);
    }

    fn step(&self, program: &Program, lengths: &[f32], reach: u32) -> Vec<Vec<f32>> {
        let width = self.graph.shape(self.keys).dims()[3];
        let bound = self.graph.shape(self.keys).dims()[2];
        let cursor_dims = self.graph.shape(self.cursor).dims();
        let bound_planes = cursor_dims[0] * cursor_dims[1];
        let rows = self.graph.shape(self.queries).dims()[2];
        let planes = lengths.len() as u32;
        let keys = data(bound * width, 43);
        let values = data(bound * width, 71);
        let queries = data(planes * rows * width, 17);
        let weight = data(planes * rows * width, 89);
        let cursors = lengths
            .iter()
            .map(|length| (length - rows as f32).max(0.0))
            .chain(std::iter::repeat_n(
                0.0,
                bound_planes as usize - lengths.len(),
            ))
            .collect::<Vec<f32>>();
        self.runtime.write(program, self.lengths, lengths);
        self.runtime.write(program, self.keys, &keys);
        self.runtime.write(program, self.values, &values);
        self.runtime.write(program, self.queries, &queries);
        self.runtime.write(program, self.weight, &weight);
        self.runtime.write(program, self.cursor, &cursors);
        self.runtime.run(program);
        let produced = self.runtime.read_many(
            program,
            &[
                self.out,
                self.gradients[0],
                self.gradients[1],
                self.gradients[2],
                self.rows[0],
                self.rows[1],
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
        let (out, query_grad, key_grad, value_grad) = trainable_reference(
            Packed {
                lengths,
                keys: &keys,
                values: &values,
                queries: &queries,
                cursors: &cursors,
                gradient: &weight,
                rows,
                reach,
            },
            width,
        );
        assert_close(&produced[0], &out, 1e-5);
        assert_close(&produced[1], &query_grad, 1e-5);
        assert_close(&produced[2], &key_grad, 1e-5);
        assert_close(&produced[3], &value_grad, 1e-5);
        let (plane, position) = row_map(lengths);
        assert_eq!(
            produced[4], plane,
            "every packed row walks the plane its offsets close",
        );
        assert_eq!(
            produced[5], position,
            "every packed row walks the place it holds in its plane",
        );
        produced
    }
}

#[test]
fn a_packed_attention_trains_the_queries_the_keys_and_the_values() {
    for lengths in [
        [3.0f32, 0.0, 5.0, 2.0].as_slice(),
        [4.0, 4.0, 4.0, 4.0].as_slice(),
        [1.0, 0.0, 0.0, 0.0].as_slice(),
        [0.0, 0.0, 0.0, 7.0].as_slice(),
        [6.0, 5.0, 3.0, 2.0].as_slice(),
        [2.0, 0.0, 14.0, 0.0].as_slice(),
    ] {
        let trainable = Trainable::windowed(PLANES, BOUND, WIDTH, 0);
        trainable.step(&trainable.compile(), lengths, 0);
    }
}

#[test]
fn a_packed_attention_trains_every_head_and_every_query_of_a_plane() {
    for (rows, lengths) in [
        (3, [3.0f32, 0.0, 5.0, 3.0].as_slice()),
        (2, [4.0, 4.0, 4.0, 4.0].as_slice()),
        (5, [6.0, 5.0, 5.0, 0.0].as_slice()),
        (1, [3.0, 0.0, 5.0, 2.0].as_slice()),
    ] {
        let trainable = Trainable::shaped(PLANES, 2, rows, BOUND, WIDTH, 0);
        trainable.step(&trainable.compile(), lengths, 0);
    }
}

#[test]
fn a_windowed_packed_attention_trains_the_keys_a_plane_reaches() {
    for reach in [1, 2, 3] {
        let trainable = Trainable::windowed(PLANES, BOUND, WIDTH, reach);
        let program = trainable.compile();
        trainable.step(&program, &[3.0, 0.0, 5.0, 2.0], reach);
        trainable.step(&program, &[6.0, 1.0, 2.0, 0.0], reach);
    }
    let trainable = Trainable::shaped(PLANES, 2, 2, BOUND, WIDTH, 2);
    let program = trainable.compile();
    trainable.step(&program, &[3.0, 0.0, 5.0, 2.0], 2);
    trainable.step(&program, &[6.0, 5.0, 3.0, 2.0], 2);
}

#[test]
fn a_bound_a_host_narrows_trains_only_the_planes_it_holds() {
    let trainable = Trainable::bound(PLANES, BOUND, WIDTH, 0, true);
    let program = trainable.compile();
    let lengths = [3.0f32, 2.0, 5.0, 4.0];
    trainable.bind(&program, PLANES);
    let wide = trainable.step(&program, &lengths, 0);
    for planes in [PLANES - 1, 1] {
        trainable.bind(&program, planes);
        let narrow = trainable.step(&program, &lengths[..planes as usize], 0);
        for (narrow, wide) in narrow.iter().zip(&wide) {
            assert_eq!(
                narrow,
                &wide[..narrow.len()],
                "the planes a binding holds walk the gradients their own lengths name",
            );
        }
    }
}

#[test]
fn a_plane_longer_than_a_workgroup_trains_every_key_of_it() {
    let trainable = Trainable::windowed(PLANES, 512, WIDTH, 0);
    let program = trainable.compile_narrow();
    trainable.step(&program, &[300.0, 70.0, 0.0, 5.0], 0);
}
