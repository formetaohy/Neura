use neura_abi::{Element, Kind};
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};
use neura_plan::{DEFAULT_ENCODING_BYTES, Plan};
use neura_runtime::Runtime;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const BOUND: u32 = 8;
const WIDTH: u32 = 4;
const LONG: u32 = 4096;

fn weights() -> Init {
    Init::Uniform {
        low: -0.25,
        high: 0.25,
    }
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

fn live_probe(bound: u32, live: u32) -> Vec<f32> {
    (0..bound)
        .map(|row| if row < live { 1.0 } else { 0.0 })
        .collect()
}

struct Model {
    probe: Value<'static>,
    tokens: Value<'static>,
    live: Value<'static>,
    out: Value<'static>,
}

fn build(graph: &Graph<'static>, bound: u32, batch: u32, layers: bool) -> Model {
    let batch_extent = graph.free(batch);
    let probe = graph.input(Shape::of([1, 1, bound, 1]), Element::Single);
    let tokens = graph.input(
        Shape::of([1, batch, bound, WIDTH]).freed(&[(1, batch_extent)]),
        Element::Single,
    );
    let weight = graph.parameter(Shape::of([WIDTH, WIDTH]), weights(), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let live = graph.trim(tokens, 2, count);
    let projected = graph.matmul(live, weight);
    let out = if layers {
        let scaled = graph.mul(projected, projected);
        let attended = graph.attention(
            scaled,
            scaled,
            scaled,
            AttentionOptions {
                scale: 0.5,
                causal: true,
                origin: None,
                segments: None,
                reach: None,
                query_segments: None,
            },
        );
        graph.matmul(attended, weight)
    } else {
        projected
    };
    graph.retain(live);
    graph.retain(out);
    Model {
        probe,
        tokens,
        live,
        out,
    }
}

struct Counted {
    probe: Value<'static>,
    tokens: Value<'static>,
    out: Value<'static>,
}

fn counted(graph: &Graph<'static>, bound: u32) -> Counted {
    let probe = graph.input(Shape::of([1, 1, bound, 1]), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, bound, WIDTH]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let live = graph.trim(tokens, 2, count);
    let total = graph.sum(live);
    let folded = graph.sum_axis(live, WIDTH - 1);
    let out = graph.mul(folded, total);
    graph.retain(out);
    Counted { probe, tokens, out }
}

fn reference(runtime: &Runtime, live: u32, batch: u32, layers: bool, seed: u32) -> Vec<f32> {
    let graph = Graph::new();
    let model = build(&graph, live, batch, layers);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.bind(&program, &[batch]);
    runtime.write(&program, model.probe, &vec![1.0; live as usize]);
    runtime.write(&program, model.tokens, &data(batch * live * WIDTH, seed));
    runtime.run(&program);
    runtime.read(&program, model.out)
}

fn reference_counted(runtime: &Runtime, live: u32, probe: &[f32], tokens: &[f32]) -> Vec<f32> {
    let graph = Graph::new();
    let model = counted(&graph, live);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.write(&program, model.probe, probe);
    runtime.write(&program, model.tokens, tokens);
    runtime.run(&program);
    runtime.read(&program, model.out)
}

#[test]
fn a_device_count_rules_the_rows_a_product_walks() {
    let runtime = open();
    let graph = Graph::new();
    let model = build(&graph, BOUND, 1, false);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.bind(&program, &[1]);
    assert!(
        program.carries_authored(),
        "a graph that trims a tensor walks a length the device authors",
    );
    let tokens = data(BOUND * WIDTH, 11);
    runtime.write(&program, model.tokens, &tokens);
    for live in [BOUND, 5, 3, 1] {
        runtime.write(&program, model.probe, &live_probe(BOUND, live));
        runtime.run(&program);
        let actual = runtime.read(&program, model.out);
        let expected = reference(&runtime, live, 1, false, 11);
        assert_close(&actual, &expected, 1e-4);
    }
}

#[test]
fn a_device_count_of_zero_walks_no_row() {
    let runtime = open();
    let graph = Graph::new();
    let model = build(&graph, BOUND, 1, false);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.bind(&program, &[1]);
    runtime.write(&program, model.tokens, &data(BOUND * WIDTH, 3));
    runtime.write(&program, model.probe, &live_probe(BOUND, 0));
    runtime.run(&program);
    assert!(
        runtime.read(&program, model.out).is_empty(),
        "a count of zero live rows leaves a product of no row",
    );
}

#[test]
fn a_trimmed_length_carries_through_a_stack() {
    let runtime = open();
    let graph = Graph::new();
    let model = build(&graph, BOUND, 1, true);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.bind(&program, &[1]);
    let tokens = data(BOUND * WIDTH, 23);
    runtime.write(&program, model.tokens, &tokens);
    for live in [BOUND, 6, 2] {
        runtime.write(&program, model.probe, &live_probe(BOUND, live));
        runtime.run(&program);
        let actual = runtime.read(&program, model.out);
        let expected = reference(&runtime, live, 1, true, 23);
        assert_close(&actual, &expected, 1e-4);
    }
}

#[test]
fn a_trimmed_tensor_reads_back_the_rows_the_device_count_leaves() {
    let runtime = open();
    let graph = Graph::new();
    let model = build(&graph, BOUND, 1, false);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.bind(&program, &[1]);
    let tokens = data(BOUND * WIDTH, 17);
    runtime.write(&program, model.tokens, &tokens);
    for live in [BOUND, 4, 0] {
        runtime.write(&program, model.probe, &live_probe(BOUND, live));
        runtime.run(&program);
        assert_close(
            &runtime.read(&program, model.live),
            &tokens[..(live * WIDTH) as usize],
            0.0,
        );
    }
}

#[test]
fn a_reduction_over_a_device_count_walks_the_live_rows() {
    let runtime = open();
    let graph = Graph::new();
    let model = counted(&graph, LONG);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let tokens = data(LONG * WIDTH, 41);
    runtime.write(&program, model.tokens, &tokens);
    for live in [LONG, LONG - 1, 3000, 7] {
        let probe = live_probe(LONG, live);
        runtime.write(&program, model.probe, &probe);
        runtime.run(&program);
        let actual = runtime.read(&program, model.out);
        let expected = reference_counted(
            &runtime,
            live,
            &probe[..live as usize],
            &tokens[..(live * WIDTH) as usize],
        );
        assert_close(&actual, &expected, 1e-2);
    }
}

struct HostCounted<'g> {
    count: Value<'g>,
    tokens: Value<'g>,
    out: Value<'g>,
}

fn host_counted(graph: &Graph<'static>, bound: u32) -> HostCounted<'static> {
    let count = graph.input(Shape::scalar(), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, bound, WIDTH]), Element::Single);
    let weight = graph.parameter(Shape::of([WIDTH, WIDTH]), weights(), Element::Single);
    let live = graph.trim(tokens, 2, count);
    let out = graph.matmul(live, weight);
    graph.retain(out);
    HostCounted { count, tokens, out }
}

fn host_reference(runtime: &Runtime, rows: u32, tokens: &[f32]) -> Vec<f32> {
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, rows, WIDTH]), Element::Single);
    let weight = graph.parameter(Shape::of([WIDTH, WIDTH]), weights(), Element::Single);
    let out = graph.matmul(data, weight);
    graph.retain(out);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.write(&program, data, tokens);
    runtime.run(&program);
    runtime.read(&program, out)
}

#[test]
fn a_count_the_host_writes_rules_the_rows_a_product_walks() {
    let runtime = open();
    let graph = Graph::new();
    let model = host_counted(&graph, BOUND);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let tokens = data(BOUND * WIDTH, 13);
    runtime.write(&program, model.tokens, &tokens);
    for rows in [BOUND, 6, 1] {
        runtime.write(&program, model.count, &[rows as f32]);
        runtime.run(&program);
        let actual = runtime.read(&program, model.out);
        let expected = host_reference(&runtime, rows, &tokens[..(rows * WIDTH) as usize]);
        assert_close(&actual, &expected, 1e-4);
    }
}

#[test]
fn a_count_beyond_the_bound_the_graph_declares_is_refused() {
    let runtime = open();
    let graph = Graph::new();
    let model = host_counted(&graph, BOUND);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.write(&program, model.tokens, &data(BOUND * WIDTH, 19));
    runtime.write(&program, model.count, &[(BOUND + 1) as f32]);
    runtime.run(&program);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.read(&program, model.out)
    }));
    assert!(
        refused.is_err(),
        "a count beyond the bound the graph declares was accepted",
    );
}

const CUT_PLANES: u32 = 4;

struct Cut {
    mask: Value<'static>,
    planes: Value<'static>,
    live: Value<'static>,
    out: Value<'static>,
}

fn cut(graph: &Graph<'static>, planes: u32, tokens: u32) -> Cut {
    let token_extent = graph.free(tokens);
    let mask = graph.input(Shape::of([planes, 1, 1, 1]), Element::Single);
    let held = graph.input(
        Shape::of([planes, 1, tokens, WIDTH]).freed(&[(2, token_extent)]),
        Element::Single,
    );
    let count = graph.sum_axis(mask, 0);
    let live = graph.trim(held, 0, count);
    let out = graph.sum_axis(live, 2);
    graph.retain(live);
    graph.retain(out);
    Cut {
        mask,
        planes: held,
        live,
        out,
    }
}

#[test]
fn a_device_count_cuts_the_planes_a_host_extent_holds() {
    let runtime = open();
    let graph = Graph::new();
    let model = cut(&graph, CUT_PLANES, BOUND);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let plan = Plan::of(
        &graph,
        runtime.alignment(),
        program.profile(),
        DEFAULT_ENCODING_BYTES,
    );
    assert!(
        plan.carries_authored(),
        "the planes of the batch walk an extent the device authors",
    );
    assert_eq!(
        plan.host_slots().len(),
        1,
        "the token extent is bound by the host while the device authors the planes",
    );
    let assembled = runtime.assembled_kernels();
    for (planes, tokens) in [(CUT_PLANES, BOUND), (CUT_PLANES, 3), (2, BOUND), (1, 2)] {
        runtime.bind(&program, &[tokens]);
        let image = data(CUT_PLANES * tokens * WIDTH, 37);
        runtime.write(&program, model.planes, &image);
        runtime.write(&program, model.mask, &live_probe(CUT_PLANES, planes));
        runtime.run(&program);
        let walked = runtime.read(&program, model.live);
        assert_close(&walked, &image[..(planes * tokens * WIDTH) as usize], 0.0);
        let summed = runtime.read(&program, model.out);
        let mut expected = Vec::new();
        for plane in 0..planes {
            for column in 0..WIDTH {
                let first = plane * tokens * WIDTH + column;
                let mut total = 0.0;
                for row in 0..tokens {
                    total += image[(first + row * WIDTH) as usize];
                }
                expected.push(total);
            }
        }
        assert_close(&summed, &expected, 1e-4);
    }
    assert_eq!(
        runtime.assembled_kernels(),
        assembled,
        "one device program serves every count a graph walks",
    );
}

const SPLIT_TOKENS: u32 = 128;
const SPLIT_DEPTH: u32 = 1024;
const SPLIT_COLUMNS: u32 = 64;

struct Split {
    probe: Value<'static>,
    tokens: Value<'static>,
    out: Value<'static>,
}

fn split_product(graph: &Graph<'static>, bound: u32) -> Split {
    let probe = graph.input(Shape::of([1, 1, bound, 1]), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, bound, SPLIT_DEPTH]), Element::Single);
    let weight = graph.parameter(
        Shape::matrix(SPLIT_DEPTH, SPLIT_COLUMNS),
        weights(),
        Element::Single,
    );
    let count = graph.sum_axis(probe, 2);
    let live = graph.trim(tokens, 2, count);
    let out = graph.matmul(live, weight);
    graph.retain(out);
    Split { probe, tokens, out }
}

fn split_reference(runtime: &Runtime, rows: u32, tokens: &[f32]) -> Vec<f32> {
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, rows, SPLIT_DEPTH]), Element::Single);
    let weight = graph.parameter(
        Shape::matrix(SPLIT_DEPTH, SPLIT_COLUMNS),
        weights(),
        Element::Single,
    );
    let out = graph.matmul(data, weight);
    graph.retain(out);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.write(&program, data, tokens);
    runtime.run(&program);
    runtime.read(&program, out)
}

#[test]
fn a_device_count_rules_the_rows_a_depth_split_product_walks() {
    let runtime = open();
    let graph = Graph::new();
    let model = split_product(&graph, SPLIT_TOKENS);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let plan = Plan::of(
        &graph,
        runtime.alignment(),
        program.profile(),
        DEFAULT_ENCODING_BYTES,
    );
    assert!(
        plan.kinds().contains(&Kind::MatmulFold),
        "the product splits its depth across tasks, and the layout of the partials the fold reads is the one the live rows give",
    );
    let tokens = data(SPLIT_TOKENS * SPLIT_DEPTH, 29);
    runtime.write(&program, model.tokens, &tokens);
    for live in [
        SPLIT_TOKENS,
        SPLIT_TOKENS * 3 / 4,
        SPLIT_TOKENS / 2,
        SPLIT_TOKENS / 4,
    ] {
        runtime.write(&program, model.probe, &live_probe(SPLIT_TOKENS, live));
        runtime.run(&program);
        let actual = runtime.read(&program, model.out);
        let expected = split_reference(&runtime, live, &tokens[..(live * SPLIT_DEPTH) as usize]);
        assert_close(&actual, &expected, 1e-3);
    }
}

const WIDE_ROWS: u32 = 12;
const NARROW_ROWS: u32 = 6;

struct Shared<'g> {
    probe: Value<'g>,
    longer: Value<'g>,
    shorter: Value<'g>,
    factor: Value<'g>,
    out: Value<'g>,
}

fn shared(graph: &Graph<'static>) -> Shared<'static> {
    let probe = graph.input(Shape::of([1, 1, WIDE_ROWS, 1]), Element::Single);
    let count = graph.relu(graph.sum(probe));
    let longer = graph.input(Shape::of([1, 1, WIDE_ROWS, WIDTH]), Element::Single);
    let shorter = graph.input(Shape::of([1, 1, NARROW_ROWS, WIDTH]), Element::Single);
    let factor = graph.input(Shape::scalar(), Element::Single);
    let wide = graph.trim(longer, 2, count);
    let short = graph.trim(shorter, 2, count);
    let out = graph.mul(
        graph.add(graph.sum(wide), graph.sum(short)),
        graph.mul(count, factor),
    );
    graph.retain(out);
    Shared {
        probe,
        longer,
        shorter,
        factor,
        out,
    }
}

#[test]
fn a_device_count_rules_the_length_of_every_tensor_it_authors() {
    let runtime = open();
    let graph = Graph::new();
    let model = shared(&graph);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    assert!(
        program.carries_authored(),
        "a graph that trims two tensors walks one count the device authors",
    );
    let longer = data(WIDE_ROWS * WIDTH, 71);
    let shorter = data(NARROW_ROWS * WIDTH, 73);
    runtime.write(&program, model.longer, &longer);
    runtime.write(&program, model.shorter, &shorter);
    for (live, factor) in [(6u32, 2.0f32), (3, -1.5), (1, 0.5), (0, 4.0)] {
        runtime.write(&program, model.probe, &live_probe(WIDE_ROWS, live));
        runtime.write(&program, model.factor, &[factor]);
        runtime.run(&program);
        let walked = live as usize * WIDTH as usize;
        let expected = (longer[..walked].iter().sum::<f32>()
            + shorter[..walked].iter().sum::<f32>())
            * (live as f32 * factor);
        assert_close(&runtime.read(&program, model.out), &[expected], 1e-3);
    }
}

#[test]
fn a_device_count_weighs_the_length_a_mean_divides_by() {
    let runtime = open();
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, BOUND, 1]), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, BOUND, WIDTH]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let live = graph.trim(tokens, 2, count);
    let mean = graph.mean_axis(live, 2);
    graph.retain(mean);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    assert!(
        program.carries_authored(),
        "a graph that trims a tensor walks a length the device authors",
    );
    let tokens_data = data(BOUND * WIDTH, 29);
    runtime.write(&program, tokens, &tokens_data);
    for live in [BOUND, 5, 1] {
        runtime.write(&program, probe, &live_probe(BOUND, live));
        runtime.run(&program);
        let expected = (0..WIDTH)
            .map(|column| {
                let total = (0..live as usize)
                    .map(|row| tokens_data[row * WIDTH as usize + column as usize])
                    .sum::<f32>();
                total / live as f32
            })
            .collect::<Vec<f32>>();
        assert_close(&runtime.read(&program, mean), &expected, 1e-5);
    }
    runtime.write(&program, probe, &live_probe(BOUND, 0));
    runtime.run(&program);
    assert!(
        runtime
            .read(&program, mean)
            .iter()
            .all(|value| value.is_nan()),
        "a mean of no numbers divides an empty sum by an empty length",
    );
}

#[test]
fn a_device_count_beyond_one_bound_it_walks_is_refused() {
    let runtime = open();
    let graph = Graph::new();
    let model = shared(&graph);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.write(&program, model.longer, &data(WIDE_ROWS * WIDTH, 79));
    runtime.write(&program, model.shorter, &data(NARROW_ROWS * WIDTH, 83));
    runtime.write(&program, model.factor, &[1.0]);
    runtime.write(
        &program,
        model.probe,
        &live_probe(WIDE_ROWS, NARROW_ROWS + 1),
    );
    runtime.run(&program);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.read(&program, model.out)
    }));
    assert!(
        refused.is_err(),
        "a count beyond the bound of one tensor it walks was accepted",
    );
}
