use neura_abi::{Element, Kind};
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};
use neura_plan::Plan;
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
    free: neura_graph::Free,
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
        free: batch_extent,
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

#[test]
fn a_host_extent_and_a_device_count_share_one_graph() {
    let runtime = open();
    let graph = Graph::new();
    let model = build(&graph, BOUND, 2, true);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    assert_eq!(
        model.free.bound(),
        2,
        "the graph declares the batch extent the host binds",
    );
    runtime.bind(&program, &[2]);
    let tokens = data(2 * BOUND * WIDTH, 31);
    runtime.write(&program, model.tokens, &tokens);
    let assembled = runtime.assembled_kernels();
    for live in [BOUND, 3] {
        runtime.write(&program, model.probe, &live_probe(BOUND, live));
        runtime.run(&program);
        let actual = runtime.read(&program, model.out);
        let expected = reference(&runtime, live, 2, true, 31);
        assert_close(&actual, &expected, 1e-4);
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
    let plan = Plan::of(&graph, runtime.alignment(), program.profile());
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
