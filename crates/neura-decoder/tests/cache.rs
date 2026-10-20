use neura_abi::Element;
use neura_decoder::Cache;
use neura_graph::{Graph, Shape};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

const WIDTH: u32 = 2;
const CAPACITY: u32 = 8;
const WINDOW: u32 = 4;

#[test]
fn a_cache_keeps_the_last_window_of_keys() {
    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest::default(),
        ..Default::default()
    })
    .expect("a device that runs the cache test");
    let graph = Graph::new();
    let cursor = graph.input(Shape::scalar(), Element::Single);
    let query = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let keys = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let values = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let cache = Cache::windowed(&graph, 1, WIDTH, CAPACITY, WINDOW, 1, 1);
    cache.write(&graph, cursor, keys, values);
    let attended = cache.attend(&graph, query, cursor, graph.fill(Shape::scalar(), 1.0));
    graph.retain(attended);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, query, &[1.0, 0.0]);
    for step in 0..2 * WINDOW {
        runtime.write(&program, cursor, &[step as f32]);
        runtime.write(&program, keys, &[1.0, 0.0]);
        runtime.write(&program, values, &[step as f32, 0.0]);
        runtime.run(&program);
        let first = step.saturating_sub(WINDOW - 1);
        let count = step - first + 1;
        let expected = (first..=step).map(|at| at as f32).sum::<f32>() / count as f32;
        let held = runtime.read(&program, attended);
        assert!(
            (held[0] - expected).abs() < 1e-5 && held[1] == 0.0,
            "a window of {WINDOW} keys at step {step} weighs {held:?} where the last keys average {expected}",
        );
    }
}

#[test]
fn a_masked_chunk_leaves_the_ring_it_wraps_into_alone() {
    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest::default(),
        ..Default::default()
    })
    .expect("a device that runs the cache test");
    let graph = Graph::new();
    let first = graph.input(Shape::scalar(), Element::Single);
    let second = graph.input(Shape::scalar(), Element::Single);
    let query = graph.fill(Shape::of([1, 1, 1, WIDTH]), 0.0);
    let key = graph.fill(Shape::of([1, 1, 1, WIDTH]), 1.0);
    let values = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let masked_values = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let cache = Cache::windowed(&graph, 1, WIDTH, WINDOW + 1, WINDOW, 1, 1);
    cache.write(&graph, first, key, values);
    cache.write_masked(
        &graph,
        second,
        key,
        masked_values,
        graph.fill(Shape::scalar(), 0.0),
    );
    let attended = cache.attend(&graph, query, first, graph.fill(Shape::scalar(), 1.0));
    graph.retain(attended);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, first, &[0.0]);
    runtime.write(&program, second, &[WINDOW as f32]);
    runtime.write(&program, values, &[7.0, 0.0]);
    runtime.write(&program, masked_values, &[9.0, 0.0]);
    runtime.run(&program);
    let held = runtime.read(&program, attended);
    assert!(
        (held[0] - 7.0).abs() < 1e-5,
        "a masked write at the slot a wrapped cursor names left {held:?} where the first key stands",
    );
}
