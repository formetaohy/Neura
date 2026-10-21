use neura_abi::Element;
use neura_decoder::{Cache, CacheShape};
use neura_graph::{Graph, Init, Shape};
use neura_runtime::{CheckpointFile, MemoryRequest, Runtime, RuntimeRequest};

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
    for element in [Element::Single, Element::Half] {
        let graph = Graph::new();
        let cursor = graph.input(Shape::scalar(), Element::Single);
        let query = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
        let keys = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
        let values = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
        let cache = Cache::windowed(
            &graph,
            "cache",
            CacheShape::new(1, WIDTH, CAPACITY, WINDOW, 1, 1),
            element,
        );
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
                "a window of {WINDOW} keys in {} storage at step {step} weighs {held:?} where the last keys average {expected}",
                element.name(),
            );
        }
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
    let cache = Cache::windowed(
        &graph,
        "cache",
        CacheShape::new(1, WIDTH, WINDOW + 1, WINDOW, 1, 1),
        Element::Single,
    );
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

#[test]
fn a_half_cache_clears_the_planes_it_resets() {
    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest::default(),
        ..Default::default()
    })
    .expect("a device that runs the cache test");
    let graph = Graph::new();
    let query = graph.input(Shape::of([1, 2, 1, WIDTH]), Element::Single);
    let cursor = graph.fill(Shape::scalar(), 0.0);
    let keys = graph.fill(Shape::of([1, 2, 1, WIDTH]), 1.0);
    let values = graph.input(Shape::of([1, 2, 1, WIDTH]), Element::Single);
    let reset = graph.input(Shape::of([1, 2, 1, 1]), Element::Single);
    let cache = Cache::new(
        &graph,
        "cache",
        CacheShape::new(1, WIDTH, CAPACITY, CAPACITY, 2, 1),
        Element::Half,
    );
    cache.write(&graph, cursor, keys, values);
    cache.reset(&graph, reset, None);
    let attended = cache.attend(&graph, query, cursor, graph.fill(Shape::scalar(), 1.0));
    graph.retain(attended);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, query, &[1.0, 0.0, 1.0, 0.0]);
    runtime.write(&program, values, &[5.0, 0.0, 5.0, 0.0]);
    runtime.write(&program, reset, &[1.0, 0.0]);
    runtime.run(&program);
    let held = runtime.read(&program, attended);
    assert_eq!(held, vec![0.0, 0.0, 5.0, 0.0]);
}

#[test]
fn a_saved_cache_state_walks_the_keys_it_held() {
    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest::default(),
        ..Default::default()
    })
    .expect("a device that runs the cache test");
    let graph = Graph::new();
    let cursor = graph.input(Shape::scalar(), Element::Single);
    let reset = graph.input(Shape::scalar(), Element::Single);
    let query = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let keys = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let values = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let cache = Cache::windowed(
        &graph,
        "cache",
        CacheShape::new(1, WIDTH, CAPACITY, WINDOW, 1, 1),
        Element::Single,
    );
    graph.named_parameter(
        "weight",
        Shape::matrix(32, 32),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    cache.write(&graph, cursor, keys, values);
    cache.reset(&graph, reset, None);
    let attended = cache.attend(&graph, query, cursor, graph.fill(Shape::scalar(), 1.0));
    graph.retain(attended);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, query, &[1.0, 0.0]);
    runtime.write(&program, keys, &[1.0, 0.0]);
    for step in 0..WINDOW {
        runtime.write(&program, cursor, &[step as f32]);
        runtime.write(&program, values, &[step as f32, 0.0]);
        runtime.write(&program, reset, &[0.0]);
        runtime.run(&program);
    }
    let held = runtime.read(&program, attended);
    let expected = (0..WINDOW).sum::<u32>() as f32 / WINDOW as f32;
    assert!((held[0] - expected).abs() < 1e-5);
    let path = std::env::temp_dir().join(format!("neura-cache-{}.neura", std::process::id()));
    let full = std::env::temp_dir().join(format!("neura-full-{}.neura", std::process::id()));
    runtime.save(&weights, &full);
    runtime.save_state(&weights, &path);
    assert!(
        std::fs::metadata(&path)
            .expect("a state checkpoint holds its bytes")
            .len()
            < std::fs::metadata(&full)
                .expect("a full checkpoint holds its bytes")
                .len(),
        "a state checkpoint holds the state alone beside the weights a full checkpoint holds",
    );
    runtime.write(&program, cursor, &[(WINDOW - 1) as f32]);
    runtime.write(&program, reset, &[1.0]);
    runtime.run(&program);
    let cleared = runtime.read(&program, attended);
    assert_eq!(cleared, vec![0.0, 0.0]);
    runtime.restore_state(&weights, &CheckpointFile::open(&path));
    runtime.write(&program, cursor, &[(WINDOW - 1) as f32]);
    runtime.write(&program, reset, &[0.0]);
    runtime.run(&program);
    let restored = runtime.read(&program, attended);
    assert_eq!(restored, held);
    std::fs::remove_file(&path).expect("a saved cache state leaves its file");
    std::fs::remove_file(&full).expect("a saved cache leaves its full checkpoint");
}
