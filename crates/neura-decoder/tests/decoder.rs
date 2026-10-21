use neura_abi::Element;
use neura_decoder::{Cache, CacheShape, matrix, vector};
use neura_gguf::Gguf;
use neura_graph::{Graph, Shape};

#[path = "support/mod.rs"]
mod support;

use support::{build, quantized, singles};

fn refuses<T>(action: impl FnOnce() -> T) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn archive(tensors: &[(&str, u32, &[u64], &[u8])]) -> Gguf {
    Gguf::decode(&build(tensors))
}

#[test]
fn a_matrix_reads_the_rows_and_the_columns_the_file_weighs() {
    let payload = singles(&[0.0; 6]);
    let source = archive(&[("weight", 0, &[3, 2], &payload)]);
    let graph = Graph::new();
    let held = matrix(&graph, &source, "weight", 2, 3);
    assert_eq!(graph.shape(held).dims(), [1, 1, 2, 3]);
    assert_eq!(graph.element(held), Element::Single);
    assert!(refuses(|| matrix(&graph, &source, "weight", 3, 2)));
    assert!(refuses(|| matrix(&graph, &source, "other", 2, 3)));
}

#[test]
fn a_quantized_matrix_walks_the_blocks_the_file_packs() {
    let payload = quantized(4);
    let source = archive(&[("weight", 8, &[32, 4], &payload)]);
    let graph = Graph::new();
    let held = matrix(&graph, &source, "weight", 4, 32);
    assert_eq!(graph.shape(held).dims(), [1, 1, 4, 32]);
    assert_eq!(graph.element(held), Element::Q8_0);
}

#[test]
fn a_vector_reads_the_width_the_file_weighs() {
    let payload = singles(&[0.0; 4]);
    let source = archive(&[("norm", 0, &[4], &payload)]);
    let graph = Graph::new();
    let held = vector(&graph, &source, "norm", 4);
    assert_eq!(graph.shape(held).dims(), [1, 1, 1, 4]);
    assert!(refuses(|| vector(&graph, &source, "norm", 5)));
    assert!(refuses(|| vector(&graph, &source, "weight", 4)));
}

#[test]
fn a_cache_takes_the_rows_of_every_chunk_it_holds() {
    let graph = Graph::new();
    let cache = Cache::new(
        &graph,
        "cache",
        CacheShape::new(2, 4, 8, 8, 1, 3),
        Element::Single,
    );
    let [keys, values] = cache.states();
    assert_eq!(graph.shape(keys).dims(), [2, 1, 8, 4]);
    assert_eq!(graph.shape(values).dims(), [2, 1, 8, 4]);
    assert_eq!(cache.rows(), 3);
    let cursor = graph.input(Shape::scalar(), Element::Single);
    let chunk = graph.fill(Shape::of([2, 1, 3, 4]), 1.0);
    let scale = graph.knob(0.5);
    cache.write(
        &graph,
        cursor,
        chunk,
        graph.fill(Shape::of([2, 1, 3, 4]), 0.0),
    );
    assert_eq!(
        graph
            .shape(cache.attend(&graph, chunk, cursor, scale))
            .dims(),
        [2, 1, 3, 4],
    );
    let row = graph.fill(Shape::of([2, 1, 1, 4]), 0.0);
    assert!(refuses(|| cache.write(&graph, cursor, row, chunk)));
    assert!(refuses(|| {
        cache.write(
            &graph,
            cursor,
            chunk,
            graph.fill(Shape::of([3, 1, 3, 4]), 0.0),
        )
    }));
    assert!(refuses(|| cache.attend(&graph, row, cursor, scale)));
    assert!(refuses(|| {
        cache.attend(
            &graph,
            graph.fill(Shape::of([2, 1, 3, 3]), 0.0),
            cursor,
            scale,
        )
    }));
    assert!(refuses(|| Cache::new(
        &graph,
        "first",
        CacheShape::new(2, 4, 8, 8, 1, 9),
        Element::Single
    )));
    assert!(refuses(|| Cache::new(
        &graph,
        "second",
        CacheShape::new(2, 4, 8, 8, 1, 0),
        Element::Single
    )));
}
