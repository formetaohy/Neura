use neura_abi::Element;
use neura_decoder::{Cache, Decoder, matrix, vector};
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
fn a_cache_walks_the_heads_of_the_keys_it_holds() {
    let graph = Graph::new();
    let cache = Cache::new(&graph, 2, 4, 8);
    let [keys, values] = cache.states();
    assert_eq!(graph.shape(keys).dims(), [2, 1, 8, 4]);
    assert_eq!(graph.shape(values).dims(), [2, 1, 8, 4]);
    let position = graph.input(Shape::matrix(1, 1), Element::Single);
    let query = graph.fill(Shape::of([2, 1, 1, 4]), 1.0);
    let scale = graph.knob(0.5);
    cache.write(
        &graph,
        position,
        query,
        graph.fill(Shape::of([2, 1, 1, 4]), 0.0),
    );
    assert_eq!(
        graph
            .shape(cache.attend(&graph, query, position, scale))
            .dims(),
        [2, 1, 1, 4],
    );
    let wide = graph.fill(Shape::of([3, 1, 1, 4]), 0.0);
    assert!(refuses(|| cache.write(&graph, position, wide, wide)));
}

#[test]
fn a_decoder_hands_out_the_token_it_chooses() {
    let graph = Graph::new();
    let cache = Cache::new(&graph, 2, 4, 8);
    let token = graph.input(Shape::matrix(1, 1), Element::Single);
    let position = graph.input(Shape::matrix(1, 1), Element::Single);
    let logits = graph.fill(Shape::of([1, 1, 1, 8]), 0.0);
    let decoder = Decoder::greedy(&graph, token, position, logits, cache.states(), 7);
    assert_eq!(graph.shape(decoder.chosen()).dims(), [1, 1, 1, 1]);
    assert_eq!(decoder.token().id(), token.id());
    assert_eq!(decoder.position().id(), position.id());
    assert_eq!(decoder.states().len(), 2);
    assert_eq!(decoder.end(), 7);
    let wide = graph.input(Shape::matrix(2, 1), Element::Single);
    assert!(refuses(|| Decoder::greedy(
        &graph,
        wide,
        position,
        logits,
        [],
        0
    )));
}

#[test]
fn a_narrow_cache_holds_the_keys_and_values_the_device_writes() {
    let graph = Graph::new();
    let single = Cache::new(&graph, 2, 4, 8);
    assert_eq!(single.element(), Element::Single);
    for element in [Element::Q8_0, Element::Half, Element::Bfloat16] {
        let cache = Cache::narrow(&graph, 2, 64, 8, element);
        assert_eq!(cache.element(), element);
        let [keys, values] = cache.states();
        assert_eq!(graph.shape(keys).dims(), [2, 1, 8, 64]);
        assert_eq!(graph.element(keys), element);
        assert_eq!(graph.element(values), element);
    }
    assert!(
        refuses(|| {
            Cache::narrow(&graph, 2, 33, 8, Element::Half);
        }),
        "a cache takes a written row in whole words, and a row of 33 half numbers spans 66 bytes",
    );
    for element in [Element::Int8, Element::Int4, Element::Q4K, Element::Q6K] {
        assert!(
            refuses(|| {
                Cache::narrow(&graph, 2, 32, 8, element);
            }),
            "a cache holds the keys a device writes in place, and {} storage is laid out by its host",
            element.name(),
        );
    }
    assert!(refuses(|| {
        Cache::narrow(&graph, 2, 32, 8, Element::Q8_0);
    }));
}
