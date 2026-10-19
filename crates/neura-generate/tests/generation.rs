use neura_abi::Element;
use neura_decoder::Cache;
use neura_generate::{Batch, BatchFill, BatchSpec, BatchStep, Family, Generator, Sampling, Step};
use neura_gguf::Gguf;
use neura_graph::Shape;
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};
use neura_tokenizer::{Rule, Tokenizer, Vocabulary};

#[path = "support/mod.rs"]
mod support;

use support::{Meta, build};

const CLASSES: u32 = 8;
const CAPACITY: u32 = 16;
const END: u32 = 3;

struct PerCharacter;

impl Rule for PerCharacter {
    fn pieces<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = Vec::new();
        let mut at = 0usize;
        for character in text.chars() {
            let next = at + character.len_utf8();
            pieces.push(&text[at..next]);
            at = next;
        }
        pieces
    }
}

fn tokenizer() -> Tokenizer {
    let tokens = (0..8).map(|held| held.to_string()).collect::<Vec<String>>();
    let held = tokens.iter().map(String::as_str).collect::<Vec<&str>>();
    let bytes = build(&[
        ("tokenizer.ggml.model", Meta::Text("gpt2")),
        ("tokenizer.ggml.tokens", Meta::Texts(&held)),
        ("tokenizer.ggml.token_type", Meta::Words(&[1; 8])),
        ("tokenizer.ggml.merges", Meta::Texts(&[])),
    ]);
    Tokenizer::new(Vocabulary::read(&Gguf::decode(&bytes)), PerCharacter)
}

fn runtime() -> Runtime {
    Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("a device that runs the tests")
}

fn family(runtime: &Runtime, chunks: &[u32], sampling: Option<Sampling>) -> Family {
    let build = |graph: &neura_graph::Graph, rows| {
        let token = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
        let cursor = graph.input(Shape::scalar(), Element::Single);
        let cache = Cache::new(graph, 1, CLASSES, CAPACITY, 1, rows);
        let held = graph.one_hot(token, CLASSES);
        cache.write(graph, cursor, held, held);
        Step {
            token,
            cursor,
            logits: graph.sum_axis(cache.states()[0], 2),
        }
    };
    match sampling {
        Some(sampling) => Family::sampled(
            runtime,
            |graph| runtime.weights(graph),
            chunks,
            build,
            sampling,
        ),
        None => Family::of(runtime, |graph| runtime.weights(graph), chunks, build),
    }
}

fn generate(
    prompt: &[u32],
    context: u32,
    wanted: usize,
    chunks: &[u32],
    sampling: Option<Sampling>,
) -> (usize, String, bool) {
    let runtime = runtime();
    let family = family(&runtime, chunks, sampling);
    let tokenizer = tokenizer();
    let mut generator = Generator::new(runtime, family, &tokenizer, END, context);
    let mut text = String::new();
    let report = generator.run(prompt, wanted, |held| text.push_str(held));
    (report.produced(), text, report.reached_context())
}

#[test]
fn a_generator_prefills_a_prompt_in_the_chunks_its_model_takes() {
    let walked = generate(&[0, 1, 1, 2, 1], 16, 5, &[1], None);
    assert_eq!(walked, (5, "0112111111".to_owned(), false));
    for chunks in [&[2u32][..], &[3], &[4], &[8], &[4, 1], &[3, 2, 1]] {
        assert_eq!(
            generate(&[0, 1, 1, 2, 1], 16, 5, chunks, None),
            walked,
            "a prefill in the chunks {chunks:?} answers with the tokens a row at a time answers with",
        );
    }
}

#[test]
fn a_generator_stops_at_the_token_count_it_was_asked_for() {
    assert_eq!(
        generate(&[0, 1, 1, 2, 1], 16, 2, &[3], None),
        (2, "0112111".to_owned(), false),
    );
}

#[test]
fn a_generator_stops_at_the_context_it_holds() {
    assert_eq!(
        generate(&[0], 3, 5, &[2], None),
        (2, "000".to_owned(), true)
    );
}

#[test]
fn a_generator_samples_the_token_its_logits_and_seed_choose() {
    let walked = generate(
        &[0, 1, 1, 2, 1],
        16,
        5,
        &[3],
        Some(Sampling::sampled(1.0, 4, 1.0)),
    );
    assert_eq!(walked, (5, "0112111111".to_owned(), false));
}

fn batch_answer(graph: &neura_graph::Graph, batch: u32, capacity: u32) -> BatchStep {
    let token = graph.input(Shape::of([1, 1, batch, 1]), Element::Single);
    let cursor = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
    let mask = graph.input(Shape::of([1, 1, batch, 1]), Element::Single);
    let reset = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
    let cache = Cache::new(graph, 1, CLASSES, capacity, batch, 1);
    let held = graph.mul(
        graph.permute(graph.one_hot(token, CLASSES), [0, 2, 1, 3]),
        graph.permute(mask, [0, 2, 1, 3]),
    );
    cache.write(graph, cursor, held, held);
    BatchStep {
        token,
        cursor,
        mask,
        reset,
        logits: graph.reshape(
            graph.sum_axis(cache.states()[0], 2),
            Shape::of([1, 1, batch, CLASSES]),
        ),
    }
}

fn batch_fill(graph: &neura_graph::Graph, rows: u32, batch: u32, capacity: u32) -> BatchFill {
    let token = graph.input(Shape::of([1, rows, batch, 1]), Element::Single);
    let cursor = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
    let mask = graph.input(Shape::of([1, 1, batch, 1]), Element::Single);
    let cache = Cache::new(graph, 1, CLASSES, capacity, batch, rows);
    let held = graph.mul(
        graph.permute(graph.one_hot(token, CLASSES), [0, 2, 1, 3]),
        graph.permute(mask, [0, 2, 1, 3]),
    );
    cache.write(graph, cursor, held, held);
    BatchFill {
        token,
        cursor,
        mask,
    }
}

fn batch_reset(graph: &neura_graph::Graph, batch: u32, capacity: u32) -> neura_graph::Value {
    let reset = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
    let cache = Cache::new(graph, 1, CLASSES, capacity, batch, 1);
    cache.reset(graph, reset);
    reset
}

fn batched(
    first_prompt: &[u32],
    second_prompt: &[u32],
    wanted: usize,
    chunks: &[u32],
    stagger: bool,
    sampling: Option<Sampling>,
) -> Vec<Vec<u32>> {
    let runtime = runtime();
    let context = 16;
    let batch = 2;
    let spec = BatchSpec::new(context, batch, chunks);
    let spec = match sampling {
        Some(sampling) => spec.sampled(sampling),
        None => spec,
    };
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    let first = batch.admit(&runtime, first_prompt);
    let mut walked = vec![Vec::new(); batch.batch() as usize];
    if stagger {
        while walked[first as usize].len() < 2 {
            for held in batch.step(&runtime) {
                if held.slot == first {
                    walked[first as usize].push(held.token);
                }
            }
        }
    }
    let second = batch.admit(&runtime, second_prompt);
    while walked.iter().any(|held| held.len() < wanted) {
        for held in batch.step(&runtime) {
            if held.token == END || walked[held.slot as usize].len() >= wanted {
                batch.finish(held.slot);
                continue;
            }
            walked[held.slot as usize].push(held.token);
        }
        assert!(
            batch.active().next().is_some(),
            "a batch with no token left still walked",
        );
    }
    assert_eq!((first, second), (0, 1));
    walked
}

fn walked_alone(
    prompt: &[u32],
    wanted: usize,
    chunks: &[u32],
    sampling: Option<Sampling>,
) -> Vec<u32> {
    let runtime = runtime();
    let context = 16;
    let batch = 1;
    let spec = BatchSpec::new(context, batch, chunks);
    let spec = match sampling {
        Some(sampling) => spec.sampled(sampling),
        None => spec,
    };
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    batch.admit(&runtime, prompt);
    let mut walked = Vec::new();
    while walked.len() < wanted && batch.active().next().is_some() {
        for held in batch.step(&runtime) {
            if held.token == END {
                batch.finish(held.slot);
                continue;
            }
            walked.push(held.token);
        }
    }
    for slot in batch.active().collect::<Vec<u32>>() {
        batch.finish(slot);
    }
    walked
}

#[test]
fn a_batch_answers_every_plane_with_the_tokens_a_row_answers_with() {
    for stagger in [false, true] {
        let walked = batched(&[0, 1, 1, 2, 1], &[2, 2, 0], 5, &[], stagger, None);
        assert_eq!(
            walked,
            vec![vec![1, 1, 1, 1, 1], vec![2, 2, 2, 2, 2]],
            "stagger {stagger}",
        );
    }
}

#[test]
fn a_batch_fills_a_single_prompt_in_the_chunks_its_model_takes() {
    let plain = walked_alone(&[0, 1, 1, 2, 1], 5, &[], None);
    let chunked = walked_alone(&[0, 1, 1, 2, 1], 5, &[8, 4, 2], None);
    assert_eq!(chunked, plain);
    let sampled = walked_alone(
        &[0, 1, 1, 2, 1],
        5,
        &[8, 4, 2],
        Some(Sampling::sampled(1.0, 4, 1.0)),
    );
    assert_eq!(sampled.len(), 5);
}
