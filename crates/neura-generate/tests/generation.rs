use neura_abi::Element;
use neura_decoder::Cache;
use neura_generate::{Family, Generator, Step};
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

fn family(runtime: &Runtime, chunks: &[u32]) -> Family {
    Family::of(
        runtime,
        |graph| runtime.weights(graph),
        chunks,
        |graph, rows| {
            let token = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
            let cursor = graph.input(Shape::scalar(), Element::Single);
            let cache = Cache::new(graph, 1, CLASSES, CAPACITY, rows);
            let held = graph.one_hot(token, CLASSES);
            cache.write(graph, cursor, held, held);
            Step {
                token,
                cursor,
                logits: graph.sum_axis(cache.states()[0], 2),
            }
        },
    )
}

fn generate(prompt: &[u32], context: u32, wanted: usize, chunks: &[u32]) -> (usize, String, bool) {
    let runtime = runtime();
    let family = family(&runtime, chunks);
    let tokenizer = tokenizer();
    let mut generator = Generator::new(runtime, family, &tokenizer, END, context);
    let mut text = String::new();
    let report = generator.run(prompt, wanted, |held| text.push_str(held));
    (report.produced(), text, report.reached_context())
}

#[test]
fn a_generator_prefills_a_prompt_in_the_chunks_its_model_takes() {
    let walked = generate(&[0, 1, 1, 2, 1], 16, 5, &[1]);
    assert_eq!(walked, (5, "0112111111".to_owned(), false));
    for chunks in [&[2u32][..], &[3], &[4], &[8], &[4, 1], &[3, 2, 1]] {
        assert_eq!(
            generate(&[0, 1, 1, 2, 1], 16, 5, chunks),
            walked,
            "a prefill in the chunks {chunks:?} answers with the tokens a row at a time answers with",
        );
    }
}

#[test]
fn a_generator_stops_at_the_token_count_it_was_asked_for() {
    assert_eq!(
        generate(&[0, 1, 1, 2, 1], 16, 2, &[3]),
        (2, "0112111".to_owned(), false),
    );
}

#[test]
fn a_generator_stops_at_the_context_it_holds() {
    assert_eq!(generate(&[0], 3, 5, &[2]), (2, "000".to_owned(), true));
}
