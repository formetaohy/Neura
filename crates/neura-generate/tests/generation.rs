use neura_abi::Element;
use neura_decoder::Decoder;
use neura_generate::Generator;
use neura_gguf::Gguf;
use neura_graph::{Graph, Shape};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};
use neura_tokenizer::{Rule, Tokenizer, Vocabulary};

#[path = "support/mod.rs"]
mod support;

use support::{Meta, build};

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

fn generate(context: u32, wanted: usize) -> (usize, String, bool) {
    let graph = Graph::new();
    let token = graph.input(Shape::matrix(1, 1), Element::Single);
    let position = graph.input(Shape::matrix(1, 1), Element::Single);
    let logits = graph.one_hot(position, 8);
    let decoder = Decoder::greedy(&graph, token, position, logits, [], 3);
    let runtime = runtime();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let tokenizer = tokenizer();
    let mut generator = Generator::new(runtime, program, decoder, &tokenizer, context);
    let mut text = String::new();
    let report = generator.run(&[0], wanted, |held| text.push_str(held));
    (report.produced(), text, report.reached_context())
}

#[test]
fn a_generator_walks_a_prompt_into_the_tokens_that_follow() {
    assert_eq!(generate(16, 5), (3, "0012".to_owned(), false));
}

#[test]
fn a_generator_stops_at_the_token_count_it_was_asked_for() {
    assert_eq!(generate(16, 2), (2, "001".to_owned(), false));
}

#[test]
fn a_generator_stops_at_the_context_it_holds() {
    assert_eq!(generate(3, 5), (2, "001".to_owned(), true));
}
