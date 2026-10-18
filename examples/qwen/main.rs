mod attention;
mod config;
mod delta;
mod mlp;
mod model;
mod ops;
mod text;
mod tokenizer;

use config::Config;
use model::Model;
use neura::generate::{Family, Generator, Sampling, Step};
use neura::gguf::GgufFile;
use neura::template::{Message, Template, Value};
use neura::{Element, MemoryRequest, Runtime, RuntimeRequest, Shape};
use std::path::Path;
use tokenizer::Tokenizer;

const CHUNKS: &[u32] = &[256, 128, 32, 8];
const CONTEXT: u32 = 1024;
const READBACK: u64 = 8 << 20;
const HEAP: u64 = 2 << 30;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments.next().unwrap_or_else(|| {
        panic!("usage: qwen <model.gguf> [prompt] [tokens], holding one Qwen3.5 gguf");
    });
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "What is the capital of France?".to_owned());
    let wanted = arguments
        .next()
        .map_or(48, |held| held.parse::<usize>().expect("a token count"));

    let source = GgufFile::open(Path::new(&path));
    let tokenizer = Tokenizer::read(&source);
    let config = Config::read(source.metadata(), tokenizer.vocabulary());
    let context = CONTEXT.min(config.context);
    println!(
        "{}: {} layers, {} hidden, {} heads of {}, {} context",
        path, config.layers, config.hidden, config.heads, config.head_width, context,
    );

    let rendered = match prompt.strip_prefix("raw:") {
        Some(held) => held.to_owned(),
        None => Template::read(&source).render(
            &[Message::new("user", prompt.as_str())],
            &[
                ("add_generation_prompt", Value::Flag(true)),
                ("enable_thinking", Value::Flag(false)),
            ],
        ),
    };
    let tokens = tokenizer.encode(&rendered);
    println!("prompt: {rendered:?}");
    println!("tokens: {}", tokens.len());

    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes: HEAP,
            readback_bytes: READBACK,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("a device that runs the model");
    let mut chunks = Vec::new();
    for chunk in CHUNKS.iter().copied() {
        let chunk = chunk.min(context);
        if chunks.last() != Some(&chunk) {
            chunks.push(chunk);
        }
    }

    let loaded = |graph: &neura::Graph| runtime.load(graph, &source);
    let built = |graph: &neura::Graph, rows: u32| {
        let token = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
        let cursor = graph.input(Shape::scalar(), Element::Single);
        let model = Model::new(graph, &source, &config, context, rows);
        Step {
            token,
            cursor,
            logits: model.step(graph, token, cursor),
        }
    };
    let sampling = sampling();
    if let Some(sampling) = sampling {
        println!("sampling: {sampling:?}");
    }
    let tuned = std::env::var("NEURA_TUNE").is_ok();
    let family = match (tuned, sampling) {
        (true, Some(sampling)) => Family::tuned_sampled(&runtime, loaded, &chunks, built, sampling),
        (true, None) => Family::tuned(&runtime, loaded, &chunks, built),
        (false, Some(sampling)) => Family::sampled(&runtime, loaded, &chunks, built, sampling),
        (false, None) => Family::of(&runtime, loaded, &chunks, built),
    };

    let mut generator = Generator::new(
        runtime,
        family,
        tokenizer.engine(),
        tokenizer.end_of_turn(),
        context,
    );
    let report = generator.run(&tokens, wanted, |held| print!("{held}"));
    println!(
        "
{} tokens in {:.2}s ({:.1} tokens/s)",
        report.produced(),
        report.seconds(),
        report.produced() as f64 / report.seconds(),
    );
    if report.reached_context() {
        println!("(the model reached its {context} token context)");
    }
}

fn sampling() -> Option<Sampling> {
    std::env::var("NEURA_SAMPLE").ok().map(|held| {
        let mut parts = held.split(',');
        let temperature = parts
            .next()
            .expect("a sample holds a temperature")
            .parse::<f32>()
            .expect("a sample temperature is a number");
        let keep = parts
            .next()
            .expect("a sample holds a candidate count")
            .parse::<u32>()
            .expect("a sample candidate count is a whole number");
        let cumulative = parts
            .next()
            .expect("a sample holds a cumulative mass")
            .parse::<f32>()
            .expect("a sample cumulative mass is a number");
        assert!(
            parts.next().is_none(),
            "a sample names a temperature, a candidate count and a cumulative mass, and {held} names more",
        );
        Sampling::sampled(temperature, keep, cumulative)
    })
}
