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
use neura::generate::{Family, Generator, Step};
use neura::gguf::GgufFile;
use neura::template::{Message, Template, Value};
use neura::{Element, MemoryRequest, Runtime, RuntimeRequest, Shape};
use std::path::Path;
use tokenizer::Tokenizer;

const CHUNK: u32 = 8;
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
    let chunk = CHUNK.min(context);

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
    let family = match std::env::var("NEURA_TUNE") {
        Ok(_) => Family::tuned(&runtime, loaded, chunk, built),
        Err(_) => Family::of(&runtime, loaded, chunk, built),
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
