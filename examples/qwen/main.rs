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
use neura::decoder::Decoder;
use neura::generate::{Generator, Phase};
use neura::gguf::GgufFile;
use neura::template::{Message, Template, Value};
use neura::{Element, Graph, MemoryRequest, Runtime, RuntimeRequest, Shape};
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

    let prefill = Graph::new();
    let prefill_model = Model::new(&prefill, &source, &config, context, chunk);
    let prefill_token = prefill.input(Shape::of([1, 1, chunk, 1]), Element::Single);
    let prefill_cursor = prefill.input(Shape::scalar(), Element::Single);
    let prefill_logits = prefill_model.step(&prefill, prefill_token, prefill_cursor);
    let prefill_decoder = Decoder::greedy(
        &prefill,
        prefill_token,
        prefill_cursor,
        prefill_logits,
        tokenizer.end_of_turn(),
    );
    let weights = runtime.load(&prefill, &source);
    let prefill = runtime.compile(&prefill, &weights);

    let decode = Graph::new();
    let decode_model = Model::new(&decode, &source, &config, context, 1);
    let decode_token = decode.input(Shape::of([1, 1, 1, 1]), Element::Single);
    let decode_cursor = decode.input(Shape::scalar(), Element::Single);
    let decode_logits = decode_model.step(&decode, decode_token, decode_cursor);
    let decode_decoder = Decoder::greedy(
        &decode,
        decode_token,
        decode_cursor,
        decode_logits,
        tokenizer.end_of_turn(),
    );
    runtime.rebind(&weights, &decode);
    let decode = runtime.compile(&decode, &weights);

    println!(
        "plan: {:.1}MB of tensors, {:.1}MB of device memory",
        prefill.tensor_bytes() as f64 / (1 << 20) as f64,
        prefill.device_bytes() as f64 / (1 << 20) as f64,
    );

    let mut generator = Generator::new(
        runtime,
        Phase::new(prefill, prefill_decoder),
        Phase::new(decode, decode_decoder),
        tokenizer.engine(),
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
