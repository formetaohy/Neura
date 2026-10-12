mod attention;
mod chat;
mod config;
mod delta;
mod mlp;
mod model;
mod ops;
mod tokenizer;
mod unicode;

use config::Config;
use model::Model;
use neura::gguf::GgufFile;
use neura::{Element, Graph, MemoryRequest, Runtime, RuntimeRequest, Shape};
use std::path::Path;
use std::time::Instant;
use tokenizer::Tokenizer;

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
    let tokenizer = Tokenizer::read(source.metadata());
    let config = Config::read(source.metadata(), tokenizer.vocabulary());
    let context = CONTEXT.min(config.context);
    println!(
        "{}: {} layers, {} hidden, {} heads of {}, {} context",
        path, config.layers, config.hidden, config.heads, config.head_width, context,
    );

    let prompt = match prompt.strip_prefix("raw:") {
        Some(held) => held.to_owned(),
        None => chat::render(&prompt),
    };
    let tokens = tokenizer.encode(&prompt);
    assert!(
        !tokens.is_empty(),
        "the prompt {prompt:?} holds no token this tokenizer knows",
    );
    assert!(
        tokens.len() < context as usize,
        "the prompt holds {} tokens where this run holds {context} of context",
        tokens.len(),
    );
    println!("prompt: {prompt:?}");
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
    let graph = Graph::new();
    let model = Model::new(&graph, &source, &config, context);
    let token = graph.input(Shape::matrix(1, 1), Element::Single);
    let position = graph.input(Shape::matrix(1, 1), Element::Single);
    let logits = model.step(&graph, token, position);
    graph.retain(logits);
    let weights = runtime.load(&graph, &source);
    let program = runtime.compile(&graph, &weights);
    println!(
        "plan: {:.1}MB of tensors, {:.1}MB of device memory",
        program.tensor_bytes() as f64 / (1 << 20) as f64,
        program.device_bytes() as f64 / (1 << 20) as f64,
    );
    for state in model.states() {
        let elements = graph.shape(state).elements();
        runtime.write(&program, state, &vec![0.0f32; elements as usize]);
    }

    let started = Instant::now();
    let mut text = Text::new();
    let mut at = 0u32;
    for id in tokens.iter().copied() {
        runtime.write(&program, token, &[id as f32]);
        runtime.write(&program, position, &[at as f32]);
        runtime.run(&program);
        text.push(&tokenizer.piece(id));
        at += 1;
    }

    let mut next = choose(&runtime.read(&program, logits));
    let mut produced = 0usize;
    while produced < wanted && next != config.end && at < context {
        runtime.write(&program, token, &[next as f32]);
        runtime.write(&program, position, &[at as f32]);
        runtime.run(&program);
        text.push(&tokenizer.piece(next));
        at += 1;
        produced += 1;
        next = choose(&runtime.read(&program, logits));
    }
    text.finish();
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "\n{produced} tokens in {elapsed:.2}s ({:.1} tokens/s)",
        produced as f64 / elapsed,
    );
    if at >= context {
        println!("(the model reached its {context} token context)");
    }
}

fn choose(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    for (at, value) in logits.iter().enumerate() {
        if *value > logits[best] {
            best = at;
        }
    }
    best as u32
}

fn emit(held: &mut Vec<u8>) {
    loop {
        match std::str::from_utf8(held) {
            Ok(_) => break,
            Err(error) => {
                let valid = error.valid_up_to();
                if valid > 0 {
                    print!(
                        "{}",
                        std::str::from_utf8(&held[..valid]).expect("a valid prefix reads as text"),
                    );
                    held.drain(..valid);
                    continue;
                }
                match error.error_len() {
                    Some(width) => {
                        print!("{}", String::from_utf8_lossy(&held[..width]));
                        held.drain(..width);
                    }
                    None => break,
                }
            }
        }
    }
    if let Ok(text) = std::str::from_utf8(held) {
        print!("{text}");
        held.clear();
    }
}

struct Text {
    held: Vec<u8>,
}

impl Text {
    fn new() -> Self {
        Self { held: Vec::new() }
    }

    fn push(&mut self, piece: &[u8]) {
        self.held.extend_from_slice(piece);
        emit(&mut self.held);
    }

    fn finish(&mut self) {
        if !self.held.is_empty() {
            print!("{}", String::from_utf8_lossy(&self.held));
            self.held.clear();
        }
    }
}
