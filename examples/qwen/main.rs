mod config;
mod model;
mod tokenizer;

use config::Config;
use model::Model;
use neura::gguf::GgufFile;
use neura::{Graph, MemoryRequest, Runtime, RuntimeRequest, Shape};
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

    let tokens = tokenizer.encode(&chat(&prompt));
    println!("prompt: {prompt}");
    println!("tokens: {} ", tokens.len());

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
    let token = graph.input(Shape::matrix(1, 1), neura::Element::Single);
    let position = graph.input(Shape::matrix(1, 1), neura::Element::Single);
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
    let mut held = Vec::new();
    let mut next = tokens[0];
    let mut produced = 0usize;
    for (at, id) in tokens.iter().enumerate() {
        let last = at + 1 == tokens.len();
        runtime.write(&program, token, &[*id as f32]);
        runtime.write(&program, position, &[at as f32]);
        runtime.run(&program);
        if last {
            next = choose(&runtime.read(&program, logits));
        }
        held.push(*id);
    }
    print!("{}", detach(&tokenizer, &held));

    while produced < wanted && next != config.eos {
        runtime.write(&program, token, &[next as f32]);
        runtime.write(&program, position, &[(tokens.len() + produced) as f32]);
        runtime.run(&program);
        held.push(next);
        print!("{}", detach(&tokenizer, &held[held.len() - 1..]));
        next = choose(&runtime.read(&program, logits));
        produced += 1;
        if tokens.len() + produced >= context as usize {
            println!("\n(the model reached its {context} token context)");
            break;
        }
    }
    println!(
        "\n{} tokens in {:.2}s ({:.1} tokens/s)",
        produced + 1,
        started.elapsed().as_secs_f64(),
        (produced + 1) as f64 / started.elapsed().as_secs_f64(),
    );
}

fn chat(prompt: &str) -> String {
    format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
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

fn detach(tokenizer: &Tokenizer, ids: &[u32]) -> String {
    let mut bytes = Vec::new();
    for id in ids {
        bytes.extend(tokenizer.piece(*id));
    }
    String::from_utf8_lossy(&bytes).into_owned()
}
