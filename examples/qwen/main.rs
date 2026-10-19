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
use neura::generate::{Batch, BatchFill, BatchSpec, BatchStep, Sampling};
use neura::gguf::GgufFile;
use neura::template::{Message, Template, Value};
use neura::tokenizer::Text;
use neura::{Element, MemoryRequest, Runtime, RuntimeRequest, Shape};
use std::path::Path;
use std::time::Instant;
use tokenizer::Tokenizer;

const CHUNKS: &[u32] = &[32, 8];
const CONTEXT: u32 = 1024;
const READBACK: u64 = 8 << 20;
const HEAP: u64 = 2 << 30;

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments.next().unwrap_or_else(|| {
        panic!("usage: qwen <model.gguf> [prompt] [tokens] [sequences], holding one Qwen3.5 gguf");
    });
    let prompt = arguments
        .next()
        .unwrap_or_else(|| "What is the capital of France?".to_owned());
    let wanted = arguments
        .next()
        .map_or(48, |held| held.parse::<usize>().expect("a token count"));
    let sequences = arguments
        .next()
        .map_or(1, |held| held.parse::<u32>().expect("a sequence count"));
    assert!(sequences > 0, "a run answers at least one sequence");

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
    let sampling = sampling();
    if let Some(sampling) = sampling {
        println!("sampling: {sampling:?}");
    }
    let tuned = std::env::var("NEURA_TUNE").is_ok();

    let loaded = |graph: &neura::Graph| runtime.load(graph, &source);
    let answer = |graph: &neura::Graph, batch: u32, capacity: u32| {
        let token = graph.input(Shape::of([1, 1, batch, 1]), Element::Single);
        let cursor = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
        let mask = graph.input(Shape::of([1, 1, batch, 1]), Element::Single);
        let reset = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, capacity, batch, 1);
        BatchStep {
            token,
            cursor,
            mask,
            reset,
            logits: model.step(graph, token, cursor, mask),
        }
    };
    let fill = |graph: &neura::Graph, rows: u32, batch: u32, capacity: u32| {
        let token = graph.input(Shape::of([1, rows, batch, 1]), Element::Single);
        let cursor = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
        let mask = graph.input(Shape::of([1, 1, batch, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, capacity, batch, rows);
        model.prefill(graph, token, cursor, mask);
        BatchFill {
            token,
            cursor,
            mask,
        }
    };
    let reset = |graph: &neura::Graph, batch: u32, capacity: u32| {
        let reset = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, capacity, batch, 1);
        model.reset(graph, reset);
        reset
    };
    let chunks = CHUNKS
        .iter()
        .copied()
        .filter(|chunk| *chunk < context)
        .collect::<Vec<u32>>();
    let mut spec = BatchSpec::new(context, sequences, &chunks);
    if let Some(sampling) = sampling {
        spec = spec.sampled(sampling);
    }
    if tuned {
        spec = spec.tuned();
    }
    let mut batch = Batch::new(&runtime, loaded, spec, answer, fill, reset);
    run(&runtime, &tokenizer, &mut batch, &tokens, wanted);
}

fn run(runtime: &Runtime, tokenizer: &Tokenizer, batch: &mut Batch, tokens: &[u32], wanted: usize) {
    let started = Instant::now();
    let context = batch.context();
    for _ in 0..batch.batch() {
        batch.admit(runtime, tokens);
    }
    let mut text = Text::new();
    let end = tokenizer.end_of_turn();
    let mut produced = 0usize;
    let mut reached_context = false;
    while produced < wanted && batch.active().next().is_some() {
        let emitted = batch.step(runtime);
        for held in emitted {
            let reached = batch
                .cursor(held.slot)
                .is_some_and(|cursor| cursor >= context);
            if held.token == end || produced >= wanted || reached {
                batch.finish(held.slot);
                reached_context |= held.slot == 0 && reached;
                continue;
            }
            if held.slot == 0 {
                let piece = tokenizer.engine().piece(held.token);
                let text = text.push(&piece);
                if !text.is_empty() {
                    print!("{text}");
                }
                produced += 1;
            }
        }
    }
    for slot in batch.active().collect::<Vec<u32>>() {
        batch.finish(slot);
    }
    let tail = text.flush();
    if !tail.is_empty() {
        print!("{tail}");
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "
{} sequences of {} tokens in {:.2}s ({:.1} tokens/s)",
        batch.batch(),
        produced,
        seconds,
        (produced as f64 * batch.batch() as f64) / seconds,
    );
    if reached_context {
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
