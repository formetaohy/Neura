mod attention;
mod config;
mod delta;
mod mlp;
mod model;
mod ops;
mod text;
mod tokenizer;

use config::Config;
use model::{Model, Planes};
use neura::generate::{Batch, BatchFill, BatchSeed, BatchShape, BatchSpec, BatchStep};
use neura::gguf::GgufFile;
use neura::template::{Message, Template, Value};
use neura::tokenizer::Text;
use neura::{Element, MemoryRequest, Runtime, RuntimeRequest, Shape};
use std::path::Path;
use std::time::{Duration, Instant};
use tokenizer::Tokenizer;

const CHUNKS: &[u32] = &[32, 8];
const CONTEXT: u32 = 1024;
const REPLY: usize = 64;
const READBACK: u64 = 8 << 20;
const HEAP: u64 = 2 << 30;

#[derive(Clone, Copy)]
struct Character {
    name: &'static str,
    system: &'static str,
}

const CHARACTERS: [Character; 2] = [
    Character {
        name: "Ada",
        system: "You are Ada, a cheerful astronomer who loves bold theories about the night sky. You are talking with Ben, a dry-witted historian who trusts primary sources. Stay in character as Ada. Reply with one or two short sentences, write only your own line without your name, and bring a new question or observation instead of repeating an earlier line.",
    },
    Character {
        name: "Ben",
        system: "You are Ben, a dry-witted historian who trusts primary sources and dislikes speculation. You are talking with Ada, a cheerful astronomer who loves bold theories about the night sky. Stay in character as Ben. Reply with one or two short sentences, write only your own line without your name, and bring a new question or observation instead of repeating an earlier line.",
    },
];

struct Speaker {
    character: Character,
    turns: Vec<(&'static str, String)>,
    slot: Option<u32>,
}

impl Speaker {
    fn new(character: Character) -> Self {
        Self {
            character,
            turns: vec![("system", character.system.to_owned())],
            slot: None,
        }
    }
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let path = arguments.next().unwrap_or_else(|| {
        panic!("usage: qwen <model.gguf> <turns>, holding one Qwen3.5 gguf and a turn count")
    });
    let turns = arguments
        .next()
        .unwrap_or_else(|| panic!("usage: qwen <model.gguf> <turns>"))
        .parse::<usize>()
        .expect("a turn count is a whole number");
    assert!(
        arguments.next().is_none(),
        "usage: qwen <model.gguf> <turns>",
    );
    assert!(turns > 0, "a conversation answers at least one turn");

    let source = GgufFile::open(Path::new(&path));
    let tokenizer = Tokenizer::read(&source);
    let config = Config::read(source.metadata(), tokenizer.vocabulary());
    let template = Template::read(&source);
    let context = CONTEXT.min(config.context);
    let chunks = CHUNKS
        .iter()
        .copied()
        .filter(|chunk| *chunk < context)
        .collect::<Vec<u32>>();

    println!(
        "{}: {} layers, {} hidden, {} context",
        path, config.layers, config.hidden, context,
    );

    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes: HEAP,
            readback_bytes: READBACK,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("a device that runs the model");
    let kv = Element::Single;
    let planes = |shape: BatchShape| Planes {
        capacity: shape.capacity,
        window: shape.window,
        batch: shape.batch,
        rows: shape.rows,
        kv,
    };
    let answer = |graph: &neura::Graph, shape: BatchShape| {
        let token = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
        let cursor = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let mask = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
        let reset = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, planes(shape));
        BatchStep {
            token,
            cursor,
            mask,
            reset,
            logits: model.step(graph, token, cursor, mask),
            states: Vec::new(),
        }
    };
    let fill = |graph: &neura::Graph, shape: BatchShape| {
        let token = graph.input(Shape::of([1, shape.rows, shape.batch, 1]), Element::Single);
        let cursor = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let mask = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, planes(shape));
        model.prefill(graph, token, cursor, mask);
        BatchFill {
            token,
            cursor,
            mask,
            states: Vec::new(),
        }
    };
    let reset = |graph: &neura::Graph, shape: BatchShape| {
        let reset = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let share = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, planes(shape));
        model.reset(graph, reset);
        BatchSeed {
            reset,
            share,
            states: Vec::new(),
        }
    };
    let frame = std::env::var("NEURA_FRAME_US").ok().map(|held| {
        let micros = held.parse::<u64>().unwrap_or_else(|_| {
            panic!("NEURA_FRAME_US holds whole microseconds, and {held} is no such count")
        });
        assert!(micros > 0, "NEURA_FRAME_US walks a frame of no microsecond");
        Duration::from_micros(micros)
    });
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.load(graph, &source),
        BatchSpec::new(context, 2, &chunks),
        answer,
        fill,
        reset,
    );

    let mut speakers = CHARACTERS
        .into_iter()
        .map(Speaker::new)
        .collect::<Vec<Speaker>>();
    let started = Instant::now();
    let mut generated = 0usize;
    for turn in 0..turns {
        let current = turn % 2;
        let other = 1 - current;
        let heard = speakers[other]
            .turns
            .iter()
            .rev()
            .find(|(role, _)| *role == "assistant")
            .map(|(_, content)| content.trim().to_owned());
        let prompt = match heard.as_deref() {
            Some(heard) if !heard.is_empty() => format!(
                "{} says: {}\nReply to {}. Write only your own next line, without your name.",
                speakers[other].character.name, heard, speakers[other].character.name,
            ),
            _ => format!(
                "Start the conversation with {}. Write only your first line, without your name.",
                speakers[other].character.name,
            ),
        };
        speakers[current].turns.push(("user", prompt.clone()));
        let slot = match speakers[current].slot {
            None => {
                let tokens = encode(&template, &tokenizer, &speakers[current].turns);
                assert!(
                    tokens.len() < context as usize,
                    "the first turn holds {} tokens where this run holds {context} of context",
                    tokens.len(),
                );
                let slot = batch.admit(&runtime, &tokens);
                speakers[current].slot = Some(slot);
                slot
            }
            Some(slot) => {
                let delta = encode_turn(&template, &tokenizer, &prompt);
                let room = batch
                    .remaining(slot)
                    .expect("a held conversation plane holds a cursor");
                if delta.len() + REPLY < room as usize {
                    batch.resume(slot, &delta);
                } else {
                    trim(&mut speakers[current].turns, context, &template, &tokenizer);
                    let tokens = encode(&template, &tokenizer, &speakers[current].turns);
                    batch.restart(&runtime, slot, &tokens);
                }
                slot
            }
        };
        let (reply, tokens, ended) = reply(
            &mut batch,
            &runtime,
            &tokenizer,
            slot,
            context,
            speakers[current].character.name,
            frame,
        );
        generated += tokens;
        speakers[current].turns.push(("assistant", reply));
        if ended {
            batch.suspend(slot);
        } else {
            batch.finish(slot);
            speakers[current].slot = None;
        }
    }

    for slot in batch.occupied().collect::<Vec<u32>>() {
        batch.finish(slot);
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "\n{turns} turns in {seconds:.2}s ({:.1} tokens/s)",
        generated as f64 / seconds,
    );
}

fn reply(
    batch: &mut Batch,
    runtime: &Runtime,
    tokenizer: &Tokenizer,
    slot: u32,
    context: u32,
    name: &str,
    frame: Option<Duration>,
) -> (String, usize, bool) {
    let end = tokenizer.end_of_turn();
    let mut text = Text::new();
    let mut reply = String::new();
    let mut generated = 0usize;
    let mut ended = false;
    let mut worst_host = Duration::ZERO;
    let mut worst_slice = 0.0f64;
    print!("{name}: ");
    while generated < REPLY {
        let emitted = match frame {
            Some(budget) => {
                let started = Instant::now();
                let emitted = batch.advance_within(runtime, budget);
                worst_host = worst_host.max(started.elapsed());
                worst_slice = worst_slice.max(batch.last_slice_seconds().unwrap_or(0.0));
                if emitted.is_empty() {
                    std::thread::yield_now();
                }
                emitted
            }
            None => batch.step(runtime),
        };
        for held in emitted {
            if held.slot != slot {
                continue;
            }
            generated += 1;
            if held.token == end {
                ended = true;
                continue;
            }
            let piece = tokenizer.engine().piece(held.token);
            let out = text.push(&piece);
            reply.push_str(&out);
            print!("{out}");
        }
        if ended {
            break;
        }
        if batch.cursor(slot).is_some_and(|cursor| cursor >= context) {
            break;
        }
    }
    let tail = text.flush();
    reply.push_str(&tail);
    print!("{tail}");
    println!();
    if let Some(budget) = frame {
        println!(
            "  frame {:.2}ms worst host {:.2}ms worst slice {:.2}ms",
            budget.as_secs_f64() * 1e3,
            worst_host.as_secs_f64() * 1e3,
            worst_slice * 1e3,
        );
    }
    (reply, generated, ended)
}

fn trim(
    turns: &mut Vec<(&'static str, String)>,
    context: u32,
    template: &Template,
    tokenizer: &Tokenizer,
) {
    let mut tokens = encode(template, tokenizer, turns);
    while tokens.len() + REPLY >= context as usize && turns.len() > 2 {
        turns.remove(1);
        if turns.get(1).is_some_and(|(role, _)| *role == "assistant") {
            turns.remove(1);
        }
        tokens = encode(template, tokenizer, turns);
    }
    assert!(
        tokens.len() + 1 < context as usize,
        "a shortened turn holds {} tokens where this run holds {context} of context",
        tokens.len(),
    );
}

fn encode(
    template: &Template,
    tokenizer: &Tokenizer,
    turns: &[(&'static str, String)],
) -> Vec<u32> {
    let messages = turns
        .iter()
        .map(|(role, content)| Message::new(*role, content.as_str()))
        .collect::<Vec<Message>>();
    let rendered = template.render(&messages, &flags());
    tokenizer.encode(&rendered)
}

fn encode_turn(template: &Template, tokenizer: &Tokenizer, prompt: &str) -> Vec<u32> {
    let rendered = template.render(&[Message::new("user", prompt)], &flags());
    tokenizer.encode(&format!("\n{rendered}"))
}

fn flags() -> [(&'static str, Value); 2] {
    [
        ("add_generation_prompt", Value::Flag(true)),
        ("enable_thinking", Value::Flag(false)),
    ]
}
