mod attention;
mod config;
mod delta;
mod mlp;
mod model;
mod ops;
mod persona;
mod text;
mod tokenizer;
mod train;

use config::Config;
use model::{Model, Planes};
use neura::generate::{
    Batch, BatchFill, BatchPrefix, BatchSeed, BatchShape, BatchSpec, BatchStep, Persona, Sampling,
};
use neura::gguf::GgufFile;
use neura::template::{Message, Template, Value};
use neura::tokenizer::Text;
use neura::{Element, MemoryRequest, Runtime, RuntimeRequest, Shape};
use std::path::{Path, PathBuf};
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
    let window = std::env::var("NEURA_WINDOW")
        .ok()
        .map(|held| held.parse::<u32>().expect("a window of tokens"));
    let context = std::env::var("NEURA_CONTEXT")
        .ok()
        .map(|held| held.parse::<u32>().expect("a context of tokens"))
        .unwrap_or(CONTEXT)
        .min(config.context);
    if let Some(window) = window {
        assert!(
            window <= context,
            "a window of {window} tokens over a context of {context}",
        );
    }
    println!(
        "{}: {} layers, {} hidden, {} heads of {}, {} context, {} window",
        path,
        config.layers,
        config.hidden,
        config.heads,
        config.head_width,
        context,
        window.map_or_else(|| "full".to_owned(), |held| held.to_string()),
    );

    let template = Template::read(&source);
    let training = std::env::var("NEURA_TRAIN").ok();
    let rendered = match &training {
        Some(_) => String::new(),
        None => match prompt.strip_prefix("raw:") {
            Some(held) => held.to_owned(),
            None => template.render(
                &[Message::new("user", prompt.as_str())],
                &[
                    ("add_generation_prompt", Value::Flag(true)),
                    ("enable_thinking", Value::Flag(false)),
                ],
            ),
        },
    };
    let tokens = match &training {
        Some(_) => Vec::new(),
        None => tokenizer.encode(&rendered),
    };
    if training.is_none() {
        println!("prompt: {rendered:?}");
        println!("tokens: {}", tokens.len());
    }
    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes: match training {
                Some(_) => 8 << 30,
                None => HEAP,
            },
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
    let shared = std::env::var("NEURA_PREFIX").is_ok();
    let frames = std::env::var("NEURA_FRAMES").ok().map(|held| {
        let waves = held.parse::<u32>().expect("a frame of waves");
        assert!(waves > 0, "a frame walks at least one wave");
        waves
    });
    let kv = match std::env::var("NEURA_KV").as_deref() {
        Ok("half") => Element::Half,
        Ok("single") | Err(_) => Element::Single,
        Ok(other) => panic!("a kv storage names half or single, and NEURA_KV names {other}"),
    };
    println!("kv: {}", kv.name());

    let table_path = std::env::var("NEURA_PERSONA").ok();
    let seed = std::env::var("NEURA_PERSONA_SEED")
        .ok()
        .map(|held| {
            held.parse::<u64>()
                .expect("a persona seed is a whole number")
        })
        .unwrap_or(1);
    let plan = std::env::var("NEURA_PERSONAS").ok().map(|held| {
        let count = held
            .parse::<u32>()
            .expect("a persona count is a whole number");
        let rank = std::env::var("NEURA_RANK")
            .ok()
            .map(|held| {
                held.parse::<u32>()
                    .expect("a persona rank is a whole number")
            })
            .unwrap_or(4);
        persona::Plan::new(count, rank)
    });
    let table = plan.map(|_| persona::Table::open(table_path.as_deref(), seed));
    if let Some(plan) = plan {
        println!(
            "personas: {} of rank {}{}",
            plan.count(),
            plan.rank(),
            match table_path.as_deref() {
                None | Some("random") => format!(", initialized from seed {seed}"),
                Some(path) => format!(", loaded from {path}"),
            },
        );
    }
    if let Some(plan) = plan {
        assert!(
            !shared,
            "a shared prefix fills the states of one persona, and NEURA_PREFIX shares them across {} personas",
            plan.count(),
        );
    }
    if let Some(corpus) = training {
        let plan = plan.expect("a training run names NEURA_PERSONAS");
        let request = train::Request {
            corpus: std::fs::read_to_string(&corpus).unwrap_or_else(|error| {
                panic!("a training corpus at {corpus} could not be read: {error}")
            }),
            rows: std::env::var("NEURA_ROWS")
                .ok()
                .map(|held| held.parse::<u32>().expect("a training window of rows"))
                .unwrap_or(16),
            steps: std::env::var("NEURA_STEPS")
                .ok()
                .map(|held| held.parse::<u32>().expect("a training step count"))
                .unwrap_or(16),
            rate: std::env::var("NEURA_RATE")
                .ok()
                .map(|held| held.parse::<f32>().expect("a training rate"))
                .unwrap_or(1e-3),
            seed,
            persona: std::env::var("NEURA_PERSONA_INDEX")
                .ok()
                .map(|held| held.parse::<u32>().expect("a persona index"))
                .unwrap_or(0),
            init: table_path.clone(),
            save: PathBuf::from(
                std::env::var("NEURA_SAVE").unwrap_or_else(|_| "persona.neura".to_owned()),
            ),
        };
        train::run(&runtime, &source, &tokenizer, &config, plan, &request);
        return;
    }
    let loaded = |graph: &neura::Graph| persona::load(&runtime, graph, &source, table.as_ref());
    let planes = |shape: BatchShape, prefix: bool| Planes {
        capacity: shape.capacity,
        window: shape.window,
        bounded: shape.bounded,
        batch: shape.batch,
        rows: shape.rows,
        shared: shape.shared,
        prefix,
        kv,
        persona: plan,
        training: false,
        training_persona: 0,
    };
    let answer = |graph: &neura::Graph, shape: BatchShape| {
        let token = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
        let cursor = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let mask = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
        let reset = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, planes(shape, false));
        BatchStep {
            token,
            cursor,
            mask,
            reset,
            logits: model.step(graph, token, cursor, mask),
            persona: model.persona().map(|value| Persona {
                value,
                count: plan.expect("a routed model holds its persona plan").count(),
            }),
        }
    };
    let fill = |graph: &neura::Graph, shape: BatchShape| {
        let token = graph.input(Shape::of([1, shape.rows, shape.batch, 1]), Element::Single);
        let cursor = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let mask = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, planes(shape, false));
        model.prefill(graph, token, cursor, mask);
        BatchFill {
            token,
            cursor,
            mask,
        }
    };
    let reset = |graph: &neura::Graph, shape: BatchShape| {
        let reset = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let share = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, planes(shape, false));
        model.reset(graph, reset, shape.shared.then_some(share));
        BatchSeed { reset, share }
    };
    let prefix = |graph: &neura::Graph, shape: BatchShape| {
        let token = graph.input(Shape::of([1, shape.rows, 1, 1]), Element::Single);
        let cursor = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
        let mask = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
        let reset = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
        let model = Model::new(graph, &source, &config, planes(shape, true));
        model.clear_shared(graph, reset);
        model.prefill(graph, token, cursor, mask);
        BatchPrefix {
            token,
            cursor,
            mask,
            reset,
        }
    };
    let chunks = CHUNKS
        .iter()
        .copied()
        .filter(|chunk| *chunk < context && window.is_none_or(|window| *chunk <= window))
        .collect::<Vec<u32>>();
    let mut spec = BatchSpec::new(context, sequences, &chunks);
    if let Some(window) = window {
        spec = spec.window(window);
    }
    if let Some(sampling) = sampling {
        spec = spec.sampled(sampling);
    }
    if tuned {
        spec = spec.tuned();
    }
    if shared {
        spec = spec.shared();
    }
    if let Some(plan) = plan {
        spec = spec.personas(plan.count());
    }
    let mut batch = Batch::new(&runtime, loaded, spec, answer, fill, reset);
    if let Some(plan) = plan {
        let held = personas(sequences, plan.count());
        for (slot, chosen) in held.iter().copied().enumerate() {
            batch.set_persona_of(&runtime, slot as u32, chosen);
        }
        println!("persona of each plane: {held:?}");
    }
    if sampling.is_some()
        && let Some(temperature) = std::env::var("NEURA_TEMPERATURE")
            .ok()
            .map(|held| held.parse::<f32>().expect("a temperature is a number"))
    {
        batch.set_temperature(&runtime, temperature);
    }
    match turns() {
        Some(prompts) => {
            assert!(
                !shared,
                "a multi-turn conversation keeps the state of each turn, and a shared prefix fills one prompt",
            );
            assert!(
                frames.is_none(),
                "a multi-turn conversation answers each turn in one step, and NEURA_FRAMES walks a single prompt",
            );
            assert_eq!(
                sequences, 1,
                "a multi-turn conversation answers one sequence at a time",
            );
            converse(
                &runtime, &tokenizer, &template, &mut batch, &prompts, wanted,
            );
        }
        None => {
            if shared && tokens.len() > 1 {
                batch.prefix(&runtime, &tokens[..tokens.len() - 1], prefix);
            }
            run(&runtime, &tokenizer, &mut batch, &tokens, wanted, frames);
        }
    }
}

fn run(
    runtime: &Runtime,
    tokenizer: &Tokenizer,
    batch: &mut Batch,
    tokens: &[u32],
    wanted: usize,
    frames: Option<u32>,
) {
    let started = Instant::now();
    let context = batch.context();
    for _ in 0..batch.batch() {
        batch.admit(runtime, tokens);
    }
    let mut text = Text::new();
    let end = tokenizer.end_of_turn();
    let mut produced = 0usize;
    let mut reached_context = false;
    let mut walked = 0u32;
    let mut first = None;
    let mut longest = 0.0f64;
    while produced < wanted && batch.active().next().is_some() {
        let started_frame = Instant::now();
        let emitted = match frames {
            Some(waves) => batch.advance(runtime, waves),
            None => batch.step(runtime),
        };
        if frames.is_some() {
            let elapsed = started_frame.elapsed().as_secs_f64();
            match first {
                None => first = Some(elapsed),
                Some(_) => longest = longest.max(elapsed),
            }
            walked += 1;
        }
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
    if let Some(first) = first {
        println!(
            "({walked} frames, first {:.2} ms, longest {:.2} ms)",
            first * 1e3,
            longest * 1e3,
        );
    }
}

fn converse(
    runtime: &Runtime,
    tokenizer: &Tokenizer,
    template: &Template,
    batch: &mut Batch,
    prompts: &[String],
    wanted: usize,
) {
    assert!(
        wanted > 0,
        "a conversation answers at least one token a turn"
    );
    assert!(
        !prompts.is_empty(),
        "a conversation holds at least one turn"
    );
    assert_eq!(
        batch.batch(),
        1,
        "a conversation answers one sequence at a time",
    );
    let started = Instant::now();
    let context = batch.context();
    let end = tokenizer.end_of_turn();
    let mut messages = Vec::new();
    let mut slot = None;
    let mut produced = 0usize;
    let mut reached_context = false;
    for (turn, prompt) in prompts.iter().enumerate() {
        assert!(
            !prompt.is_empty(),
            "turn {} of a conversation holds no prompt",
            turn + 1,
        );
        messages.push(Message::new("user", prompt.as_str()));
        match slot {
            None => {
                let tokens = encode(template, tokenizer, &messages);
                assert!(
                    tokens.len() < context as usize,
                    "the first turn holds {} tokens where this run holds {context} of context",
                    tokens.len(),
                );
                println!("turn {}: admits {} tokens", turn + 1, tokens.len());
                slot = Some(batch.admit(runtime, &tokens));
            }
            Some(held) => {
                let delta = encode_turn(template, tokenizer, prompt);
                let room = batch
                    .remaining(held)
                    .expect("a held conversation plane holds a cursor");
                if delta.len() + wanted < room as usize {
                    println!(
                        "turn {}: reuses {} tokens, prefills {}",
                        turn + 1,
                        context - room,
                        delta.len(),
                    );
                    batch.resume(held, &delta);
                } else {
                    let mut tokens = encode(template, tokenizer, &messages);
                    while tokens.len() + wanted >= context as usize && messages.len() > 1 {
                        messages.remove(0);
                        if messages
                            .first()
                            .is_some_and(|message| message.role() == "assistant")
                        {
                            messages.remove(0);
                        }
                        tokens = encode(template, tokenizer, &messages);
                    }
                    assert!(
                        tokens.len() + 1 < context as usize,
                        "turn {} holds {} tokens where this run holds {context} of context, and no earlier turn is left to drop",
                        turn + 1,
                        tokens.len(),
                    );
                    println!("turn {}: resets, prefills {}", turn + 1, tokens.len());
                    batch.restart(runtime, held, &tokens);
                }
            }
        }
        print!("turn {}: ", turn + 1);
        let mut text = Text::new();
        let mut assistant = String::new();
        let mut generated = Vec::new();
        loop {
            let emitted = batch.step(runtime);
            let mut ended = false;
            for held in emitted {
                generated.push(held.token);
                if held.token == end {
                    ended = true;
                    continue;
                }
                let piece = tokenizer.engine().piece(held.token);
                let out = text.push(&piece);
                assistant.push_str(&out);
                if !out.is_empty() {
                    print!("{out}");
                }
            }
            let cursor = batch
                .cursor(slot.expect("a conversation turn holds a plane"))
                .expect("a held conversation plane holds a cursor");
            if ended || generated.len() >= wanted || cursor >= context {
                reached_context |= cursor >= context;
                break;
            }
        }
        let tail = text.flush();
        assistant.push_str(&tail);
        if !tail.is_empty() {
            print!("{tail}");
        }
        messages.push(Message::new("assistant", assistant));
        batch.suspend(slot.expect("a conversation turn holds a plane"));
        produced += generated.len();
        println!("\nturn {} answered {} tokens", turn + 1, generated.len());
    }
    if let Some(slot) = slot {
        batch.finish(slot);
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "\n{} turns of {} tokens in {:.2}s ({:.1} tokens/s)",
        prompts.len(),
        produced,
        seconds,
        produced as f64 / seconds,
    );
    if reached_context {
        println!("(the model reached its {context} token context)");
    }
}

fn encode_turn(template: &Template, tokenizer: &Tokenizer, prompt: &str) -> Vec<u32> {
    let rendered = template.render(
        &[Message::new("user", prompt)],
        &[
            ("add_generation_prompt", Value::Flag(true)),
            ("enable_thinking", Value::Flag(false)),
        ],
    );
    tokenizer.encode(&format!("\n{rendered}"))
}

fn encode(template: &Template, tokenizer: &Tokenizer, messages: &[Message]) -> Vec<u32> {
    let rendered = template.render(
        messages,
        &[
            ("add_generation_prompt", Value::Flag(true)),
            ("enable_thinking", Value::Flag(false)),
        ],
    );
    tokenizer.encode(&rendered)
}

fn personas(sequences: u32, count: u32) -> Vec<u32> {
    match std::env::var("NEURA_PERSONA_OF") {
        Ok(held) => {
            let chosen = held
                .split(',')
                .map(|part| {
                    part.parse::<u32>()
                        .expect("a persona of a plane is a whole number")
                })
                .collect::<Vec<u32>>();
            assert_eq!(
                chosen.len(),
                sequences as usize,
                "NEURA_PERSONA_OF names {} personas for a run of {sequences} planes",
                chosen.len(),
            );
            for persona in &chosen {
                assert!(
                    *persona < count,
                    "a run of {count} personas names no persona {persona}",
                );
            }
            chosen
        }
        Err(_) => (0..sequences).map(|slot| slot % count).collect(),
    }
}

fn turns() -> Option<Vec<String>> {
    std::env::var("NEURA_TURNS")
        .ok()
        .map(|held| held.split('|').map(str::to_owned).collect())
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
