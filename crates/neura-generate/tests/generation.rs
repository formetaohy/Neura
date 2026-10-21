use neura_abi::Element;
use neura_decoder::{Cache, CacheShape};
use neura_generate::{
    Batch, BatchFill, BatchPrefix, BatchSeed, BatchShape, BatchSnapshot, BatchSpec, BatchStep,
    Family, Generator, PlaneState, Sampling, Step,
};
use neura_gguf::Gguf;
use neura_graph::{Init, Shape};
use neura_runtime::{CheckpointFile, MemoryRequest, Runtime, RuntimeRequest};
use neura_tokenizer::{Rule, Tokenizer, Vocabulary};
use std::collections::BTreeSet;
use std::time::Duration;

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

fn family(runtime: &Runtime, chunks: &[u32], sampling: Option<Sampling>) -> Family {
    let build = |graph: &neura_graph::Graph, rows| {
        let token = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
        let cursor = graph.input(Shape::scalar(), Element::Single);
        let cache = Cache::new(
            graph,
            "cache",
            CacheShape::new(1, CLASSES, CAPACITY, CAPACITY, 1, rows),
            Element::Single,
        );
        let held = graph.one_hot(token, CLASSES);
        cache.write(graph, cursor, held, held);
        Step {
            token,
            cursor,
            logits: graph.sum_axis(cache.states()[0], 2),
        }
    };
    match sampling {
        Some(sampling) => Family::sampled(
            runtime,
            |graph| runtime.weights(graph),
            chunks,
            build,
            sampling,
        ),
        None => Family::of(runtime, |graph| runtime.weights(graph), chunks, build),
    }
}

fn generate(
    prompt: &[u32],
    context: u32,
    wanted: usize,
    chunks: &[u32],
    sampling: Option<Sampling>,
) -> (usize, String, bool) {
    let runtime = runtime();
    let family = family(&runtime, chunks, sampling);
    let tokenizer = tokenizer();
    let mut generator = Generator::new(runtime, family, &tokenizer, END, context);
    let mut text = String::new();
    let report = generator.run(prompt, wanted, |held| text.push_str(held));
    (report.produced(), text, report.reached_context())
}

#[test]
fn a_generator_prefills_a_prompt_in_the_chunks_its_model_takes() {
    let walked = generate(&[0, 1, 1, 2, 1], 16, 5, &[1], None);
    assert_eq!(walked, (5, "0112111111".to_owned(), false));
    for chunks in [&[2u32][..], &[3], &[4], &[8], &[4, 1], &[3, 2, 1]] {
        assert_eq!(
            generate(&[0, 1, 1, 2, 1], 16, 5, chunks, None),
            walked,
            "a prefill in the chunks {chunks:?} answers with the tokens a row at a time answers with",
        );
    }
}

#[test]
fn a_generator_stops_at_the_token_count_it_was_asked_for() {
    assert_eq!(
        generate(&[0, 1, 1, 2, 1], 16, 2, &[3], None),
        (2, "0112111".to_owned(), false),
    );
}

#[test]
fn a_generator_stops_at_the_context_it_holds() {
    assert_eq!(
        generate(&[0], 3, 5, &[2], None),
        (2, "000".to_owned(), true)
    );
}

#[test]
fn a_generator_samples_the_token_its_logits_and_seed_choose() {
    let walked = generate(
        &[0, 1, 1, 2, 1],
        16,
        5,
        &[3],
        Some(Sampling::sampled(1.0, 4, 1.0)),
    );
    assert_eq!(walked, (5, "0112101111".to_owned(), false));
}

fn cache(shape: BatchShape, graph: &neura_graph::Graph, rows: u32, shared: bool) -> Cache {
    match (shared, shape.bounded) {
        (false, false) => Cache::new(
            graph,
            "cache",
            CacheShape::new(1, CLASSES, shape.capacity, shape.window, shape.batch, rows),
            Element::Single,
        ),
        (false, true) => Cache::windowed(
            graph,
            "cache",
            CacheShape::new(1, CLASSES, shape.capacity, shape.window, shape.batch, rows),
            Element::Single,
        ),
        (true, false) => Cache::with_shared(
            graph,
            "cache",
            CacheShape::new(1, CLASSES, shape.capacity, shape.window, shape.batch, rows),
            Element::Single,
        ),
        (true, true) => Cache::windowed_with_shared(
            graph,
            "cache",
            CacheShape::new(1, CLASSES, shape.capacity, shape.window, shape.batch, rows),
            Element::Single,
        ),
    }
}

fn batch_answer(graph: &neura_graph::Graph, shape: BatchShape) -> BatchStep {
    let token = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
    let cursor = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
    let mask = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
    let reset = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
    let cache = cache(shape, graph, 1, shape.shared);
    let held = graph.mul(
        graph.permute(graph.one_hot(token, CLASSES), [0, 2, 1, 3]),
        graph.permute(mask, [0, 2, 1, 3]),
    );
    cache.write_masked(graph, cursor, held, held, graph.permute(mask, [0, 2, 1, 3]));
    BatchStep {
        token,
        cursor,
        mask,
        reset,
        logits: graph.reshape(
            graph.sum_axis(cache.states()[0], 2),
            Shape::of([1, 1, shape.batch, CLASSES]),
        ),
        states: Vec::new(),
    }
}

fn batch_fill(graph: &neura_graph::Graph, shape: BatchShape) -> BatchFill {
    let token = graph.input(Shape::of([1, shape.rows, shape.batch, 1]), Element::Single);
    let cursor = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
    let mask = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
    let cache = cache(shape, graph, shape.rows, shape.shared);
    let held = graph.mul(
        graph.permute(graph.one_hot(token, CLASSES), [0, 2, 1, 3]),
        graph.permute(mask, [0, 2, 1, 3]),
    );
    cache.write_masked(graph, cursor, held, held, graph.permute(mask, [0, 2, 1, 3]));
    BatchFill {
        token,
        cursor,
        mask,
        states: Vec::new(),
    }
}

fn batch_reset(graph: &neura_graph::Graph, shape: BatchShape) -> BatchSeed {
    let reset = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
    let share = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
    let cache = cache(shape, graph, 1, shape.shared);
    cache.reset(graph, reset, shape.shared.then_some(share));
    BatchSeed {
        reset,
        share,
        states: Vec::new(),
    }
}

fn batch_prefix(graph: &neura_graph::Graph, shape: BatchShape) -> BatchPrefix {
    let token = graph.input(Shape::of([1, shape.rows, 1, 1]), Element::Single);
    let cursor = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
    let mask = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
    let reset = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
    let cache = match shape.bounded {
        true => Cache::windowed_shared(
            graph,
            "cache",
            CacheShape::new(
                1,
                CLASSES,
                shape.capacity,
                shape.window,
                shape.batch,
                shape.rows,
            ),
            Element::Single,
        ),
        false => Cache::shared(
            graph,
            "cache",
            CacheShape::new(
                1,
                CLASSES,
                shape.capacity,
                shape.window,
                shape.batch,
                shape.rows,
            ),
            Element::Single,
        ),
    };
    cache.clear_shared(graph, reset);
    let held = graph.mul(
        graph.permute(graph.one_hot(token, CLASSES), [0, 2, 1, 3]),
        graph.permute(mask, [0, 2, 1, 3]),
    );
    cache.write(graph, cursor, held, held);
    BatchPrefix {
        token,
        cursor,
        mask,
        reset,
        states: Vec::new(),
    }
}

fn batched(
    first_prompt: &[u32],
    second_prompt: &[u32],
    wanted: usize,
    chunks: &[u32],
    stagger: bool,
    sampling: Option<Sampling>,
    prefix: Option<&[u32]>,
) -> Vec<Vec<u32>> {
    let runtime = runtime();
    let context = 16;
    let batch = 2;
    let spec = BatchSpec::new(context, batch, chunks);
    let spec = match sampling {
        Some(sampling) => spec.sampled(sampling),
        None => spec,
    };
    let spec = match prefix {
        Some(_) => spec.shared(),
        None => spec,
    };
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    if let Some(prefix) = prefix {
        batch.prefix(&runtime, prefix, batch_prefix);
    }
    let first = batch.admit(&runtime, first_prompt);
    let mut walked = vec![Vec::new(); batch.batch() as usize];
    if stagger {
        while walked[first as usize].len() < 2 {
            for held in batch.step(&runtime) {
                if held.slot == first {
                    walked[first as usize].push(held.token);
                }
            }
        }
    }
    let second = batch.admit(&runtime, second_prompt);
    while walked.iter().any(|held| held.len() < wanted) {
        for held in batch.step(&runtime) {
            if held.token == END || walked[held.slot as usize].len() >= wanted {
                batch.finish(held.slot);
                continue;
            }
            walked[held.slot as usize].push(held.token);
        }
        assert!(
            batch.active().next().is_some(),
            "a batch with no token left still walked",
        );
    }
    assert_eq!((first, second), (0, 1));
    walked
}

fn walked_alone(
    prompt: &[u32],
    wanted: usize,
    chunks: &[u32],
    sampling: Option<Sampling>,
) -> Vec<u32> {
    let runtime = runtime();
    let context = 16;
    let batch = 1;
    let spec = BatchSpec::new(context, batch, chunks);
    let spec = match sampling {
        Some(sampling) => spec.sampled(sampling),
        None => spec,
    };
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    batch.admit(&runtime, prompt);
    let mut walked = Vec::new();
    while walked.len() < wanted && batch.active().next().is_some() {
        for held in batch.step(&runtime) {
            if held.token == END {
                batch.finish(held.slot);
                continue;
            }
            walked.push(held.token);
        }
    }
    for slot in batch.active().collect::<Vec<u32>>() {
        batch.finish(slot);
    }
    walked
}

fn framed(
    first_prompt: &[u32],
    second_prompt: &[u32],
    wanted: usize,
    chunks: &[u32],
    waves: u32,
    prefix: Option<&[u32]>,
) -> Vec<Vec<u32>> {
    let runtime = runtime();
    let spec = BatchSpec::new(16, 2, chunks);
    let spec = match prefix {
        Some(_) => spec.shared(),
        None => spec,
    };
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    if let Some(prefix) = prefix {
        batch.prefix(&runtime, prefix, batch_prefix);
    }
    batch.admit(&runtime, first_prompt);
    batch.admit(&runtime, second_prompt);
    let mut walked = vec![Vec::new(); batch.batch() as usize];
    while walked.iter().any(|held| held.len() < wanted) && batch.active().next().is_some() {
        for held in batch.advance(&runtime, waves) {
            if held.token == END || walked[held.slot as usize].len() >= wanted {
                batch.finish(held.slot);
                continue;
            }
            walked[held.slot as usize].push(held.token);
        }
    }
    walked
}

fn budgeted(
    first_prompt: &[u32],
    second_prompt: &[u32],
    wanted: usize,
    chunks: &[u32],
    budget: Duration,
) -> Vec<Vec<u32>> {
    let runtime = runtime();
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        BatchSpec::new(16, 2, chunks),
        batch_answer,
        batch_fill,
        batch_reset,
    );
    batch.admit(&runtime, first_prompt);
    batch.admit(&runtime, second_prompt);
    let mut walked = vec![Vec::new(); batch.batch() as usize];
    while walked.iter().any(|held| held.len() < wanted) && batch.active().next().is_some() {
        for held in batch.advance_within(&runtime, budget) {
            if held.token == END || walked[held.slot as usize].len() >= wanted {
                batch.finish(held.slot);
                continue;
            }
            walked[held.slot as usize].push(held.token);
        }
    }
    walked
}

#[test]
fn a_batch_walks_the_tokens_a_frame_budget_allows() {
    let first = &[0, 1, 1, 2, 1];
    let second = &[0, 1, 2];
    let expected = batched(first, second, 5, &[4, 2], false, None, None);
    for budget in [Duration::from_micros(1), Duration::from_millis(2)] {
        assert_eq!(
            budgeted(first, second, 5, &[4, 2], budget),
            expected,
            "a batch that walks {budget:?} a frame answers with the tokens a whole plan answers with",
        );
    }
}

#[test]
fn a_batch_walks_the_frames_of_a_plan_a_slice_at_a_time() {
    let first = &[0, 1, 1, 2, 1];
    let second = &[0, 1, 2];
    let expected = batched(first, second, 5, &[4, 2], false, None, None);
    for waves in [1, 2, 3, 8] {
        assert_eq!(
            framed(first, second, 5, &[4, 2], waves, None),
            expected,
            "a batch that walks {waves} waves a frame answers with the tokens a whole plan answers with",
        );
    }
    let shared = batched(first, second, 5, &[4, 2], false, None, Some(&[0, 1]));
    assert_eq!(
        framed(first, second, 5, &[4, 2], 2, Some(&[0, 1])),
        shared,
        "a batch that walks a shared prefix a frame at a time answers with the tokens a whole plan answers with",
    );
}

#[test]
fn a_batch_answers_every_plane_with_the_tokens_a_row_answers_with() {
    for chunks in [&[][..], &[4, 2]] {
        for stagger in [false, true] {
            let walked = batched(&[0, 1, 1, 2, 1], &[2, 2, 0], 5, chunks, stagger, None, None);
            assert_eq!(
                walked,
                vec![vec![1, 1, 1, 1, 1], vec![2, 2, 2, 2, 2]],
                "chunks {chunks:?} stagger {stagger}",
            );
        }
    }
}

#[test]
fn a_batch_reuses_the_prefix_its_planes_share() {
    for chunks in [&[][..], &[4, 2]] {
        for sampling in [None, Some(Sampling::sampled(1.0, 4, 1.0))] {
            let plain = batched(
                &[0, 1, 1, 2, 1],
                &[0, 1, 1, 2, 0],
                5,
                chunks,
                false,
                sampling,
                None,
            );
            let shared = batched(
                &[0, 1, 1, 2, 1],
                &[0, 1, 1, 2, 0],
                5,
                chunks,
                false,
                sampling,
                Some(&[0, 1, 1]),
            );
            assert_eq!(shared, plain, "chunks {chunks:?} sampling {sampling:?}");
        }
    }
}

#[test]
fn a_batch_resumes_the_state_a_suspended_plane_kept() {
    let runtime = runtime();
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        BatchSpec::new(16, 1, &[]),
        batch_answer,
        batch_fill,
        batch_reset,
    );
    let slot = batch.admit(&runtime, &[0, 1, 1, 2, 1]);
    let mut first = Vec::new();
    while first.len() < 2 {
        for held in batch.step(&runtime) {
            first.push(held.token);
        }
    }
    batch.suspend(slot);
    assert!(batch.active().next().is_none());
    assert_eq!(batch.occupied().count(), 1);
    batch.resume(slot, &[2, 0]);
    let mut second = Vec::new();
    while second.len() < 2 {
        for held in batch.step(&runtime) {
            second.push(held.token);
        }
    }
    let mut resumed = vec![0, 1, 1, 2, 1];
    resumed.extend_from_slice(&first);
    resumed.extend_from_slice(&[2, 0]);
    assert_eq!(
        second,
        walked_alone(&resumed, 2, &[], None),
        "a resumed plane answers with the tokens a plane that walked the whole history answers with",
    );
    batch.suspend(slot);
    batch.restart(&runtime, slot, &[2, 2, 0]);
    let mut restarted = Vec::new();
    while restarted.len() < 2 {
        for held in batch.step(&runtime) {
            restarted.push(held.token);
        }
    }
    batch.finish(slot);
    assert_eq!(
        restarted,
        walked_alone(&[2, 2, 0], 2, &[], None),
        "a restarted plane answers with the tokens a plane admitted afresh answers with",
    );
}

#[test]
fn a_batch_fills_a_single_prompt_in_the_chunks_its_model_takes() {
    let plain = walked_alone(&[0, 1, 1, 2, 1], 5, &[], None);
    let chunked = walked_alone(&[0, 1, 1, 2, 1], 5, &[8, 4, 2], None);
    assert_eq!(chunked, plain);
    let sampled = walked_alone(
        &[0, 1, 1, 2, 1],
        5,
        &[8, 4, 2],
        Some(Sampling::sampled(1.0, 4, 1.0)),
    );
    assert_eq!(sampled.len(), 5);
}

#[test]
fn a_suspended_plane_keeps_its_cache_while_another_plane_answers() {
    let runtime = runtime();
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        BatchSpec::new(16, 2, &[]),
        batch_answer,
        batch_fill,
        batch_reset,
    );
    let first = batch.admit(&runtime, &[0, 1, 1, 2, 1]);
    let mut held = Vec::new();
    while held.len() < 2 {
        for emitted in batch.step(&runtime) {
            if emitted.slot == first {
                held.push(emitted.token);
            }
        }
    }
    batch.suspend(first);
    let second = batch.admit(&runtime, &[2, 2, 0]);
    let mut other = Vec::new();
    while other.len() < 5 {
        for emitted in batch.step(&runtime) {
            if emitted.slot == second {
                other.push(emitted.token);
            }
        }
    }
    batch.resume(first, &[2, 0]);
    let mut resumed = Vec::new();
    while resumed.len() < 2 {
        for emitted in batch.step(&runtime) {
            if emitted.slot == first {
                resumed.push(emitted.token);
            }
        }
    }
    let mut history = vec![0, 1, 1, 2, 1];
    history.extend(held);
    history.extend([2, 0]);
    assert_eq!(
        resumed,
        walked_alone(&history, 2, &[], None),
        "a plane suspended while another plane answered resumes the state it kept",
    );
}

#[test]
fn a_batch_keeps_only_the_window_of_tokens_it_declares() {
    const WINDOW: u32 = 4;
    let runtime = runtime();
    let prompt = [2, 2, 2, 2, 0, 0, 0, 0];
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        BatchSpec::new(16, 1, &[]).window(WINDOW),
        batch_answer,
        batch_fill,
        batch_reset,
    );
    batch.admit(&runtime, &prompt);
    let mut walked = Vec::new();
    while walked.len() < 4 {
        for emitted in batch.step(&runtime) {
            walked.push(emitted.token);
        }
    }
    batch.finish(0);
    let mut history = prompt.to_vec();
    let mut expected = Vec::new();
    for _ in 0..4 {
        let mut counts = [0u32; CLASSES as usize];
        for token in history.iter().rev().take(WINDOW as usize) {
            counts[*token as usize] += 1;
        }
        let mut next = 0u32;
        for class in 1..CLASSES {
            if counts[class as usize] > counts[next as usize] {
                next = class;
            }
        }
        expected.push(next);
        history.push(next);
    }
    assert_eq!(
        walked, expected,
        "a window of {WINDOW} tokens answers with the tokens its last keys choose",
    );
}

#[test]
fn a_temperature_the_host_writes_steers_the_next_draw_without_recompiling() {
    let runtime = runtime();
    let build = |graph: &neura_graph::Graph, _rows| {
        let token = graph.input(Shape::of([1, 1, 1, 1]), Element::Single);
        let cursor = graph.input(Shape::scalar(), Element::Single);
        let parts = (0..CLASSES)
            .map(|class| graph.fill(Shape::of([1, 1, 1, 1]), if class == 0 { 1.0 } else { 0.0 }))
            .collect::<Vec<_>>();
        let logits = graph.concat(&parts, 3);
        Step {
            token,
            cursor,
            logits,
        }
    };
    let mut family = Family::sampled(
        &runtime,
        |graph| runtime.weights(graph),
        &[1],
        build,
        Sampling::sampled(1.0, CLASSES, 1.0),
    );
    family.set_temperature(&runtime, 0.01);
    for _ in 0..64 {
        assert_eq!(
            family.answer(&runtime, 0, 0),
            0,
            "a temperature of 0.01 keeps the largest logit of a row that holds one",
        );
    }
    family.set_temperature(&runtime, 1000.0);
    let mut drawn = BTreeSet::new();
    for _ in 0..256 {
        drawn.insert(family.answer(&runtime, 0, 0));
    }
    assert!(
        drawn.len() > 1,
        "a temperature of 1000 spreads a row of one largest logit over one class, and the draws stayed in {drawn:?}",
    );
    assert_eq!(
        runtime.built_plans(),
        1,
        "writing a temperature reuses the program the family compiled",
    );
}

#[test]
fn a_candidate_count_the_host_writes_steers_one_plane_alone() {
    let runtime = runtime();
    let spec = BatchSpec::new(16, 2, &[2]).sampled(Sampling::sampled(1.0, CLASSES, 1.0));
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    batch.admit(&runtime, &[0, 1]);
    batch.admit(&runtime, &[0, 1]);
    let plans = runtime.built_plans();
    batch.set_keep_of(&runtime, 0, 1);
    let mut drew = false;
    for _ in 0..64 {
        for held in batch.step(&runtime) {
            if held.slot == 0 {
                assert_eq!(
                    held.token, 0,
                    "a plane that keeps one candidate answers with the class its logits rank first",
                );
            }
            if held.slot == 1 && held.token != 0 {
                drew = true;
            }
        }
        if drew {
            break;
        }
    }
    assert!(
        drew,
        "the plane that keeps every candidate draws no class beyond the one its neighbor keeps",
    );
    assert_eq!(
        runtime.built_plans(),
        plans,
        "writing a candidate count reuses the program the batch compiled",
    );
}

fn state_answer(graph: &neura_graph::Graph, shape: BatchShape) -> BatchStep {
    let token = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
    let cursor = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
    let mask = graph.input(Shape::of([1, 1, shape.batch, 1]), Element::Single);
    let reset = graph.input(Shape::of([1, shape.batch, 1, 1]), Element::Single);
    let cache = cache(shape, graph, 1, shape.shared);
    let held = graph.mul(
        graph.permute(graph.one_hot(token, CLASSES), [0, 2, 1, 3]),
        graph.permute(mask, [0, 2, 1, 3]),
    );
    cache.write_masked(graph, cursor, held, held, graph.permute(mask, [0, 2, 1, 3]));
    let route = graph.named_state(
        "route",
        Shape::of([1, 1, shape.batch, 1]),
        Init::Zero,
        Element::Single,
    );
    BatchStep {
        token,
        cursor,
        mask,
        reset,
        logits: graph.mul(
            graph.one_hot(route, CLASSES),
            graph.fill(Shape::scalar(), 1e6),
        ),
        states: vec![PlaneState::new("route", route, 0.0)],
    }
}

fn state_fill(graph: &neura_graph::Graph, shape: BatchShape) -> BatchFill {
    let mut fill = batch_fill(graph, shape);
    let route = graph.named_state(
        "route",
        Shape::of([1, 1, shape.batch, 1]),
        Init::Zero,
        Element::Single,
    );
    fill.states = vec![PlaneState::new("route", route, 0.0)];
    fill
}

fn state_reset(graph: &neura_graph::Graph, shape: BatchShape) -> BatchSeed {
    let mut seed = batch_reset(graph, shape);
    let route = graph.named_state(
        "route",
        Shape::of([1, 1, shape.batch, 1]),
        Init::Zero,
        Element::Single,
    );
    seed.states = vec![PlaneState::new("route", route, 0.0)];
    seed
}

#[test]
fn a_state_the_host_writes_routes_one_plane_alone() {
    let runtime = runtime();
    let spec = BatchSpec::new(16, 2, &[2]);
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        state_answer,
        state_fill,
        state_reset,
    );
    batch.admit(&runtime, &[0, 1]);
    batch.admit(&runtime, &[0, 1]);
    let state = batch.state("route");
    let plans = runtime.built_plans();
    batch.set_state_of(&runtime, state, 0, 2.0);
    batch.set_state_of(&runtime, state, 1, 1.0);
    assert_eq!(batch.state_of(state, 0), 2.0);
    assert_eq!(batch.state_of(state, 1), 1.0);
    for _ in 0..4 {
        for held in batch.step(&runtime) {
            assert_eq!(
                held.token as f32,
                batch.state_of(state, held.slot),
                "plane {} answers with the class its own state names",
                held.slot,
            );
        }
    }
    assert_eq!(
        runtime.built_plans(),
        plans,
        "writing a state reuses the program the batch compiled",
    );
    batch.set_state(&runtime, state, 3.0);
    batch.step(&runtime);
    let mut routed = false;
    for _ in 0..2 {
        for held in batch.step(&runtime) {
            assert_eq!(
                held.token, 3,
                "a state written for every plane routes every plane",
            );
            routed = true;
        }
    }
    assert!(routed, "the planes answered after their state changed");
}

#[test]
fn a_cumulative_mass_the_host_writes_steers_one_plane_alone() {
    let runtime = runtime();
    let spec = BatchSpec::new(16, 2, &[2]).sampled(Sampling::sampled(1.0, CLASSES, 1.0));
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    batch.admit(&runtime, &[0, 1]);
    batch.admit(&runtime, &[0, 1]);
    let plans = runtime.built_plans();
    batch.set_keep(&runtime, CLASSES);
    batch.set_cumulative_of(&runtime, 0, 0.01);
    let mut drew = false;
    for _ in 0..64 {
        for held in batch.step(&runtime) {
            if held.slot == 0 {
                assert_eq!(
                    held.token, 0,
                    "a plane whose cumulative mass holds one candidate answers with the class its logits rank first",
                );
            }
            if held.slot == 1 && held.token != 0 {
                drew = true;
            }
        }
        if drew {
            break;
        }
    }
    assert!(
        drew,
        "the plane whose cumulative mass holds every candidate draws no class beyond the one its neighbor keeps",
    );
    assert_eq!(
        runtime.built_plans(),
        plans,
        "writing a cumulative mass reuses the program the batch compiled",
    );
}

fn seeded_batch(prompts: &[&[u32]], seeds: &[u32], wanted: usize) -> Vec<Vec<u32>> {
    let runtime = runtime();
    let spec = BatchSpec::new(16, prompts.len() as u32, &[2])
        .sampled(Sampling::sampled(1.0, CLASSES, 1.0));
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    for (slot, seed) in seeds.iter().enumerate() {
        batch.set_seed_of(&runtime, slot as u32, *seed);
    }
    for prompt in prompts {
        batch.admit(&runtime, prompt);
    }
    let mut walked = vec![Vec::new(); prompts.len()];
    while walked.iter().any(|held| held.len() < wanted) && batch.active().next().is_some() {
        for held in batch.step(&runtime) {
            if held.token == END || walked[held.slot as usize].len() >= wanted {
                batch.finish(held.slot);
                continue;
            }
            walked[held.slot as usize].push(held.token);
        }
    }
    walked
}

#[test]
fn a_plane_draws_the_stream_its_own_seed_names_whatever_slot_it_holds() {
    let prompt = [0u32, 1, 1];
    let seeds = [0x1234_5678u32, 0x9abc_def0];
    let pair = seeded_batch(&[&prompt, &prompt], &seeds, 6);
    assert_eq!(
        pair[0],
        seeded_batch(&[&prompt], &seeds[..1], 6)[0],
        "the first plane of a pair draws what the same seed draws alone",
    );
    assert_eq!(
        pair[1],
        seeded_batch(&[&prompt], &seeds[1..], 6)[0],
        "the second plane of a pair draws what the same seed draws alone",
    );
    assert_ne!(pair[0], pair[1], "two seeds draw two streams");
}

#[test]
fn a_suspended_plane_spends_no_seed_while_its_neighbor_draws() {
    let runtime = runtime();
    let spec = BatchSpec::new(16, 2, &[2]).sampled(Sampling::sampled(1.0, CLASSES, 1.0));
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        spec,
        batch_answer,
        batch_fill,
        batch_reset,
    );
    for _ in 0..2 {
        batch.admit(&runtime, &[0, 1, 1]);
    }
    let plans = runtime.built_plans();
    batch.set_seed_of(&runtime, 1, 0x2f0f_1a3b);
    assert_eq!(
        runtime.built_plans(),
        plans,
        "writing a seed reuses the program the batch compiled",
    );
    for _ in 0..3 {
        batch.step(&runtime);
    }
    let plans = runtime.built_plans();
    batch.suspend(1);
    let held = batch.seed_of(1);
    let walked = batch.seed_of(0);
    for _ in 0..4 {
        batch.step(&runtime);
    }
    assert_eq!(
        batch.seed_of(1),
        held,
        "a suspended plane spends no seed while its neighbor draws",
    );
    assert_ne!(
        batch.seed_of(0),
        walked,
        "the plane that draws spends the seed of every draw",
    );
    assert_eq!(
        runtime.built_plans(),
        plans,
        "suspending a plane reuses the program the batch compiled",
    );
}

fn walked(batch: &mut Batch, runtime: &Runtime, slot: u32, count: usize) -> Vec<u32> {
    let mut tokens = Vec::new();
    while tokens.len() < count {
        for held in batch.step(runtime) {
            if held.slot == slot {
                tokens.push(held.token);
            }
        }
    }
    tokens
}

#[test]
fn a_batch_snapshot_walks_the_state_it_held() {
    let runtime = runtime();
    let mut batch = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        BatchSpec::new(16, 2, &[2]),
        batch_answer,
        batch_fill,
        batch_reset,
    );
    let slot = batch.admit(&runtime, &[0, 1, 2]);
    walked(&mut batch, &runtime, slot, 2);
    let snapshot = batch.snapshot().to_bytes();
    let path = std::env::temp_dir().join(format!("neura-batch-{}.neura", std::process::id()));
    batch.save_state(&runtime, &path);
    let expected = walked(&mut batch, &runtime, slot, 4);
    let mut restored = Batch::new(
        &runtime,
        |graph| runtime.weights(graph),
        BatchSpec::new(16, 2, &[2]),
        batch_answer,
        batch_fill,
        batch_reset,
    );
    restored.restore_state(
        &runtime,
        &CheckpointFile::open(&path),
        BatchSnapshot::from_bytes(&snapshot),
    );
    assert_eq!(walked(&mut restored, &runtime, slot, 4), expected);
    std::fs::remove_file(&path).expect("a saved batch state leaves its file");
}
