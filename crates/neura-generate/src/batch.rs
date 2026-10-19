use crate::generator::{Draw, Sampling, chosen};
use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Program, Runtime, Weights};
use std::collections::VecDeque;

#[derive(Clone, Copy)]
pub struct BatchStep {
    pub token: Value,
    pub cursor: Value,
    pub mask: Value,
    pub reset: Value,
    pub logits: Value,
}

#[derive(Clone, Copy)]
pub struct BatchFill {
    pub token: Value,
    pub cursor: Value,
    pub mask: Value,
}

#[derive(Clone, Copy)]
pub struct BatchPrefix {
    pub token: Value,
    pub cursor: Value,
    pub mask: Value,
    pub reset: Value,
}

#[derive(Clone, Copy)]
pub struct BatchSeed {
    pub reset: Value,
    pub share: Value,
}

#[derive(Clone, Debug)]
pub struct BatchSpec {
    pub context: u32,
    pub batch: u32,
    pub chunks: Vec<u32>,
    pub sampling: Sampling,
    pub tune: bool,
    pub shared: bool,
}

impl BatchSpec {
    pub fn new(context: u32, batch: u32, chunks: &[u32]) -> Self {
        Self {
            context,
            batch,
            chunks: chunks.to_vec(),
            sampling: Sampling::GREEDY,
            tune: false,
            shared: false,
        }
    }

    pub fn sampled(mut self, sampling: Sampling) -> Self {
        self.sampling = sampling;
        self
    }

    pub fn tuned(mut self) -> Self {
        self.tune = true;
        self
    }

    pub fn shared(mut self) -> Self {
        self.shared = true;
        self
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Emitted {
    pub slot: u32,
    pub token: u32,
}

pub struct Batch {
    store: Weights,
    answer: Pending,
    fills: Vec<(u32, PendingFill)>,
    prefix_fills: Vec<(u32, PrefixEntry)>,
    reset: Program,
    reset_seed: BatchSeed,
    compile: fn(&Runtime, &Graph, &Weights) -> Program,
    batch: u32,
    context: u32,
    capacity: u32,
    seeded: bool,
    shared: bool,
    prefix: Option<Vec<u32>>,
    slots: Vec<Option<Slot>>,
}

struct Slot {
    cursor: u32,
    prompt: VecDeque<u32>,
    feed: u32,
    generating: bool,
}

struct Pending {
    graph: Graph,
    step: BatchStep,
    draw: Draw,
    entry: Option<Entry>,
}

struct Entry {
    program: Program,
    step: BatchStep,
    advance: Option<Value>,
    chosen: Value,
}

struct PendingFill {
    graph: Graph,
    step: BatchFill,
    entry: Option<FillEntry>,
}

struct FillEntry {
    program: Program,
    step: BatchFill,
}

struct PrefixEntry {
    program: Program,
    step: BatchPrefix,
}

impl Batch {
    pub fn new(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        spec: BatchSpec,
        answer: impl Fn(&Graph, u32, u32, bool) -> BatchStep,
        fill: impl Fn(&Graph, u32, u32, u32, bool) -> BatchFill,
        reset: impl Fn(&Graph, u32, u32, bool) -> BatchSeed,
    ) -> Self {
        let BatchSpec {
            context,
            batch,
            chunks,
            sampling,
            tune,
            shared,
        } = spec;
        assert!(batch > 0, "a batch of no plane answers nothing");
        assert!(context > 0, "a context of no token holds no batch");
        for pair in chunks.windows(2) {
            assert!(
                pair[0] > pair[1],
                "a batch fills the largest chunk of {chunks:?} first, and fills {} then {}",
                pair[0],
                pair[1],
            );
        }
        for chunk in &chunks {
            assert!(
                *chunk > 1 && *chunk < context,
                "a batch fills {chunk} rows of a context of {context}, and a chunk holds between two rows and the rows a prompt leaves for the answer",
            );
        }
        let widest = chunks.first().copied().unwrap_or(1).max(batch);
        let capacity = context
            .checked_add(widest)
            .expect("a cache of the context and one scratch chunk a plane fits");
        let compile = match tune {
            true => Runtime::tuned,
            false => Runtime::compile,
        };
        let seeded = sampling != Sampling::GREEDY;
        let mut fills = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let graph = Graph::new();
            let step = fill(&graph, chunk, batch, capacity, shared);
            held_fill(&graph, &step, chunk, batch);
            if seeded {
                graph.state(Shape::scalar(), Init::Zero, Element::Single);
            }
            fills.push((
                chunk,
                PendingFill {
                    graph,
                    step,
                    entry: None,
                },
            ));
        }
        let answering = Graph::new();
        let step = answer(&answering, batch, capacity, shared);
        held_answer(&answering, &step, batch);
        let seed = seeded.then(|| answering.state(Shape::scalar(), Init::Zero, Element::Single));
        let advance = seeded.then(|| answering.input(Shape::scalar(), Element::Single));
        let store = weights(&answering);
        let mut answer = Pending {
            graph: answering,
            step,
            draw: Draw {
                sampling,
                seed,
                advance,
            },
            entry: None,
        };
        answer.entry(runtime, &store, compile);
        let clearing = Graph::new();
        let reset_seed = reset(&clearing, batch, capacity, shared);
        held_seed(&clearing, &reset_seed, batch);
        if seeded {
            clearing.state(Shape::scalar(), Init::Zero, Element::Single);
        }
        let reset = compile(runtime, &clearing, &store);
        Self {
            store,
            answer,
            fills,
            prefix_fills: Vec::new(),
            reset,
            reset_seed,
            compile,
            batch,
            context,
            capacity,
            seeded,
            shared,
            prefix: None,
            slots: (0..batch).map(|_| None).collect(),
        }
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn context(&self) -> u32 {
        self.context
    }

    pub fn chunks(&self) -> impl Iterator<Item = u32> + '_ {
        self.fills.iter().map(|(chunk, _)| *chunk)
    }

    pub fn prefix(
        &mut self,
        runtime: &Runtime,
        tokens: &[u32],
        build: impl Fn(&Graph, u32, u32, u32) -> BatchPrefix,
    ) {
        assert!(
            self.shared,
            "a batch fills a shared prefix only when it declares one",
        );
        assert!(
            self.slots.iter().all(Option::is_none),
            "a batch fills a shared prefix while no plane holds a sequence",
        );
        assert!(
            !tokens.is_empty(),
            "a shared prefix holds at least one token",
        );
        assert!(
            tokens.len() < self.context as usize,
            "a shared prefix holds {} tokens where this batch holds {} of context, and the prompt after it holds at least one token",
            tokens.len(),
            self.context,
        );
        if self.prefix.as_deref() == Some(tokens) {
            return;
        }
        let bulk = self.fills.first().map_or(1, |(chunk, _)| *chunk);
        let mut spans = Vec::new();
        let mut at = 0usize;
        while bulk > 1 && tokens.len() - at >= bulk as usize {
            spans.push((at, bulk));
            at += bulk as usize;
        }
        spans.extend((at..tokens.len()).map(|row| (row, 1)));
        for (index, (start, rows)) in spans.into_iter().enumerate() {
            let position = match self
                .prefix_fills
                .iter()
                .position(|(chunk, _)| *chunk == rows)
            {
                Some(position) => position,
                None => {
                    let graph = Graph::new();
                    let step = build(&graph, rows, self.batch, self.capacity);
                    held_prefix(&graph, &step, rows);
                    if self.seeded {
                        graph.state(Shape::scalar(), Init::Zero, Element::Single);
                    }
                    let program = (self.compile)(runtime, &graph, &self.store);
                    self.prefix_fills
                        .push((rows, PrefixEntry { program, step }));
                    self.prefix_fills.len() - 1
                }
            };
            let (_, entry) = &self.prefix_fills[position];
            let ids = tokens[start..start + rows as usize]
                .iter()
                .map(|token| *token as f32)
                .collect::<Vec<f32>>();
            runtime.write(&entry.program, entry.step.token, &ids);
            runtime.write(&entry.program, entry.step.cursor, &[start as f32]);
            runtime.write(&entry.program, entry.step.mask, &[1.0]);
            runtime.write(
                &entry.program,
                entry.step.reset,
                &[if index == 0 { 1.0 } else { 0.0 }],
            );
            runtime.run(&entry.program);
        }
        self.prefix = Some(tokens.to_vec());
    }

    pub fn forget_prefix(&mut self) {
        self.prefix = None;
    }

    pub fn admit(&mut self, runtime: &Runtime, tokens: &[u32]) -> u32 {
        assert!(
            !tokens.is_empty(),
            "a batch admits a prompt of at least one token",
        );
        assert!(
            tokens.len() < self.context as usize,
            "the prompt holds {} tokens where this batch holds {} of context",
            tokens.len(),
            self.context,
        );
        let (cursor, prompt, share) = match &self.prefix {
            Some(prefix) => {
                assert!(
                    tokens.starts_with(prefix),
                    "a batch shares a prefix of {} tokens, and this prompt does not start with it",
                    prefix.len(),
                );
                assert!(
                    tokens.len() > prefix.len(),
                    "a prompt of {} tokens holds no token after the shared prefix of {} tokens",
                    tokens.len(),
                    prefix.len(),
                );
                (prefix.len() as u32, tokens[prefix.len()..].to_vec(), 1.0f32)
            }
            None => (0, tokens.to_vec(), 0.0f32),
        };
        let slot = self
            .slots
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| panic!("all {} planes of this batch hold a sequence", self.batch));
        let slot = slot as u32;
        let mut reset = vec![0.0f32; self.batch as usize];
        reset[slot as usize] = 1.0;
        let mut shared = vec![0.0f32; self.batch as usize];
        shared[slot as usize] = share;
        runtime.write(&self.reset, self.reset_seed.reset, &reset);
        runtime.write(&self.reset, self.reset_seed.share, &shared);
        runtime.run(&self.reset);
        self.slots[slot as usize] = Some(Slot {
            cursor,
            prompt: prompt.into_iter().collect(),
            feed: 0,
            generating: false,
        });
        slot
    }

    pub fn step(&mut self, runtime: &Runtime) -> Vec<Emitted> {
        if self.slots.iter().all(Option::is_none) {
            return Vec::new();
        }
        self.prefill(runtime);
        self.answer(runtime)
    }

    pub fn finish(&mut self, slot: u32) {
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
        self.slots[slot as usize] = None;
    }

    pub fn active(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some())
            .map(|(slot, _)| slot as u32)
    }

    pub fn cursor(&self, slot: u32) -> Option<u32> {
        self.slots[slot as usize].as_ref().map(|held| held.cursor)
    }

    fn prefill(&mut self, runtime: &Runtime) {
        let mut groups = vec![Vec::new(); self.fills.len()];
        for (slot, held) in self.slots.iter().enumerate() {
            let Some(held) = held else {
                continue;
            };
            let left = held.prompt.len();
            if left <= 1 {
                continue;
            }
            if let Some(index) = self
                .fills
                .iter()
                .position(|(chunk, _)| (*chunk as usize) < left)
            {
                groups[index].push(slot);
            }
        }
        for (index, group) in groups.into_iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let chunk = self.fills[index].0;
            let rows = chunk as usize;
            let planes = self.batch as usize;
            let mut tokens = vec![0.0f32; rows * planes];
            let mut cursors = vec![self.context as f32; planes];
            let mut mask = vec![0.0f32; planes];
            for slot in &group {
                let held = self.slots[*slot]
                    .as_ref()
                    .expect("a plane of a fill group holds a sequence");
                cursors[*slot] = held.cursor as f32;
                mask[*slot] = 1.0;
                for (row, token) in held.prompt.iter().take(rows).enumerate() {
                    tokens[row * planes + *slot] = *token as f32;
                }
            }
            let Self {
                fills,
                store,
                compile,
                ..
            } = self;
            let fill = fills[index].1.entry(runtime, store, *compile);
            runtime.write(&fill.program, fill.step.token, &tokens);
            runtime.write(&fill.program, fill.step.cursor, &cursors);
            runtime.write(&fill.program, fill.step.mask, &mask);
            runtime.run(&fill.program);
            for slot in group {
                let held = self.slots[slot]
                    .as_mut()
                    .expect("a plane of a fill group holds a sequence");
                held.cursor += chunk;
                for _ in 0..chunk {
                    held.prompt.pop_front();
                }
            }
        }
    }

    fn answer(&mut self, runtime: &Runtime) -> Vec<Emitted> {
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        let mut tokens = vec![0.0f32; self.batch as usize];
        let mut cursor = Vec::with_capacity(self.batch as usize);
        let mut mask = Vec::with_capacity(self.batch as usize);
        let mut feeding = Vec::with_capacity(self.batch as usize);
        let mut drawn = false;
        for slot in 0..self.batch {
            match self.slots[slot as usize].as_mut() {
                Some(held) => {
                    let prompt = held.prompt.pop_front();
                    let last = prompt.is_some() && held.prompt.is_empty();
                    drawn |= held.generating || last;
                    let token = prompt.unwrap_or(held.feed);
                    tokens[slot as usize] = token as f32;
                    cursor.push(held.cursor as f32);
                    mask.push(1.0);
                    feeding.push(prompt.is_some());
                }
                None => {
                    cursor.push(self.context as f32);
                    mask.push(0.0);
                    feeding.push(false);
                }
            }
        }
        let reset = vec![0.0f32; self.batch as usize];
        runtime.write(&answer.program, answer.step.token, &tokens);
        runtime.write(&answer.program, answer.step.cursor, &cursor);
        runtime.write(&answer.program, answer.step.mask, &mask);
        runtime.write(&answer.program, answer.step.reset, &reset);
        if let Some(advance) = answer.advance {
            runtime.write(&answer.program, advance, &[if drawn { 1.0 } else { 0.0 }]);
        }
        runtime.run(&answer.program);
        let chosen = runtime.read(&answer.program, answer.chosen);
        let mut emitted = Vec::new();
        for slot in 0..self.batch {
            let Some(held) = self.slots[slot as usize].as_mut() else {
                continue;
            };
            let token = token_of(&chosen[slot as usize]);
            let feeding_prompt = feeding[slot as usize];
            let prompt_left = !held.prompt.is_empty();
            if held.generating {
                emitted.push(Emitted {
                    slot,
                    token: held.feed,
                });
                held.feed = token;
            } else if feeding_prompt && !prompt_left {
                held.feed = token;
                held.generating = true;
            }
            held.cursor += 1;
        }
        emitted
    }
}

impl Pending {
    fn entry(
        &mut self,
        runtime: &Runtime,
        weights: &Weights,
        compile: fn(&Runtime, &Graph, &Weights) -> Program,
    ) -> &mut Entry {
        if self.entry.is_none() {
            let chosen = chosen(&self.graph, self.step.logits, self.draw);
            self.entry = Some(Entry {
                program: compile(runtime, &self.graph, weights),
                step: self.step,
                advance: self.draw.advance,
                chosen,
            });
        }
        self.entry
            .as_mut()
            .expect("a pending entry compiles the program it holds")
    }
}

impl PendingFill {
    fn entry(
        &mut self,
        runtime: &Runtime,
        weights: &Weights,
        compile: fn(&Runtime, &Graph, &Weights) -> Program,
    ) -> &mut FillEntry {
        if self.entry.is_none() {
            self.entry = Some(FillEntry {
                program: compile(runtime, &self.graph, weights),
                step: self.step,
            });
        }
        self.entry
            .as_mut()
            .expect("a pending fill compiles the program it holds")
    }
}

fn token_of(value: &f32) -> u32 {
    assert!(
        value.fract() == 0.0 && *value >= 0.0 && *value <= u32::MAX as f32,
        "the model chose {value} where a token stands",
    );
    *value as u32
}

fn held_answer(graph: &Graph, step: &BatchStep, batch: u32) {
    assert_eq!(
        graph.shape(step.token).dims(),
        [1, 1, batch, 1],
        "an answer over {batch} planes reads one token a plane, and the token value {} walks {:?}",
        step.token.id(),
        graph.shape(step.token).dims(),
    );
    assert_eq!(
        graph.shape(step.mask).dims(),
        [1, 1, batch, 1],
        "an answer over {batch} planes carries one mask a plane, and the mask value {} walks {:?}",
        step.mask.id(),
        graph.shape(step.mask).dims(),
    );
    for (value, name) in [(step.cursor, "cursor"), (step.reset, "reset")] {
        assert_eq!(
            graph.shape(value).dims(),
            [1, batch, 1, 1],
            "an answer of {batch} planes walks one {name} a plane, and value {} walks {:?}",
            value.id(),
            graph.shape(value).dims(),
        );
    }
    let logits = graph.shape(step.logits).dims();
    assert_eq!(
        &logits[..3],
        &[1, 1, batch],
        "an answer over {batch} planes answers with one row a plane, and the logits value {} walks {:?}",
        step.logits.id(),
        logits,
    );
}

fn held_fill(graph: &Graph, step: &BatchFill, rows: u32, batch: u32) {
    assert_eq!(
        graph.shape(step.token).dims(),
        [1, rows, batch, 1],
        "a fill of {rows} rows over {batch} planes reads one token a row a plane, and the token value {} walks {:?}",
        step.token.id(),
        graph.shape(step.token).dims(),
    );
    assert_eq!(
        graph.shape(step.mask).dims(),
        [1, 1, batch, 1],
        "a fill over {batch} planes carries one mask a plane, and the mask value {} walks {:?}",
        step.mask.id(),
        graph.shape(step.mask).dims(),
    );
    assert_eq!(
        graph.shape(step.cursor).dims(),
        [1, batch, 1, 1],
        "a fill of {rows} rows over {batch} planes starts each plane at one cursor, and the cursor value {} walks {:?}",
        step.cursor.id(),
        graph.shape(step.cursor).dims(),
    );
}

fn held_prefix(graph: &Graph, step: &BatchPrefix, rows: u32) {
    assert_eq!(
        graph.shape(step.token).dims(),
        [1, rows, 1, 1],
        "a prefix fill of {rows} rows reads one token a row, and the token value {} walks {:?}",
        step.token.id(),
        graph.shape(step.token).dims(),
    );
    for (value, name) in [
        (step.cursor, "cursor"),
        (step.mask, "mask"),
        (step.reset, "reset"),
    ] {
        assert_eq!(
            graph.shape(value).dims(),
            [1, 1, 1, 1],
            "a prefix fill of {rows} rows walks one {name}, and value {} walks {:?}",
            value.id(),
            graph.shape(value).dims(),
        );
    }
}

fn held_seed(graph: &Graph, seed: &BatchSeed, batch: u32) {
    for (value, name) in [(seed.reset, "reset"), (seed.share, "share")] {
        assert_eq!(
            graph.shape(value).dims(),
            [1, batch, 1, 1],
            "a seed of {batch} planes walks one {name} a plane, and value {} walks {:?}",
            value.id(),
            graph.shape(value).dims(),
        );
    }
}
