use crate::generator::{Draw, Sampling, chosen};
use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Cursor, Program, Runtime, Weights};
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

#[derive(Clone, Copy, Debug)]
pub struct BatchShape {
    pub batch: u32,
    pub rows: u32,
    pub capacity: u32,
    pub window: u32,
    pub bounded: bool,
    pub shared: bool,
}

#[derive(Clone, Debug)]
pub struct BatchSpec {
    pub context: u32,
    pub batch: u32,
    pub chunks: Vec<u32>,
    pub sampling: Sampling,
    pub tune: bool,
    pub shared: bool,
    pub window: Option<u32>,
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
            window: None,
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

    pub fn window(mut self, window: u32) -> Self {
        assert!(window > 0, "a batch keeps a window of no token");
        self.window = Some(window);
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
    temperature: Option<Value>,
    temperatures: Vec<f32>,
    compile: fn(&Runtime, &Graph, &Weights) -> Program,
    batch: u32,
    context: u32,
    capacity: u32,
    window: u32,
    bounded: bool,
    seeded: bool,
    shared: bool,
    prefix: Option<Vec<u32>>,
    slots: Vec<Option<Slot>>,
    frame: Option<Frame>,
}

struct Frame {
    fill: Option<(usize, Vec<u32>)>,
    cursor: Cursor,
}

struct Slot {
    cursor: u32,
    phase: Phase,
}

enum Phase {
    Prompting(VecDeque<u32>),
    Generating(u32),
    Suspended,
}

impl Phase {
    fn running(&self) -> bool {
        !matches!(self, Self::Suspended)
    }
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
        answer: impl Fn(&Graph, BatchShape) -> BatchStep,
        fill: impl Fn(&Graph, BatchShape) -> BatchFill,
        reset: impl Fn(&Graph, BatchShape) -> BatchSeed,
    ) -> Self {
        let BatchSpec {
            context,
            batch,
            chunks,
            sampling,
            tune,
            shared,
            window,
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
        let widest = chunks.first().copied().unwrap_or(1);
        let bounded = window.is_some();
        let window = window.unwrap_or(context);
        assert!(
            window <= context,
            "a batch keeps a window of {window} tokens over a context of {context}",
        );
        assert!(
            widest <= window,
            "a batch fills chunks of {widest} rows into a window of {window} tokens",
        );
        let capacity = window
            .checked_add(widest)
            .expect("a cache of the window and the scratch rows a masked chunk fills fits");
        let compile = match tune {
            true => Runtime::tuned,
            false => Runtime::compile,
        };
        let seeded = sampling != Sampling::GREEDY;
        let initial = sampling.temperature().unwrap_or(1.0);
        let mut fills = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let graph = Graph::new();
            let step = fill(
                &graph,
                BatchShape {
                    batch,
                    rows: chunk,
                    capacity,
                    window,
                    bounded,
                    shared,
                },
            );
            held_fill(&graph, &step, chunk, batch);
            if seeded {
                graph.state(Shape::scalar(), Init::Zero, Element::Single);
                temperature_state(&graph, batch, initial);
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
        let step = answer(
            &answering,
            BatchShape {
                batch,
                rows: 1,
                capacity,
                window,
                bounded,
                shared,
            },
        );
        held_answer(&answering, &step, batch);
        let seed = seeded.then(|| answering.state(Shape::scalar(), Init::Zero, Element::Single));
        let advance = seeded.then(|| answering.input(Shape::scalar(), Element::Single));
        let temperature = seeded.then(|| temperature_state(&answering, batch, initial));
        let store = weights(&answering);
        let mut answer = Pending {
            graph: answering,
            step,
            draw: Draw {
                sampling,
                seed,
                advance,
                temperature,
            },
            entry: None,
        };
        answer.entry(runtime, &store, compile);
        let clearing = Graph::new();
        let reset_seed = reset(
            &clearing,
            BatchShape {
                batch,
                rows: 1,
                capacity,
                window,
                bounded,
                shared,
            },
        );
        held_seed(&clearing, &reset_seed, batch);
        if seeded {
            clearing.state(Shape::scalar(), Init::Zero, Element::Single);
            temperature_state(&clearing, batch, initial);
        }
        let reset = compile(runtime, &clearing, &store);
        Self {
            store,
            answer,
            fills,
            prefix_fills: Vec::new(),
            reset,
            reset_seed,
            temperature,
            temperatures: vec![initial; batch as usize],
            compile,
            batch,
            context,
            capacity,
            window,
            bounded,
            seeded,
            shared,
            prefix: None,
            slots: (0..batch).map(|_| None).collect(),
            frame: None,
        }
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn context(&self) -> u32 {
        self.context
    }

    pub fn window(&self) -> u32 {
        self.window
    }

    pub fn chunks(&self) -> impl Iterator<Item = u32> + '_ {
        self.fills.iter().map(|(chunk, _)| *chunk)
    }

    pub fn set_temperature(&mut self, runtime: &Runtime, value: f32) {
        assert!(
            value > 0.0 && value.is_finite(),
            "a sampling temperature of {value} spreads no logits",
        );
        self.temperatures.fill(value);
        self.write_temperatures(runtime);
    }

    pub fn set_temperature_of(&mut self, runtime: &Runtime, slot: u32, value: f32) {
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
        assert!(
            value > 0.0 && value.is_finite(),
            "a sampling temperature of {value} spreads no logits",
        );
        self.temperatures[slot as usize] = value;
        self.write_temperatures(runtime);
    }

    fn write_temperatures(&mut self, runtime: &Runtime) {
        let temperature = self
            .temperature
            .expect("a greedy batch draws no temperature");
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        runtime.write(&answer.program, temperature, &self.temperatures);
    }

    pub fn prefix(
        &mut self,
        runtime: &Runtime,
        tokens: &[u32],
        build: impl Fn(&Graph, BatchShape) -> BatchPrefix,
    ) {
        assert!(
            self.frame.is_none(),
            "a batch fills a shared prefix before it walks a frame",
        );
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
                    let step = build(
                        &graph,
                        BatchShape {
                            batch: self.batch,
                            rows,
                            capacity: self.capacity,
                            window: self.window,
                            bounded: self.bounded,
                            shared: true,
                        },
                    );
                    held_prefix(&graph, &step, rows);
                    if self.seeded {
                        graph.state(Shape::scalar(), Init::Zero, Element::Single);
                        temperature_state(&graph, self.batch, 1.0);
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
            self.frame.is_none(),
            "a batch admits a sequence before it walks a frame",
        );
        let slot = self
            .slots
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| panic!("all {} planes of this batch hold a sequence", self.batch));
        let slot = slot as u32;
        self.restart(runtime, slot, tokens);
        slot
    }

    pub fn resume(&mut self, slot: u32, tokens: &[u32]) {
        assert!(
            self.frame.is_none(),
            "a batch resumes a sequence before it walks a frame",
        );
        let held = self.slots[slot as usize]
            .as_ref()
            .unwrap_or_else(|| panic!("plane {slot} of this batch holds no sequence"));
        assert!(
            matches!(held.phase, Phase::Suspended),
            "plane {slot} resumes only after it is suspended, and it is {} its sequence",
            match held.phase {
                Phase::Prompting(_) => "still feeding",
                Phase::Generating(_) => "still answering",
                Phase::Suspended => "holding",
            },
        );
        assert!(
            !tokens.is_empty(),
            "a sequence resumes with at least one token",
        );
        assert!(
            held.cursor < self.context,
            "a sequence at cursor {} of a context of {} resumes only after the application restarts or releases it",
            held.cursor,
            self.context,
        );
        assert!(
            tokens.len() < (self.context - held.cursor) as usize,
            "a sequence at cursor {} resumes with {} tokens where this batch holds {} of context",
            held.cursor,
            tokens.len(),
            self.context,
        );
        let cursor = held.cursor;
        self.slots[slot as usize] = Some(Slot {
            cursor,
            phase: Phase::Prompting(tokens.iter().copied().collect()),
        });
    }

    pub fn restart(&mut self, runtime: &Runtime, slot: u32, tokens: &[u32]) {
        assert!(
            self.frame.is_none(),
            "a batch restarts a sequence before it walks a frame",
        );
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
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
        self.compile_fills(runtime, &prompt);
        let mut reset = vec![0.0f32; self.batch as usize];
        reset[slot as usize] = 1.0;
        let mut shared = vec![0.0f32; self.batch as usize];
        shared[slot as usize] = share;
        runtime.write(&self.reset, self.reset_seed.reset, &reset);
        runtime.write(&self.reset, self.reset_seed.share, &shared);
        runtime.run(&self.reset);
        self.slots[slot as usize] = Some(Slot {
            cursor,
            phase: Phase::Prompting(prompt.into_iter().collect()),
        });
    }

    pub fn suspend(&mut self, slot: u32) {
        assert!(
            self.frame.is_none(),
            "a batch suspends a sequence before it walks a frame",
        );
        let held = self.slots[slot as usize]
            .as_mut()
            .unwrap_or_else(|| panic!("plane {slot} of this batch holds no sequence"));
        match &held.phase {
            Phase::Generating(_) => held.phase = Phase::Suspended,
            Phase::Prompting(prompt) => panic!(
                "a sequence suspends after it has fed its prompt, and plane {slot} still holds {} prompt tokens",
                prompt.len(),
            ),
            Phase::Suspended => panic!("plane {slot} of this batch is already suspended"),
        }
    }

    pub fn step(&mut self, runtime: &Runtime) -> Vec<Emitted> {
        assert!(
            self.frame.is_none(),
            "a batch steps a whole plan, and this batch walks a frame of one; advance the frame or finish it first",
        );
        if self.active().next().is_none() {
            return Vec::new();
        }
        self.prefill(runtime);
        self.answer(runtime)
    }

    pub fn finish(&mut self, slot: u32) {
        assert!(
            self.frame.is_none(),
            "a batch finishes a sequence before it walks a frame",
        );
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
            .filter(|(_, slot)| slot.as_ref().is_some_and(|held| held.phase.running()))
            .map(|(slot, _)| slot as u32)
    }

    pub fn occupied(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, slot)| slot.is_some())
            .map(|(slot, _)| slot as u32)
    }

    pub fn cursor(&self, slot: u32) -> Option<u32> {
        self.slots[slot as usize].as_ref().map(|held| held.cursor)
    }

    pub fn remaining(&self, slot: u32) -> Option<u32> {
        self.cursor(slot).map(|cursor| self.context - cursor)
    }

    pub fn advance(&mut self, runtime: &Runtime, waves: u32) -> Vec<Emitted> {
        assert!(waves > 0, "a batch walks at least one wave a frame");
        if self.active().next().is_none() {
            assert!(
                self.frame.is_none(),
                "a batch with no running sequence holds no frame",
            );
            return Vec::new();
        }
        if let Some(frame) = self.frame.take() {
            return self.walk(runtime, frame, waves);
        }
        match self.next_fill() {
            Some((index, slots)) => self.start_fill(runtime, index, slots, waves),
            None => self.start_answer(runtime, waves),
        }
    }

    fn walk(&mut self, runtime: &Runtime, frame: Frame, waves: u32) -> Vec<Emitted> {
        match frame.fill {
            Some((index, slots)) => self.resume_fill(runtime, index, slots, frame.cursor, waves),
            None => self.resume_answer(runtime, frame.cursor, waves),
        }
    }

    fn compile_fills(&mut self, runtime: &Runtime, tokens: &[u32]) {
        let mut left = tokens.len();
        while left > 1 {
            let Some(index) = self
                .fills
                .iter()
                .position(|(chunk, _)| (*chunk as usize) < left)
            else {
                break;
            };
            let Self {
                fills,
                store,
                compile,
                ..
            } = self;
            fills[index].1.entry(runtime, store, *compile);
            left -= fills[index].0 as usize;
        }
    }

    fn next_fill(&self) -> Option<(usize, Vec<u32>)> {
        let mut groups = vec![Vec::new(); self.fills.len()];
        for (slot, held) in self.slots.iter().enumerate() {
            let Some(held) = held else {
                continue;
            };
            let Phase::Prompting(prompt) = &held.phase else {
                continue;
            };
            let left = prompt.len();
            if left <= 1 {
                continue;
            }
            assert!(
                held.cursor < self.context,
                "a sequence at cursor {} of a context of {} no longer reaches the prompt it holds",
                held.cursor,
                self.context,
            );
            if let Some(index) = self
                .fills
                .iter()
                .position(|(chunk, _)| (*chunk as usize) < left)
            {
                groups[index].push(slot as u32);
            }
        }
        groups
            .into_iter()
            .enumerate()
            .find(|(_, group)| !group.is_empty())
    }

    fn prepare_fill(&mut self, runtime: &Runtime, index: usize, group: &[u32]) {
        let chunk = self.fills[index].0;
        let rows = chunk as usize;
        let planes = self.batch as usize;
        let mut tokens = vec![0.0f32; rows * planes];
        let mut cursors = vec![0.0f32; planes];
        let mut mask = vec![0.0f32; planes];
        for slot in group {
            let held = self.slots[*slot as usize]
                .as_ref()
                .expect("a plane of a fill group holds a sequence");
            cursors[*slot as usize] = held.cursor as f32;
            mask[*slot as usize] = 1.0;
            let Phase::Prompting(prompt) = &held.phase else {
                panic!("a plane of a fill group feeds a prompt");
            };
            for (row, token) in prompt.iter().take(rows).enumerate() {
                tokens[row * planes + *slot as usize] = *token as f32;
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
    }

    fn complete_fill(&mut self, index: usize, group: &[u32]) {
        let chunk = self.fills[index].0;
        for slot in group {
            let held = self.slots[*slot as usize]
                .as_mut()
                .expect("a plane of a fill group holds a sequence");
            held.cursor += chunk;
            let Phase::Prompting(prompt) = &mut held.phase else {
                panic!("a plane of a fill group feeds a prompt");
            };
            for _ in 0..chunk {
                prompt.pop_front();
            }
        }
    }

    fn fill(&mut self, runtime: &Runtime, index: usize, group: Vec<u32>) {
        self.prepare_fill(runtime, index, &group);
        {
            let Self {
                fills,
                store,
                compile,
                ..
            } = self;
            let fill = fills[index].1.entry(runtime, store, *compile);
            runtime.run(&fill.program);
        }
        self.complete_fill(index, &group);
    }

    fn start_fill(
        &mut self,
        runtime: &Runtime,
        index: usize,
        slots: Vec<u32>,
        waves: u32,
    ) -> Vec<Emitted> {
        self.prepare_fill(runtime, index, &slots);
        let held = {
            let program = &mut self.fills[index]
                .1
                .entry
                .as_mut()
                .expect("a prepared fill holds the program it walks")
                .program;
            let mut session = runtime.session(program);
            session.submit(waves);
            if session.done() {
                session.wait();
                None
            } else {
                Some(session.pause())
            }
        };
        match held {
            Some(cursor) => {
                self.frame = Some(Frame {
                    fill: Some((index, slots)),
                    cursor,
                })
            }
            None => self.complete_fill(index, &slots),
        }
        Vec::new()
    }

    fn resume_fill(
        &mut self,
        runtime: &Runtime,
        index: usize,
        slots: Vec<u32>,
        cursor: Cursor,
        waves: u32,
    ) -> Vec<Emitted> {
        let held = {
            let program = &mut self.fills[index]
                .1
                .entry
                .as_mut()
                .expect("a resumed fill holds the program it walks")
                .program;
            let mut session = runtime.resume(program, cursor);
            session.submit(waves);
            if session.done() {
                session.wait();
                None
            } else {
                Some(session.pause())
            }
        };
        match held {
            Some(cursor) => {
                self.frame = Some(Frame {
                    fill: Some((index, slots)),
                    cursor,
                })
            }
            None => self.complete_fill(index, &slots),
        }
        Vec::new()
    }

    fn prefill(&mut self, runtime: &Runtime) {
        while let Some((index, group)) = self.next_fill() {
            self.fill(runtime, index, group);
        }
    }

    fn answer_inputs(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>, bool) {
        let mut tokens = vec![0.0f32; self.batch as usize];
        let mut cursor = Vec::with_capacity(self.batch as usize);
        let mut mask = Vec::with_capacity(self.batch as usize);
        let mut drawn = false;
        for slot in 0..self.batch {
            let Some(held) = self.slots[slot as usize].as_ref() else {
                cursor.push(0.0);
                mask.push(0.0);
                continue;
            };
            match &held.phase {
                Phase::Prompting(prompt) => {
                    assert!(
                        held.cursor < self.context,
                        "a sequence at cursor {} of a context of {} no longer feeds the prompt it holds",
                        held.cursor,
                        self.context,
                    );
                    let token = *prompt
                        .front()
                        .expect("a prompting plane holds the token it feeds next");
                    drawn |= prompt.len() == 1;
                    tokens[slot as usize] = token as f32;
                    cursor.push(held.cursor as f32);
                    mask.push(1.0);
                }
                Phase::Generating(feed) => {
                    assert!(
                        held.cursor < self.context,
                        "a sequence at cursor {} of a context of {} no longer answers; suspend or release it",
                        held.cursor,
                        self.context,
                    );
                    drawn = true;
                    tokens[slot as usize] = *feed as f32;
                    cursor.push(held.cursor as f32);
                    mask.push(1.0);
                }
                Phase::Suspended => {
                    cursor.push(0.0);
                    mask.push(0.0);
                }
            }
        }
        (tokens, cursor, mask, drawn)
    }

    fn write_answer(
        &mut self,
        runtime: &Runtime,
        tokens: &[f32],
        cursor: &[f32],
        mask: &[f32],
        drawn: bool,
    ) {
        let reset = vec![0.0f32; self.batch as usize];
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        runtime.write(&answer.program, answer.step.token, tokens);
        runtime.write(&answer.program, answer.step.cursor, cursor);
        runtime.write(&answer.program, answer.step.mask, mask);
        runtime.write(&answer.program, answer.step.reset, &reset);
        if let Some(advance) = answer.advance {
            runtime.write(&answer.program, advance, &[if drawn { 1.0 } else { 0.0 }]);
        }
    }

    fn complete_answer(&mut self, chosen: &[f32]) -> Vec<Emitted> {
        let mut emitted = Vec::new();
        for slot in 0..self.batch {
            let Some(held) = self.slots[slot as usize].as_mut() else {
                continue;
            };
            let token = token_of(&chosen[slot as usize]);
            match &mut held.phase {
                Phase::Prompting(prompt) => {
                    prompt.pop_front();
                    if prompt.is_empty() {
                        held.phase = Phase::Generating(token);
                    }
                }
                Phase::Generating(feed) => {
                    emitted.push(Emitted { slot, token: *feed });
                    *feed = token;
                }
                Phase::Suspended => {}
            }
            if held.phase.running() {
                held.cursor += 1;
            }
        }
        emitted
    }

    fn answer(&mut self, runtime: &Runtime) -> Vec<Emitted> {
        let (tokens, cursor, mask, drawn) = self.answer_inputs();
        self.write_answer(runtime, &tokens, &cursor, &mask, drawn);
        let chosen = {
            let answer = self.answer.entry(runtime, &self.store, self.compile);
            runtime.run(&answer.program);
            runtime.read(&answer.program, answer.chosen)
        };
        self.complete_answer(&chosen)
    }

    fn start_answer(&mut self, runtime: &Runtime, waves: u32) -> Vec<Emitted> {
        let (tokens, cursor, mask, drawn) = self.answer_inputs();
        self.write_answer(runtime, &tokens, &cursor, &mask, drawn);
        let (chosen, held) = {
            let held = self
                .answer
                .entry
                .as_ref()
                .expect("a written answer holds a program")
                .chosen;
            let program = &mut self
                .answer
                .entry
                .as_mut()
                .expect("a written answer holds a program")
                .program;
            let mut session = runtime.session(program);
            session.submit(waves);
            if session.done() {
                session.wait();
                (Some(runtime.read(program, held)), None)
            } else {
                (None, Some(session.pause()))
            }
        };
        match held {
            Some(cursor) => {
                self.frame = Some(Frame { fill: None, cursor });
                Vec::new()
            }
            None => self.complete_answer(&chosen.expect("a finished answer holds its choice")),
        }
    }

    fn resume_answer(&mut self, runtime: &Runtime, cursor: Cursor, waves: u32) -> Vec<Emitted> {
        let (chosen, held) = {
            let held = self
                .answer
                .entry
                .as_ref()
                .expect("a resumed answer holds a program")
                .chosen;
            let program = &mut self
                .answer
                .entry
                .as_mut()
                .expect("a resumed answer holds a program")
                .program;
            let mut session = runtime.resume(program, cursor);
            session.submit(waves);
            if session.done() {
                session.wait();
                (Some(runtime.read(program, held)), None)
            } else {
                (None, Some(session.pause()))
            }
        };
        match held {
            Some(cursor) => {
                self.frame = Some(Frame { fill: None, cursor });
                Vec::new()
            }
            None => self.complete_answer(&chosen.expect("a finished answer holds its choice")),
        }
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

fn temperature_state(graph: &Graph, batch: u32, temperature: f32) -> Value {
    graph.state(
        Shape::of([1, 1, batch, 1]),
        Init::Constant(temperature),
        Element::Single,
    )
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
