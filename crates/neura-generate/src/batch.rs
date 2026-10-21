use crate::generator::{Draw, Sampling, chosen};
use neura_abi::{CANDIDATES, Element};
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Cursor, Program, Runtime, Weights};
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug)]
pub struct PlaneState {
    name: &'static str,
    value: Value,
    initial: f32,
}

impl PlaneState {
    pub fn new(name: &'static str, value: Value, initial: f32) -> Self {
        assert!(!name.is_empty(), "a plane state names itself");
        assert!(
            initial.is_finite(),
            "a plane state named {name} starts at {initial}",
        );
        Self {
            name,
            value,
            initial,
        }
    }

    pub const fn name(self) -> &'static str {
        self.name
    }

    pub const fn value(self) -> Value {
        self.value
    }

    pub const fn initial(self) -> f32 {
        self.initial
    }
}

pub struct BatchStep {
    pub token: Value,
    pub cursor: Value,
    pub mask: Value,
    pub reset: Value,
    pub logits: Value,
    pub states: Vec<PlaneState>,
}

#[derive(Clone)]
pub struct BatchFill {
    pub token: Value,
    pub cursor: Value,
    pub mask: Value,
    pub states: Vec<PlaneState>,
}

pub struct BatchPrefix {
    pub token: Value,
    pub cursor: Value,
    pub mask: Value,
    pub reset: Value,
    pub states: Vec<PlaneState>,
}

pub struct BatchSeed {
    pub reset: Value,
    pub share: Value,
    pub states: Vec<PlaneState>,
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
    seed: Option<Value>,
    temperature: Option<Value>,
    keep: Option<Value>,
    cumulative: Option<Value>,
    states: Vec<PlaneState>,
    state_values: Vec<Vec<f32>>,
    seeds: Vec<u32>,
    temperatures: Vec<f32>,
    keeps: Vec<f32>,
    cumulatives: Vec<f32>,
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
        let initial_keep = sampling.keep().unwrap_or(1) as f32;
        let initial_cumulative = sampling.cumulative().unwrap_or(1.0);
        let answering = Graph::new();
        let mut step = answer(
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
        let states = std::mem::take(&mut step.states);
        let state_values = states
            .iter()
            .map(|state| vec![state.initial(); batch as usize])
            .collect::<Vec<Vec<f32>>>();
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
            known_states(&states, &step.states, "fill");
            if seeded {
                seed_state(&graph, batch, 0);
                temperature_state(&graph, batch, initial);
                keep_state(&graph, batch, initial_keep);
                cumulative_state(&graph, batch, initial_cumulative);
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
        let seed = seeded.then(|| seed_state(&answering, batch, 0));
        let temperature = seeded.then(|| temperature_state(&answering, batch, initial));
        let keep = seeded.then(|| keep_state(&answering, batch, initial_keep));
        let cumulative = seeded.then(|| cumulative_state(&answering, batch, initial_cumulative));
        let store = weights(&answering);
        let mut answer = Pending {
            graph: answering,
            step,
            draw: Draw {
                sampling,
                seed,
                advance: None,
                temperature,
                keep,
                cumulative,
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
        known_states(&states, &reset_seed.states, "reset");
        if seeded {
            seed_state(&clearing, batch, 0);
            temperature_state(&clearing, batch, initial);
            keep_state(&clearing, batch, initial_keep);
            cumulative_state(&clearing, batch, initial_cumulative);
        }
        let reset = compile(runtime, &clearing, &store);
        let seeds = (0..batch)
            .map(|slot| plane_seed(0, slot))
            .collect::<Vec<_>>();
        let held = Self {
            store,
            answer,
            fills,
            prefix_fills: Vec::new(),
            reset,
            reset_seed,
            seed,
            temperature,
            keep,
            cumulative,
            states,
            state_values,
            seeds,
            temperatures: vec![initial; batch as usize],
            keeps: vec![initial_keep; batch as usize],
            cumulatives: vec![initial_cumulative; batch as usize],
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
        };
        held.write_seeds(runtime);
        held.write_states(runtime);
        held
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

    pub fn set_keep(&mut self, runtime: &Runtime, keep: u32) {
        assert!(
            (1..=CANDIDATES).contains(&keep),
            "a batch keeps {keep} candidates where a row holds between one and {CANDIDATES}",
        );
        self.keeps.fill(keep as f32);
        self.write_keeps(runtime);
    }

    pub fn set_keep_of(&mut self, runtime: &Runtime, slot: u32, keep: u32) {
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
        assert!(
            (1..=CANDIDATES).contains(&keep),
            "a batch keeps {keep} candidates where a row holds between one and {CANDIDATES}",
        );
        self.keeps[slot as usize] = keep as f32;
        self.write_keeps(runtime);
    }

    fn write_keeps(&mut self, runtime: &Runtime) {
        let keep = self.keep.expect("a greedy batch draws no candidate count");
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        runtime.write(&answer.program, keep, &self.keeps);
    }

    pub fn set_cumulative(&mut self, runtime: &Runtime, cumulative: f32) {
        assert!(
            cumulative > 0.0 && cumulative <= 1.0,
            "a cumulative mass of {cumulative} keeps no candidate or names more than every candidate",
        );
        self.cumulatives.fill(cumulative);
        self.write_cumulatives(runtime);
    }

    pub fn set_cumulative_of(&mut self, runtime: &Runtime, slot: u32, cumulative: f32) {
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
        assert!(
            cumulative > 0.0 && cumulative <= 1.0,
            "a cumulative mass of {cumulative} keeps no candidate or names more than every candidate",
        );
        self.cumulatives[slot as usize] = cumulative;
        self.write_cumulatives(runtime);
    }

    fn write_cumulatives(&mut self, runtime: &Runtime) {
        let cumulative = self
            .cumulative
            .expect("a greedy batch draws no cumulative mass");
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        runtime.write(&answer.program, cumulative, &self.cumulatives);
    }

    pub fn states(&self) -> &[PlaneState] {
        &self.states
    }

    pub fn state(&self, name: &str) -> PlaneState {
        self.states
            .iter()
            .copied()
            .find(|held| held.name() == name)
            .unwrap_or_else(|| panic!("a batch registers no state named {name}"))
    }

    pub fn state_of(&self, state: PlaneState, slot: u32) -> f32 {
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
        self.state_values[self.state_index(state)][slot as usize]
    }

    pub fn set_state(&mut self, runtime: &Runtime, state: PlaneState, value: f32) {
        assert!(
            self.frame.is_none(),
            "a batch sets a state before it walks a frame",
        );
        let index = self.state_index(state);
        self.state_values[index].fill(value);
        self.write_states(runtime);
    }

    pub fn set_state_of(&mut self, runtime: &Runtime, state: PlaneState, slot: u32, value: f32) {
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
        assert!(
            self.frame.is_none(),
            "a batch sets a state of a plane before it walks a frame",
        );
        let index = self.state_index(state);
        self.state_values[index][slot as usize] = value;
        self.write_states(runtime);
    }

    fn state_index(&self, state: PlaneState) -> usize {
        self.states
            .iter()
            .position(|held| held.name() == state.name())
            .unwrap_or_else(|| panic!("a batch registers no state named {}", state.name()))
    }

    fn write_states(&self, runtime: &Runtime) {
        let answer = self
            .answer
            .entry
            .as_ref()
            .expect("a written answer holds the program its states belong to");
        write_program_states(
            runtime,
            &answer.program,
            &self.states,
            &self.states,
            &self.state_values,
        );
        for (_, fill) in &self.fills {
            if let Some(entry) = &fill.entry {
                write_program_states(
                    runtime,
                    &entry.program,
                    &entry.step.states,
                    &self.states,
                    &self.state_values,
                );
            }
        }
        for (_, prefix) in &self.prefix_fills {
            write_program_states(
                runtime,
                &prefix.program,
                &prefix.step.states,
                &self.states,
                &self.state_values,
            );
        }
        write_program_states(
            runtime,
            &self.reset,
            &self.reset_seed.states,
            &self.states,
            &self.state_values,
        );
    }

    pub fn seed_of(&self, slot: u32) -> u32 {
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
        assert!(self.seeded, "a greedy batch draws no seed");
        self.seeds[slot as usize]
    }

    pub fn set_seed(&mut self, runtime: &Runtime, seed: u32) {
        assert!(
            self.frame.is_none(),
            "a batch sets the seed of a plane before it walks a frame",
        );
        assert!(self.seeded, "a greedy batch draws no seed");
        for slot in 0..self.batch {
            self.seeds[slot as usize] = plane_seed(seed, slot);
        }
        self.write_seeds(runtime);
    }

    pub fn set_seed_of(&mut self, runtime: &Runtime, slot: u32, seed: u32) {
        assert!(
            slot < self.batch,
            "a batch of {} planes names no plane {slot}",
            self.batch,
        );
        assert!(
            self.frame.is_none(),
            "a batch sets the seed of a plane before it walks a frame",
        );
        assert!(self.seeded, "a greedy batch draws no seed");
        self.seeds[slot as usize] = seed;
        self.write_seeds(runtime);
    }

    fn write_seeds(&self, runtime: &Runtime) {
        let Some(seed) = self.seed else {
            return;
        };
        let values = self
            .seeds
            .iter()
            .copied()
            .map(seed_bits)
            .map(f32::from_bits)
            .collect::<Vec<f32>>();
        let answer = self
            .answer
            .entry
            .as_ref()
            .expect("a written answer holds the program its seeds belong to");
        runtime.write(&answer.program, seed, &values);
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
                    held_prefix(&graph, &step, rows, self.batch);
                    known_states(&self.states, &step.states, "prefix");
                    if self.seeded {
                        seed_state(&graph, self.batch, 0);
                        temperature_state(&graph, self.batch, 1.0);
                        keep_state(&graph, self.batch, 1.0);
                        cumulative_state(&graph, self.batch, 1.0);
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
            write_program_states(
                runtime,
                &entry.program,
                &entry.step.states,
                &self.states,
                &self.state_values,
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
            states,
            state_values,
            ..
        } = self;
        let fill = fills[index].1.entry(runtime, store, *compile);
        runtime.write(&fill.program, fill.step.token, &tokens);
        runtime.write(&fill.program, fill.step.cursor, &cursors);
        runtime.write(&fill.program, fill.step.mask, &mask);
        write_program_states(
            runtime,
            &fill.program,
            &fill.step.states,
            states,
            state_values,
        );
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

    fn answer_inputs(&self) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut tokens = vec![0.0f32; self.batch as usize];
        let mut cursor = Vec::with_capacity(self.batch as usize);
        let mut mask = Vec::with_capacity(self.batch as usize);
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
        (tokens, cursor, mask)
    }

    fn write_answer(&mut self, runtime: &Runtime, tokens: &[f32], cursor: &[f32], mask: &[f32]) {
        let step = &self.answer.step;
        let token = step.token;
        let cursor_value = step.cursor;
        let mask_value = step.mask;
        let reset_value = step.reset;
        let reset = vec![0.0f32; self.batch as usize];
        self.write_seeds(runtime);
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        runtime.write(&answer.program, token, tokens);
        runtime.write(&answer.program, cursor_value, cursor);
        runtime.write(&answer.program, mask_value, mask);
        runtime.write(&answer.program, reset_value, &reset);
    }

    fn complete_answer(&mut self, chosen: &[f32]) -> Vec<Emitted> {
        let mut emitted = Vec::new();
        for slot in 0..self.batch {
            let Some(held) = self.slots[slot as usize].as_mut() else {
                continue;
            };
            let token = token_of(&chosen[slot as usize]);
            let drew = match &held.phase {
                Phase::Prompting(prompt) => prompt.len() == 1,
                Phase::Generating(_) => true,
                Phase::Suspended => false,
            };
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
            if drew {
                self.seeds[slot as usize] = self.seeds[slot as usize].wrapping_add(1);
            }
            if held.phase.running() {
                held.cursor += 1;
            }
        }
        emitted
    }

    fn answer(&mut self, runtime: &Runtime) -> Vec<Emitted> {
        let (tokens, cursor, mask) = self.answer_inputs();
        self.write_answer(runtime, &tokens, &cursor, &mask);
        let chosen = {
            let answer = self.answer.entry(runtime, &self.store, self.compile);
            runtime.run(&answer.program);
            runtime.read(&answer.program, answer.chosen)
        };
        self.complete_answer(&chosen)
    }

    fn start_answer(&mut self, runtime: &Runtime, waves: u32) -> Vec<Emitted> {
        let (tokens, cursor, mask) = self.answer_inputs();
        self.write_answer(runtime, &tokens, &cursor, &mask);
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
                step: self.step.clone(),
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

fn seed_state(graph: &Graph, batch: u32, seed: u32) -> Value {
    graph.state(
        Shape::of([1, 1, batch, 1]),
        Init::Constant(f32::from_bits(seed)),
        Element::Single,
    )
}

fn plane_seed(seed: u32, slot: u32) -> u32 {
    seed ^ slot.wrapping_add(1).wrapping_mul(0x9e37_79b9)
}

fn seed_bits(seed: u32) -> u32 {
    0x3f80_0000 | ((seed ^ (seed >> 23)) & 0x007f_ffff)
}

fn temperature_state(graph: &Graph, batch: u32, temperature: f32) -> Value {
    graph.state(
        Shape::of([1, 1, batch, 1]),
        Init::Constant(temperature),
        Element::Single,
    )
}

fn keep_state(graph: &Graph, batch: u32, keep: f32) -> Value {
    graph.state(
        Shape::of([1, 1, batch, 1]),
        Init::Constant(keep),
        Element::Single,
    )
}

fn cumulative_state(graph: &Graph, batch: u32, cumulative: f32) -> Value {
    graph.state(
        Shape::of([1, 1, batch, 1]),
        Init::Constant(cumulative),
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
    held_states(graph, &step.states, batch);
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
    held_states(graph, &step.states, batch);
}

fn held_prefix(graph: &Graph, step: &BatchPrefix, rows: u32, batch: u32) {
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
    held_states(graph, &step.states, batch);
}

fn held_states(graph: &Graph, states: &[PlaneState], batch: u32) {
    for (index, state) in states.iter().enumerate() {
        assert!(
            states[..index]
                .iter()
                .all(|held| held.name() != state.name()),
            "a batch names state {} twice",
            state.name(),
        );
        if let Some(held) = states[..index]
            .iter()
            .find(|held| held.value() == state.value())
        {
            panic!(
                "a batch writes states {} and {} through one value",
                held.name(),
                state.name(),
            );
        }
        assert_eq!(
            graph.element(state.value()),
            Element::Single,
            "state {} holds {} numbers",
            state.name(),
            graph.element(state.value()).name(),
        );
        assert_eq!(
            graph.shape(state.value()).elements(),
            batch,
            "state {} of {batch} planes walks {:?}",
            state.name(),
            graph.shape(state.value()).dims(),
        );
    }
}

fn known_states(registered: &[PlaneState], states: &[PlaneState], what: &str) {
    assert_eq!(
        states.len(),
        registered.len(),
        "a {what} declares {} states where the answer declares {}",
        states.len(),
        registered.len(),
    );
    for state in states {
        assert!(
            registered.iter().any(|held| held.name() == state.name()),
            "a {what} names state {} that the answer registers no state for",
            state.name(),
        );
    }
}

fn write_program_states(
    runtime: &Runtime,
    program: &Program,
    states: &[PlaneState],
    registered: &[PlaneState],
    values: &[Vec<f32>],
) {
    for state in states {
        let index = registered
            .iter()
            .position(|held| held.name() == state.name())
            .unwrap_or_else(|| {
                panic!(
                    "a program names state {} that the batch does not register",
                    state.name(),
                )
            });
        runtime.write(program, state.value(), &values[index]);
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
    held_states(graph, &seed.states, batch);
}
