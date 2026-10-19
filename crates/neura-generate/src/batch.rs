use crate::generator::{Sampling, chosen};
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Emitted {
    pub slot: u32,
    pub token: u32,
}

pub struct Batch {
    store: Weights,
    answer: Pending,
    reset: Program,
    reset_input: Value,
    compile: fn(&Runtime, &Graph, &Weights) -> Program,
    batch: u32,
    context: u32,
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
    sampling: Sampling,
    seed: Option<Value>,
    entry: Option<Entry>,
}

struct Entry {
    program: Program,
    step: BatchStep,
    chosen: Value,
}

struct Setup {
    context: u32,
    batch: u32,
    sampling: Sampling,
    compile: fn(&Runtime, &Graph, &Weights) -> Program,
}

impl Batch {
    pub fn of(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        context: u32,
        batch: u32,
        answer: impl Fn(&Graph, u32, u32) -> BatchStep,
        reset: impl Fn(&Graph, u32, u32) -> Value,
    ) -> Self {
        Self::with(
            runtime,
            weights,
            Setup {
                context,
                batch,
                sampling: Sampling::GREEDY,
                compile: Runtime::compile,
            },
            answer,
            reset,
        )
    }

    pub fn sampled(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        context: u32,
        batch: u32,
        answer: impl Fn(&Graph, u32, u32) -> BatchStep,
        reset: impl Fn(&Graph, u32, u32) -> Value,
        sampling: Sampling,
    ) -> Self {
        Self::with(
            runtime,
            weights,
            Setup {
                context,
                batch,
                sampling,
                compile: Runtime::compile,
            },
            answer,
            reset,
        )
    }

    pub fn tuned(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        context: u32,
        batch: u32,
        answer: impl Fn(&Graph, u32, u32) -> BatchStep,
        reset: impl Fn(&Graph, u32, u32) -> Value,
    ) -> Self {
        Self::with(
            runtime,
            weights,
            Setup {
                context,
                batch,
                sampling: Sampling::GREEDY,
                compile: Runtime::tuned,
            },
            answer,
            reset,
        )
    }

    pub fn tuned_sampled(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        context: u32,
        batch: u32,
        answer: impl Fn(&Graph, u32, u32) -> BatchStep,
        reset: impl Fn(&Graph, u32, u32) -> Value,
        sampling: Sampling,
    ) -> Self {
        Self::with(
            runtime,
            weights,
            Setup {
                context,
                batch,
                sampling,
                compile: Runtime::tuned,
            },
            answer,
            reset,
        )
    }

    fn with(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        setup: Setup,
        answer: impl Fn(&Graph, u32, u32) -> BatchStep,
        reset: impl Fn(&Graph, u32, u32) -> Value,
    ) -> Self {
        let Setup {
            context,
            batch,
            sampling,
            compile,
        } = setup;
        assert!(batch > 0, "a batch of no plane answers nothing");
        assert!(context > 0, "a context of no token holds no batch");
        let capacity = context
            .checked_add(batch)
            .expect("a cache of the context and one scratch slot a plane fits");
        let seeded = sampling != Sampling::GREEDY;

        let answering = Graph::new();
        let step = answer(&answering, batch, capacity);
        held_answer(&answering, &step, batch);
        let seed = seeded.then(|| answering.state(Shape::scalar(), Init::Zero, Element::Single));
        let store = weights(&answering);
        let mut answer = Pending {
            graph: answering,
            step,
            sampling,
            seed,
            entry: None,
        };
        answer.entry(runtime, &store, compile);

        let clearing = Graph::new();
        let reset_input = reset(&clearing, batch, capacity);
        if seeded {
            clearing.state(Shape::scalar(), Init::Zero, Element::Single);
        }
        let reset = compile(runtime, &clearing, &store);

        Self {
            store,
            answer,
            reset,
            reset_input,
            compile,
            batch,
            context,
            slots: (0..batch).map(|_| None).collect(),
        }
    }

    pub fn batch(&self) -> u32 {
        self.batch
    }

    pub fn context(&self) -> u32 {
        self.context
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
        let slot = self
            .slots
            .iter()
            .position(Option::is_none)
            .unwrap_or_else(|| panic!("all {} planes of this batch hold a sequence", self.batch));
        let slot = slot as u32;
        let mut reset = vec![0.0f32; self.batch as usize];
        reset[slot as usize] = 1.0;
        runtime.write(&self.reset, self.reset_input, &reset);
        runtime.run(&self.reset);
        self.slots[slot as usize] = Some(Slot {
            cursor: 0,
            prompt: tokens.iter().copied().collect(),
            feed: 0,
            generating: false,
        });
        slot
    }

    pub fn step(&mut self, runtime: &Runtime) -> Vec<Emitted> {
        if self.slots.iter().all(Option::is_none) {
            return Vec::new();
        }
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        let mut ids = vec![0.0f32; self.batch as usize];
        let mut cursor = Vec::with_capacity(self.batch as usize);
        let mut mask = Vec::with_capacity(self.batch as usize);
        let mut feeding = Vec::with_capacity(self.batch as usize);
        for slot in 0..self.batch {
            match self.slots[slot as usize].as_mut() {
                Some(held) => {
                    let prompt = held.prompt.pop_front();
                    let token = prompt.unwrap_or(held.feed);
                    ids[slot as usize] = token as f32;
                    cursor.push(held.cursor as f32);
                    mask.push(1.0);
                    feeding.push(prompt.is_some());
                }
                None => {
                    cursor.push((self.context + slot) as f32);
                    mask.push(0.0);
                    feeding.push(false);
                }
            }
        }
        let reset = vec![0.0f32; self.batch as usize];
        runtime.write(&answer.program, answer.step.token, &ids);
        runtime.write(&answer.program, answer.step.cursor, &cursor);
        runtime.write(&answer.program, answer.step.mask, &mask);
        runtime.write(&answer.program, answer.step.reset, &reset);
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
}

impl Pending {
    fn entry(
        &mut self,
        runtime: &Runtime,
        weights: &Weights,
        compile: fn(&Runtime, &Graph, &Weights) -> Program,
    ) -> &mut Entry {
        if self.entry.is_none() {
            let chosen = chosen(&self.graph, self.step.logits, self.sampling, self.seed);
            self.entry = Some(Entry {
                program: compile(runtime, &self.graph, weights),
                step: self.step,
                chosen,
            });
        }
        self.entry
            .as_mut()
            .expect("a pending entry compiles the program it holds")
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
    for (value, name) in [
        (step.cursor, "cursor"),
        (step.mask, "mask"),
        (step.reset, "reset"),
    ] {
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
