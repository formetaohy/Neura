use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Program, Runtime, Weights};
use neura_tokenizer::{Text, Tokenizer};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Sampling {
    Greedy,
    Sampled {
        temperature: f32,
        keep: u32,
        cumulative: f32,
    },
}

impl Sampling {
    pub const GREEDY: Self = Self::Greedy;

    pub fn sampled(temperature: f32, keep: u32, cumulative: f32) -> Self {
        assert!(
            temperature > 0.0 && temperature.is_finite(),
            "a sampling temperature of {temperature} spreads no logits",
        );
        assert!(
            (1..=neura_abi::CANDIDATES).contains(&keep),
            "a sample keeps {keep} candidates where a row holds between one and {}",
            neura_abi::CANDIDATES,
        );
        assert!(
            cumulative > 0.0 && cumulative <= 1.0,
            "a cumulative mass of {cumulative} keeps no candidate or names more than every candidate",
        );
        Self::Sampled {
            temperature,
            keep,
            cumulative,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Step {
    pub token: Value,
    pub cursor: Value,
    pub logits: Value,
}

pub struct Family {
    fills: Vec<(u32, Entry)>,
    answer: Entry,
}

struct Entry {
    program: Program,
    token: Value,
    cursor: Value,
    chosen: Value,
}

impl Family {
    pub fn of(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        chunks: &[u32],
        build: impl Fn(&Graph, u32) -> Step,
    ) -> Self {
        Self::with(
            runtime,
            weights,
            chunks,
            build,
            Sampling::GREEDY,
            Runtime::compile,
        )
    }

    pub fn tuned(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        chunks: &[u32],
        build: impl Fn(&Graph, u32) -> Step,
    ) -> Self {
        Self::with(
            runtime,
            weights,
            chunks,
            build,
            Sampling::GREEDY,
            Runtime::tuned,
        )
    }

    pub fn sampled(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        chunks: &[u32],
        build: impl Fn(&Graph, u32) -> Step,
        sampling: Sampling,
    ) -> Self {
        Self::with(runtime, weights, chunks, build, sampling, Runtime::compile)
    }

    pub fn tuned_sampled(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        chunks: &[u32],
        build: impl Fn(&Graph, u32) -> Step,
        sampling: Sampling,
    ) -> Self {
        Self::with(runtime, weights, chunks, build, sampling, Runtime::tuned)
    }

    fn with(
        runtime: &Runtime,
        weights: impl FnOnce(&Graph) -> Weights,
        chunks: &[u32],
        build: impl Fn(&Graph, u32) -> Step,
        sampling: Sampling,
        compile: fn(&Runtime, &Graph, &Weights) -> Program,
    ) -> Self {
        assert!(
            !chunks.is_empty(),
            "a family fills the rows of at least one chunk, and this family holds none",
        );
        assert!(
            chunks.iter().all(|chunk| *chunk > 0),
            "a family fills chunks of rows, and this family fills {chunks:?}",
        );
        for pair in chunks.windows(2) {
            assert!(
                pair[0] > pair[1],
                "a family fills the largest chunk of {chunks:?} first, and this family fills {} then {}",
                pair[0],
                pair[1],
            );
        }
        let seeded = sampling != Sampling::GREEDY;
        let mut graphs = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let graph = Graph::new();
            let step = build(&graph, *chunk);
            held(&graph, &step, *chunk);
            if seeded {
                graph.state(Shape::scalar(), Init::Zero, Element::Single);
            }
            graphs.push((graph, step));
        }
        let store = weights(&graphs[0].0);
        let fills = graphs
            .iter()
            .zip(chunks)
            .map(|((graph, step), chunk)| {
                (
                    *chunk,
                    Entry::of(
                        runtime,
                        graph,
                        &store,
                        *step,
                        compile,
                        Sampling::GREEDY,
                        None,
                    ),
                )
            })
            .collect();
        let answering = Graph::new();
        let step = build(&answering, 1);
        held(&answering, &step, 1);
        let seed = seeded.then(|| answering.state(Shape::scalar(), Init::Zero, Element::Single));
        let answer = Entry::of(runtime, &answering, &store, step, compile, sampling, seed);
        Self { fills, answer }
    }

    pub fn chunks(&self) -> impl Iterator<Item = u32> + '_ {
        self.fills.iter().map(|(chunk, _)| *chunk)
    }

    pub fn fill(&self, runtime: &Runtime, tokens: &[u32], cursor: u32) {
        let fill = self
            .fills
            .iter()
            .find(|(chunk, _)| *chunk == tokens.len() as u32)
            .map(|(_, fill)| fill)
            .unwrap_or_else(|| {
                panic!(
                    "a family fills chunks {:?}, and {} tokens arrived",
                    self.fills
                        .iter()
                        .map(|(chunk, _)| *chunk)
                        .collect::<Vec<_>>(),
                    tokens.len(),
                )
            });
        let ids = tokens.iter().map(|id| *id as f32).collect::<Vec<f32>>();
        runtime.write(&fill.program, fill.token, &ids);
        runtime.write(&fill.program, fill.cursor, &[cursor as f32]);
        runtime.run(&fill.program);
    }

    pub fn answer(&self, runtime: &Runtime, token: u32, cursor: u32) -> u32 {
        runtime.write(&self.answer.program, self.answer.token, &[token as f32]);
        runtime.write(&self.answer.program, self.answer.cursor, &[cursor as f32]);
        runtime.run(&self.answer.program);
        let held = runtime.read(&self.answer.program, self.answer.chosen);
        let value = held[0];
        assert!(
            value.fract() == 0.0 && value >= 0.0 && value <= u32::MAX as f32,
            "the model chose {value} where a token stands",
        );
        value as u32
    }
}

impl Entry {
    fn of(
        runtime: &Runtime,
        graph: &Graph,
        weights: &Weights,
        step: Step,
        compile: fn(&Runtime, &Graph, &Weights) -> Program,
        sampling: Sampling,
        seed: Option<Value>,
    ) -> Self {
        let chosen = match sampling {
            Sampling::Greedy => graph.argmax(step.logits),
            Sampling::Sampled {
                temperature,
                keep,
                cumulative,
            } => {
                let seed = seed.expect("a sampled program walks a seed of its own");
                let logits = if temperature == 1.0 {
                    step.logits
                } else {
                    graph.mul(step.logits, graph.fill(Shape::scalar(), 1.0 / temperature))
                };
                graph.add_into(seed, graph.fill(Shape::scalar(), 1.0));
                graph.sample(
                    logits,
                    seed,
                    graph.fill(Shape::scalar(), keep as f32),
                    graph.fill(Shape::scalar(), cumulative),
                )
            }
        };
        graph.retain(chosen);
        Self {
            program: compile(runtime, graph, weights),
            token: step.token,
            cursor: step.cursor,
            chosen,
        }
    }
}

fn held(graph: &Graph, step: &Step, rows: u32) {
    assert_eq!(
        graph.shape(step.token).dims(),
        [1, 1, rows, 1],
        "a step of {rows} rows reads one token a row, and the token value {} walks {:?}",
        step.token.id(),
        graph.shape(step.token).dims(),
    );
    assert_eq!(
        graph.shape(step.cursor).dims(),
        [1, 1, 1, 1],
        "a step of {rows} rows starts at one cursor, and the cursor value {} walks {:?}",
        step.cursor.id(),
        graph.shape(step.cursor).dims(),
    );
    let logits = graph.shape(step.logits).dims();
    assert_eq!(
        &logits[..3],
        &[1, 1, 1],
        "a step of {rows} rows answers with the logits of its last row alone, and the logits value {} walks {:?}",
        step.logits.id(),
        logits,
    );
}

pub struct Generator<'t> {
    runtime: Runtime,
    family: Family,
    tokenizer: &'t Tokenizer,
    end: u32,
    text: Text,
    position: u32,
    context: u32,
}

pub struct Report {
    produced: usize,
    seconds: f64,
    reached_context: bool,
}

impl Report {
    pub fn produced(&self) -> usize {
        self.produced
    }

    pub fn seconds(&self) -> f64 {
        self.seconds
    }

    pub fn reached_context(&self) -> bool {
        self.reached_context
    }
}

impl<'t> Generator<'t> {
    pub fn new(
        runtime: Runtime,
        family: Family,
        tokenizer: &'t Tokenizer,
        end: u32,
        context: u32,
    ) -> Self {
        for chunk in family.chunks() {
            assert!(
                chunk <= context,
                "a family walks chunks of {chunk} rows where this run holds {context} tokens of context",
            );
        }
        Self {
            runtime,
            family,
            tokenizer,
            end,
            text: Text::new(),
            position: 0,
            context,
        }
    }

    pub fn run(&mut self, prompt: &[u32], wanted: usize, mut sink: impl FnMut(&str)) -> Report {
        assert!(
            !prompt.is_empty(),
            "a prompt holds no token this tokenizer knows",
        );
        assert!(
            prompt.len() < self.context as usize,
            "the prompt holds {} tokens where this run holds {} of context",
            prompt.len(),
            self.context,
        );
        let started = Instant::now();
        let mut at = 0usize;
        loop {
            let left = prompt.len() - at;
            let filled_chunk = self
                .family
                .chunks()
                .find(|chunk| left as u32 > *chunk)
                .map(|chunk| chunk as usize);
            let Some(chunk) = filled_chunk else {
                break;
            };
            let held = &prompt[at..at + chunk];
            self.family.fill(&self.runtime, held, at as u32);
            for id in held {
                self.emit(*id, &mut sink);
            }
            at += chunk;
            self.position = at as u32;
        }
        let mut next = self.walk(&prompt[at..], &mut sink);
        let mut produced = 0usize;
        while produced < wanted && next != self.end && self.position < self.context {
            self.emit(next, &mut sink);
            next = self.family.answer(&self.runtime, next, self.position);
            self.position += 1;
            produced += 1;
        }
        let reached_context = self.position >= self.context;
        let tail = self.text.flush();
        if !tail.is_empty() {
            sink(&tail);
        }
        let seconds = started.elapsed().as_secs_f64();
        Report {
            produced,
            seconds,
            reached_context,
        }
    }

    fn walk(&mut self, tokens: &[u32], sink: &mut impl FnMut(&str)) -> u32 {
        let mut held = tokens.iter();
        let first = held
            .next()
            .expect("a prompt leaves a row for its walk to answer alone");
        let mut next = self.family.answer(&self.runtime, *first, self.position);
        self.emit(*first, sink);
        self.position += 1;
        for id in held {
            next = self.family.answer(&self.runtime, *id, self.position);
            self.emit(*id, sink);
            self.position += 1;
        }
        next
    }

    fn emit(&mut self, id: u32, sink: &mut impl FnMut(&str)) {
        let text = self.text.push(&self.tokenizer.piece(id));
        if !text.is_empty() {
            sink(&text);
        }
    }
}
