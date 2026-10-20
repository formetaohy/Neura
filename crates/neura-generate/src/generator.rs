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

    pub fn temperature(self) -> Option<f32> {
        match self {
            Self::Greedy => None,
            Self::Sampled { temperature, .. } => Some(temperature),
        }
    }

    pub fn keep(self) -> Option<u32> {
        match self {
            Self::Greedy => None,
            Self::Sampled { keep, .. } => Some(keep),
        }
    }

    pub fn cumulative(self) -> Option<f32> {
        match self {
            Self::Greedy => None,
            Self::Sampled { cumulative, .. } => Some(cumulative),
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
    store: Weights,
    fills: Vec<(u32, Pending)>,
    answer: Pending,
    temperature: Option<Value>,
    compile: fn(&Runtime, &Graph, &Weights) -> Program,
}

#[derive(Clone, Copy)]
pub(crate) struct Draw {
    pub(crate) sampling: Sampling,
    pub(crate) seed: Option<Value>,
    pub(crate) advance: Option<Value>,
    pub(crate) temperature: Option<Value>,
    pub(crate) keep: Option<Value>,
    pub(crate) cumulative: Option<Value>,
}

struct Pending {
    graph: Graph,
    step: Step,
    draw: Draw,
    entry: Option<Entry>,
}

struct Entry {
    program: Program,
    token: Value,
    cursor: Value,
    advance: Option<Value>,
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
        let initial = sampling.temperature().unwrap_or(1.0);
        let mut graphs = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let graph = Graph::new();
            let step = build(&graph, *chunk);
            held(&graph, &step, *chunk);
            if seeded {
                graph.state(Shape::scalar(), Init::Zero, Element::Single);
                graph.knob(initial);
            }
            graphs.push((graph, step));
        }
        let store = weights(&graphs[0].0);
        let fills = graphs
            .into_iter()
            .zip(chunks)
            .map(|((graph, step), chunk)| {
                (
                    *chunk,
                    Pending {
                        graph,
                        step,
                        draw: Draw {
                            sampling: Sampling::GREEDY,
                            seed: None,
                            advance: None,
                            temperature: None,
                            keep: None,
                            cumulative: None,
                        },
                        entry: None,
                    },
                )
            })
            .collect();
        let answering = Graph::new();
        let step = build(&answering, 1);
        held(&answering, &step, 1);
        let seed = seeded.then(|| answering.state(Shape::scalar(), Init::Zero, Element::Single));
        let advance = seeded.then(|| answering.input(Shape::scalar(), Element::Single));
        let temperature = seeded.then(|| answering.knob(initial));
        let draw = Draw {
            sampling,
            seed,
            advance,
            temperature,
            keep: None,
            cumulative: None,
        };
        let entry = Entry::of(runtime, &answering, &store, step, compile, draw);
        let answer = Pending {
            graph: answering,
            step,
            draw,
            entry: Some(entry),
        };
        Self {
            store,
            fills,
            answer,
            temperature,
            compile,
        }
    }

    pub fn chunks(&self) -> impl Iterator<Item = u32> + '_ {
        self.fills.iter().map(|(chunk, _)| *chunk)
    }

    pub fn set_temperature(&mut self, runtime: &Runtime, value: f32) {
        assert!(
            value > 0.0 && value.is_finite(),
            "a sampling temperature of {value} spreads no logits",
        );
        let temperature = self
            .temperature
            .expect("a greedy family draws no temperature");
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        runtime.write(&answer.program, temperature, &[value]);
    }

    pub fn fill(&mut self, runtime: &Runtime, tokens: &[u32], cursor: u32) {
        let Self {
            store,
            fills,
            compile,
            ..
        } = self;
        let chunks = fills.iter().map(|(chunk, _)| *chunk).collect::<Vec<_>>();
        let fill = fills
            .iter_mut()
            .find(|(chunk, _)| *chunk == tokens.len() as u32)
            .map(|(_, fill)| fill)
            .unwrap_or_else(|| {
                panic!(
                    "a family fills chunks {chunks:?}, and {} tokens arrived",
                    tokens.len(),
                )
            });
        let fill = fill.entry(runtime, store, *compile);
        let ids = tokens.iter().map(|id| *id as f32).collect::<Vec<f32>>();
        runtime.write(&fill.program, fill.token, &ids);
        runtime.write(&fill.program, fill.cursor, &[cursor as f32]);
        runtime.run(&fill.program);
    }

    pub fn answer(&mut self, runtime: &Runtime, token: u32, cursor: u32) -> u32 {
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        runtime.write(&answer.program, answer.token, &[token as f32]);
        runtime.write(&answer.program, answer.cursor, &[cursor as f32]);
        if let Some(advance) = answer.advance {
            runtime.write(&answer.program, advance, &[1.0]);
        }
        runtime.run(&answer.program);
        let held = runtime.read(&answer.program, answer.chosen);
        let value = held[0];
        assert!(
            value.fract() == 0.0 && value >= 0.0 && value <= u32::MAX as f32,
            "the model chose {value} where a token stands",
        );
        value as u32
    }

    pub fn feed(&mut self, runtime: &Runtime, token: u32, cursor: u32) {
        let answer = self.answer.entry(runtime, &self.store, self.compile);
        runtime.write(&answer.program, answer.token, &[token as f32]);
        runtime.write(&answer.program, answer.cursor, &[cursor as f32]);
        if let Some(advance) = answer.advance {
            runtime.write(&answer.program, advance, &[0.0]);
        }
        runtime.run(&answer.program);
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
            self.entry = Some(Entry::of(
                runtime,
                &self.graph,
                weights,
                self.step,
                compile,
                self.draw,
            ));
        }
        self.entry
            .as_mut()
            .expect("a pending entry compiles the program it holds")
    }
}

impl Entry {
    fn of(
        runtime: &Runtime,
        graph: &Graph,
        weights: &Weights,
        step: Step,
        compile: fn(&Runtime, &Graph, &Weights) -> Program,
        draw: Draw,
    ) -> Self {
        let chosen = chosen(graph, step.logits, draw);
        Self {
            program: compile(runtime, graph, weights),
            token: step.token,
            cursor: step.cursor,
            advance: draw.advance,
            chosen,
        }
    }
}

pub(crate) fn chosen(graph: &Graph, logits: Value, draw: Draw) -> Value {
    let chosen = match draw.sampling {
        Sampling::Greedy => graph.argmax(logits),
        Sampling::Sampled {
            temperature: _,
            keep,
            cumulative,
        } => {
            let seed = draw
                .seed
                .expect("a sampled program walks a seed of its own");
            let temperature = draw
                .temperature
                .expect("a sampled program walks the temperature of its own");
            let logits = graph.mul(logits, graph.recip(temperature));
            if let Some(advance) = draw.advance {
                graph.add_into(seed, advance);
            }
            let keep = draw
                .keep
                .unwrap_or_else(|| graph.fill(Shape::scalar(), keep as f32));
            let cumulative = draw
                .cumulative
                .unwrap_or_else(|| graph.fill(Shape::scalar(), cumulative));
            graph.sample(logits, seed, keep, cumulative)
        }
    };
    graph.retain(chosen);
    chosen
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

    pub fn set_temperature(&mut self, value: f32) {
        self.family.set_temperature(&self.runtime, value);
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
        let (last, head) = tokens
            .split_last()
            .expect("a prompt leaves a row for its walk to answer alone");
        for id in head {
            self.family.feed(&self.runtime, *id, self.position);
            self.emit(*id, sink);
            self.position += 1;
        }
        let next = self.family.answer(&self.runtime, *last, self.position);
        self.emit(*last, sink);
        self.position += 1;
        next
    }

    fn emit(&mut self, id: u32, sink: &mut impl FnMut(&str)) {
        let text = self.text.push(&self.tokenizer.piece(id));
        if !text.is_empty() {
            sink(&text);
        }
    }
}
