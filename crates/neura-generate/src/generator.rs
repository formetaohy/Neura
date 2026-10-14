use neura_graph::{Graph, Value};
use neura_runtime::{Program, Runtime, Weights};
use neura_tokenizer::{Text, Tokenizer};
use std::time::Instant;

pub struct Step {
    pub token: Value,
    pub cursor: Value,
    pub logits: Value,
}

pub struct Family {
    chunk: u32,
    fill: Entry,
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
        chunk: u32,
        build: impl Fn(&Graph, u32) -> Step,
    ) -> Self {
        assert!(
            chunk > 0,
            "a family walks chunks of {} rows, and no step fits a chunk that small",
            chunk,
        );
        let filling = Graph::new();
        let step = build(&filling, chunk);
        held(&filling, &step, chunk);
        let store = weights(&filling);
        let fill = Entry::of(runtime, &filling, &store, step);
        let answering = Graph::new();
        let step = build(&answering, 1);
        held(&answering, &step, 1);
        let answer = Entry::of(runtime, &answering, &store, step);
        Self {
            chunk,
            fill,
            answer,
        }
    }

    pub fn chunk(&self) -> u32 {
        self.chunk
    }

    pub fn fill(&self, runtime: &Runtime, tokens: &[u32], cursor: u32) {
        assert_eq!(
            tokens.len(),
            self.chunk as usize,
            "a fill walks the {} rows its program takes, and {} tokens arrived",
            self.chunk,
            tokens.len(),
        );
        let ids = tokens.iter().map(|id| *id as f32).collect::<Vec<f32>>();
        runtime.write(&self.fill.program, self.fill.token, &ids);
        runtime.write(&self.fill.program, self.fill.cursor, &[cursor as f32]);
        runtime.run(&self.fill.program);
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
    fn of(runtime: &Runtime, graph: &Graph, weights: &Weights, step: Step) -> Self {
        let chosen = graph.argmax(step.logits);
        graph.retain(chosen);
        Self {
            program: runtime.compile(graph, weights),
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
        assert!(
            family.chunk() <= context,
            "a family walks chunks of {} rows where this run holds {} tokens of context",
            family.chunk(),
            context,
        );
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
        let chunk = self.family.chunk() as usize;
        let mut at = 0usize;
        while at + chunk < prompt.len() {
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
        Report {
            produced,
            seconds: started.elapsed().as_secs_f64(),
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
