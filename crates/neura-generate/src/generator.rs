use neura_decoder::Decoder;
use neura_runtime::{Program, Runtime};
use neura_tokenizer::{Text, Tokenizer};
use std::time::Instant;

pub struct Generator<'t> {
    runtime: Runtime,
    program: Program,
    decoder: Decoder,
    tokenizer: &'t Tokenizer,
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
        program: Program,
        decoder: Decoder,
        tokenizer: &'t Tokenizer,
        context: u32,
    ) -> Self {
        for state in decoder.states() {
            let elements = program.span(*state).elements as usize;
            runtime.write(&program, *state, &vec![0.0f32; elements]);
        }
        Self {
            runtime,
            program,
            decoder,
            tokenizer,
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
        for id in prompt.iter().copied() {
            self.step(id);
            self.emit(id, &mut sink);
        }
        let mut next = self.choose();
        let mut produced = 0usize;
        while produced < wanted && next != self.decoder.end() && self.position < self.context {
            self.step(next);
            self.emit(next, &mut sink);
            produced += 1;
            next = self.choose();
        }
        let tail = self.text.flush();
        if !tail.is_empty() {
            sink(&tail);
        }
        Report {
            produced,
            seconds: started.elapsed().as_secs_f64(),
            reached_context: self.position >= self.context,
        }
    }

    fn step(&mut self, id: u32) {
        self.runtime
            .write(&self.program, self.decoder.token(), &[id as f32]);
        self.runtime.write(
            &self.program,
            self.decoder.position(),
            &[self.position as f32],
        );
        self.runtime.run(&self.program);
        self.position += 1;
    }

    fn choose(&self) -> u32 {
        let held = self.runtime.read(&self.program, self.decoder.chosen())[0];
        assert!(
            held.fract() == 0.0 && held >= 0.0 && held <= u32::MAX as f32,
            "the decoder chose {held} where a token stands",
        );
        held as u32
    }

    fn emit(&mut self, id: u32, sink: &mut impl FnMut(&str)) {
        let text = self.text.push(&self.tokenizer.piece(id));
        if !text.is_empty() {
            sink(&text);
        }
    }
}
