use neura_decoder::Decoder;
use neura_runtime::{Program, Runtime};
use neura_tokenizer::{Text, Tokenizer};
use std::time::Instant;

pub struct Phase {
    program: Program,
    decoder: Decoder,
}

pub struct Generator<'t> {
    runtime: Runtime,
    prefill: Phase,
    decode: Phase,
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

impl Phase {
    pub fn new(program: Program, decoder: Decoder) -> Self {
        Self { program, decoder }
    }

    pub fn rows(&self) -> u32 {
        self.decoder.rows()
    }
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
        prefill: Phase,
        decode: Phase,
        tokenizer: &'t Tokenizer,
        context: u32,
    ) -> Self {
        assert_eq!(
            decode.rows(),
            1,
            "a decode walks one row at a time, and this phase hands the device {} rows",
            decode.rows(),
        );
        assert_eq!(
            prefill.decoder.end(),
            decode.decoder.end(),
            "a prefill ends on token {} where this decode ends on {}",
            prefill.decoder.end(),
            decode.decoder.end(),
        );
        assert!(
            prefill.rows() <= context,
            "a prefill chunk of {} rows outruns the {context} tokens this run holds",
            prefill.rows(),
        );
        Self {
            runtime,
            prefill,
            decode,
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
        let chunk = self.prefill.rows() as usize;
        let mut walked = 0usize;
        while walked + chunk <= prompt.len() {
            let held = &prompt[walked..walked + chunk];
            feed(&self.runtime, &self.prefill, held, walked as u32);
            for id in held {
                self.emit(*id, &mut sink);
            }
            walked += chunk;
            self.position = walked as u32;
        }
        let whole = walked > 0 && walked == prompt.len();
        while walked < prompt.len() {
            feed(
                &self.runtime,
                &self.decode,
                &prompt[walked..walked + 1],
                walked as u32,
            );
            self.emit(prompt[walked], &mut sink);
            walked += 1;
            self.position = walked as u32;
        }
        let held = match whole {
            true => &self.prefill,
            false => &self.decode,
        };
        let mut next = choose(&self.runtime, held);
        let mut produced = 0usize;
        while produced < wanted && next != self.decode.decoder.end() && self.position < self.context
        {
            feed(&self.runtime, &self.decode, &[next], self.position);
            self.emit(next, &mut sink);
            self.position += 1;
            produced += 1;
            next = choose(&self.runtime, &self.decode);
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

    fn emit(&mut self, id: u32, sink: &mut impl FnMut(&str)) {
        let text = self.text.push(&self.tokenizer.piece(id));
        if !text.is_empty() {
            sink(&text);
        }
    }
}

fn feed(runtime: &Runtime, phase: &Phase, tokens: &[u32], cursor: u32) {
    let ids = tokens.iter().map(|id| *id as f32).collect::<Vec<f32>>();
    runtime.write(&phase.program, phase.decoder.token(), &ids);
    runtime.write(&phase.program, phase.decoder.cursor(), &[cursor as f32]);
    runtime.run(&phase.program);
}

fn choose(runtime: &Runtime, phase: &Phase) -> u32 {
    let held = runtime.read(&phase.program, phase.decoder.chosen());
    let value = held[phase.decoder.rows() as usize - 1];
    assert!(
        value.fract() == 0.0 && value >= 0.0 && value <= u32::MAX as f32,
        "the decoder chose {value} where a token stands",
    );
    value as u32
}
