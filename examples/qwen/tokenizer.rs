use crate::text::qwen::Qwen;
use neura::gguf::Archive;
use neura::tokenizer::Vocabulary;

const END_OF_TURN: &str = "<|im_end|>";

pub struct Tokenizer {
    engine: neura::tokenizer::Tokenizer,
    end: u32,
}

impl Tokenizer {
    pub fn read(source: &impl Archive) -> Self {
        let metadata = source.metadata();
        let pre = metadata.text("tokenizer.ggml.pre");
        assert_eq!(
            pre, "qwen35",
            "this tokenizer splits the text the qwen35 way, and this gguf names {pre} as its tokenizer.ggml.pre",
        );
        let engine = neura::tokenizer::Tokenizer::new(Vocabulary::read(source), Qwen);
        let end = metadata
            .optional_word("tokenizer.ggml.eos_token_id")
            .or_else(|| engine.id(END_OF_TURN))
            .unwrap_or_else(|| {
                panic!("this gguf names no end of turn, and holds no {END_OF_TURN}");
            });
        Self { engine, end }
    }

    pub fn engine(&self) -> &neura::tokenizer::Tokenizer {
        &self.engine
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.engine.encode(text)
    }

    pub fn vocabulary(&self) -> usize {
        self.engine.vocabulary()
    }

    pub fn end_of_turn(&self) -> u32 {
        self.end
    }
}
