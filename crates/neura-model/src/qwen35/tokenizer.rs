use crate::text::qwen::Qwen;
use neura_gguf::Archive;
use neura_tokenizer::Vocabulary;

const END_OF_TURN: &str = "<|im_end|>";

pub struct Tokenizer {
    engine: neura_tokenizer::Tokenizer,
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
        let engine = neura_tokenizer::Tokenizer::new(Vocabulary::read(source), Qwen);
        let end = metadata
            .optional_word("tokenizer.ggml.eos_token_id")
            .or_else(|| engine.id(END_OF_TURN))
            .unwrap_or_else(|| {
                panic!("this gguf names no end of turn, and holds no {END_OF_TURN}");
            });
        Self { engine, end }
    }

    pub fn engine(&self) -> &neura_tokenizer::Tokenizer {
        &self.engine
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.engine.encode(text)
    }

    pub fn piece(&self, id: u32) -> Vec<u8> {
        self.engine.piece(id)
    }

    pub fn vocabulary(&self) -> usize {
        self.engine.vocabulary()
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.engine.id(token)
    }

    pub fn end_of_turn(&self) -> u32 {
        self.end
    }
}
