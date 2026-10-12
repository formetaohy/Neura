use crate::rule::Rule;
use crate::vocabulary::{Part, Vocabulary};

pub struct Tokenizer {
    vocabulary: Vocabulary,
    rule: Box<dyn Rule>,
}

impl Tokenizer {
    pub fn new(vocabulary: Vocabulary, rule: impl Rule + 'static) -> Self {
        Self {
            vocabulary,
            rule: Box::new(rule),
        }
    }

    pub fn vocabulary(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.vocabulary.id(token)
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for part in self.vocabulary.parted(text) {
            match part {
                Part::Token(id) => ids.push(id),
                Part::Text(held) => {
                    for piece in self.rule.pieces(held) {
                        assert!(
                            !piece.is_empty(),
                            "a split rule names the pieces of the text it reads, and one of them holds no character",
                        );
                        ids.extend(self.vocabulary.merge(&self.vocabulary.spell(piece)));
                    }
                }
            }
        }
        ids
    }

    pub fn piece(&self, id: u32) -> Vec<u8> {
        self.vocabulary.piece(id)
    }
}
