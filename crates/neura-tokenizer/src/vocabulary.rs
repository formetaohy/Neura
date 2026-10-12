use neura_gguf::Archive;
use std::collections::{HashMap, HashSet};

pub struct Vocabulary {
    tokens: Vec<String>,
    special: Vec<bool>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    control: Vec<(String, u32)>,
    alphabet: [char; 256],
    bytes: HashMap<char, u8>,
}

pub(crate) enum Part<'a> {
    Token(u32),
    Text(&'a str),
}

impl Vocabulary {
    pub fn read(source: &impl Archive) -> Self {
        let metadata = source.metadata();
        let model = metadata.text("tokenizer.ggml.model");
        assert_eq!(
            model, "gpt2",
            "this tokenizer reads the byte level bpe a gpt2 gguf holds, and this gguf names {model} as its tokenizer.ggml.model",
        );
        let tokens = metadata
            .texts("tokenizer.ggml.tokens")
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<String>>();
        let kinds = metadata
            .array("tokenizer.ggml.token_type")
            .iter()
            .map(|value| {
                let held = value.integer().unwrap_or_else(|| {
                    panic!(
                        "the tokenizer.ggml.token_type of this gguf holds {} where a whole number stands",
                        value.name(),
                    );
                });
                u32::try_from(held).unwrap_or_else(|_| {
                    panic!(
                        "the tokenizer.ggml.token_type of this gguf holds {held} where a word holds what fits one",
                    );
                })
            })
            .collect::<Vec<u32>>();
        let merges = metadata
            .texts("tokenizer.ggml.merges")
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<String>>();
        let mut ids = HashMap::with_capacity(tokens.len());
        let mut special = Vec::with_capacity(tokens.len());
        let mut control = Vec::new();
        for (at, token) in tokens.iter().enumerate() {
            ids.insert(token.clone(), at as u32);
            let held = matches!(kinds.get(at).copied(), Some(2..=4));
            special.push(held);
            if held {
                control.push((token.clone(), at as u32));
            }
        }
        control.sort_by_key(|(token, _)| std::cmp::Reverse(token.len()));
        let mut ranks = HashMap::with_capacity(merges.len());
        for (at, merge) in merges.iter().enumerate() {
            let (left, right) = merge.split_once(' ').unwrap_or_else(|| {
                panic!(
                    "the merge {merge} of this gguf names two pieces where it splits on no space"
                );
            });
            ranks.insert((left.to_owned(), right.to_owned()), at as u32);
        }
        let (alphabet, bytes) = byte_alphabet();
        Self {
            tokens,
            special,
            ids,
            ranks,
            control,
            alphabet,
            bytes,
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    pub(crate) fn spell(&self, text: &str) -> String {
        text.bytes()
            .map(|byte| self.alphabet[byte as usize])
            .collect()
    }

    pub(crate) fn merge(&self, piece: &str) -> Vec<u32> {
        if let Some(id) = self.ids.get(piece) {
            return vec![*id];
        }
        let mut parts = piece
            .chars()
            .map(|character| character.to_string())
            .collect::<Vec<String>>();
        loop {
            let mut best: Option<(u32, usize)> = None;
            for at in 0..parts.len().saturating_sub(1) {
                let rank = self
                    .ranks
                    .get(&(parts[at].clone(), parts[at + 1].clone()))
                    .copied();
                if let Some(rank) = rank
                    && best.is_none_or(|(held, _)| rank < held)
                {
                    best = Some((rank, at));
                }
            }
            let Some((_, at)) = best else {
                break;
            };
            let joined = format!("{}{}", parts[at], parts[at + 1]);
            parts.splice(at..at + 2, [joined]);
        }
        parts
            .iter()
            .map(|part| {
                *self.ids.get(part).unwrap_or_else(|| {
                    panic!("this vocabulary holds no piece {part}");
                })
            })
            .collect()
    }

    pub(crate) fn parted<'a>(&self, text: &'a str) -> Vec<Part<'a>> {
        let mut parts = vec![Part::Text(text)];
        for (token, id) in &self.control {
            let mut next = Vec::with_capacity(parts.len());
            for part in parts {
                let Part::Text(held) = part else {
                    next.push(part);
                    continue;
                };
                let mut at = 0usize;
                while let Some(found) = held[at..].find(token.as_str()) {
                    let start = at + found;
                    if start > at {
                        next.push(Part::Text(&held[at..start]));
                    }
                    next.push(Part::Token(*id));
                    at = start + token.len();
                }
                if at < held.len() {
                    next.push(Part::Text(&held[at..]));
                }
            }
            parts = next;
        }
        parts
    }

    pub fn piece(&self, id: u32) -> Vec<u8> {
        let held = self.tokens.get(id as usize).unwrap_or_else(|| {
            panic!(
                "this vocabulary holds no token {id} of {} tokens",
                self.tokens.len()
            );
        });
        if self.special[id as usize] {
            return held.as_bytes().to_vec();
        }
        held.chars()
            .map(|character| {
                self.bytes.get(&character).copied().unwrap_or_else(|| {
                    panic!("the token {held} holds a character no byte of it stands behind");
                })
            })
            .collect()
    }
}

fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let printable = (0x21u32..=0x7e)
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect::<HashSet<u32>>();
    let mut table = [char::MAX; 256];
    let mut extra = 0u32;
    for byte in 0..=0xffu32 {
        let character = if printable.contains(&byte) {
            byte
        } else {
            let held = 256 + extra;
            extra += 1;
            held
        };
        table[byte as usize] = char::from_u32(character).expect("a byte spells a character");
    }
    let bytes = table
        .iter()
        .enumerate()
        .map(|(byte, character)| (*character, byte as u8))
        .collect::<HashMap<char, u8>>();
    (table, bytes)
}
