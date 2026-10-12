use crate::unicode;
use neura::gguf::Value;
use std::collections::{BTreeMap, HashMap};

pub struct Tokenizer {
    tokens: Vec<String>,
    special: Vec<bool>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    control: Vec<(String, u32)>,
    alphabet: [char; 256],
    bytes: HashMap<char, u8>,
}

enum Part<'a> {
    Token(u32),
    Text(&'a str),
}

impl Tokenizer {
    pub fn read(metadata: &BTreeMap<String, Value>) -> Self {
        let tokens = array(metadata, "tokenizer.ggml.tokens")
            .iter()
            .map(|value| value.text().expect("a token is text").to_owned())
            .collect::<Vec<String>>();
        let kinds = array(metadata, "tokenizer.ggml.token_type")
            .iter()
            .map(|value| value.integer().expect("a token kind is a number") as u32)
            .collect::<Vec<u32>>();
        let merges = array(metadata, "tokenizer.ggml.merges")
            .iter()
            .map(|value| value.text().expect("a merge is text").to_owned())
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
            let (left, right) = merge
                .split_once(' ')
                .unwrap_or_else(|| panic!("a merge {merge} names two pieces"));
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

    pub fn vocabulary(&self) -> usize {
        self.tokens.len()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for part in self.parted(text) {
            match part {
                Part::Token(id) => ids.push(id),
                Part::Text(held) => {
                    for piece in self.pieces(held) {
                        ids.extend(self.merge(&piece));
                    }
                }
            }
        }
        ids
    }

    pub fn piece(&self, id: u32) -> Vec<u8> {
        let held = &self.tokens[id as usize];
        if self.special[id as usize] {
            return held.as_bytes().to_vec();
        }
        held.chars()
            .map(|character| {
                self.bytes.get(&character).copied().unwrap_or_else(|| {
                    panic!("the token {held} holds a character no byte of it stands behind")
                })
            })
            .collect()
    }

    fn parted<'a>(&self, text: &'a str) -> Vec<Part<'a>> {
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

    fn pieces(&self, text: &str) -> Vec<String> {
        let held = text.chars().collect::<Vec<char>>();
        let letterish = |at: usize| held.get(at).is_some_and(|held| unicode::letterish(*held));
        let whitespace = |at: usize| held.get(at).is_some_and(|held| held.is_whitespace());
        let lower = |at: usize| {
            held.get(at)
                .and_then(|held| held.to_lowercase().next())
                .unwrap_or(char::MAX)
        };
        let mut pieces = Vec::new();
        let mut at = 0usize;
        while at < held.len() {
            let start = at;
            let cpt = held[at];
            if cpt == '\'' && matches!(lower(at + 1), 's' | 't' | 'm' | 'd') {
                at += 2;
                pieces.push(self.spell(&held[start..at]));
                continue;
            }
            if cpt == '\''
                && matches!(
                    (lower(at + 1), lower(at + 2)),
                    ('r', 'e') | ('v', 'e') | ('l', 'l')
                )
            {
                at += 3;
                pieces.push(self.spell(&held[start..at]));
                continue;
            }
            if !(cpt == '\r' || cpt == '\n' || unicode::is_number(cpt))
                && (unicode::letterish(cpt) || letterish(at + 1))
            {
                at += 1;
                while letterish(at) {
                    at += 1;
                }
                pieces.push(self.spell(&held[start..at]));
                continue;
            }
            if unicode::is_number(cpt) {
                at += 1;
                pieces.push(self.spell(&held[start..at]));
                continue;
            }
            let punctuation = |character: char| {
                !character.is_whitespace()
                    && !unicode::letterish(character)
                    && !unicode::is_number(character)
            };
            let leading = if cpt == ' ' {
                held.get(at + 1).copied()
            } else {
                Some(cpt)
            };
            if leading.is_some_and(punctuation) {
                at += usize::from(cpt == ' ');
                while held.get(at).copied().is_some_and(punctuation) {
                    at += 1;
                }
                while matches!(held.get(at), Some('\r' | '\n')) {
                    at += 1;
                }
                pieces.push(self.spell(&held[start..at]));
                continue;
            }
            let mut run = 0usize;
            let mut newline = 0usize;
            while whitespace(at + run) {
                if matches!(held[at + run], '\r' | '\n') {
                    newline = at + run + 1;
                }
                run += 1;
            }
            if newline > 0 {
                at = newline;
                pieces.push(self.spell(&held[start..at]));
                continue;
            }
            if run > 1 && at + run < held.len() {
                at += run - 1;
                pieces.push(self.spell(&held[start..at]));
                continue;
            }
            if run > 0 {
                at += run;
                pieces.push(self.spell(&held[start..at]));
                continue;
            }
            at += 1;
            pieces.push(self.spell(&held[start..at]));
        }
        pieces
    }

    fn spell(&self, held: &[char]) -> String {
        let mut piece = String::with_capacity(held.len());
        let mut bytes = [0u8; 4];
        for character in held {
            for byte in character.encode_utf8(&mut bytes).as_bytes() {
                piece.push(self.alphabet[*byte as usize]);
            }
        }
        piece
    }

    fn merge(&self, piece: &str) -> Vec<u32> {
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
                *self
                    .ids
                    .get(part)
                    .unwrap_or_else(|| panic!("this tokenizer holds no piece {part}"))
            })
            .collect()
    }
}

fn byte_alphabet() -> ([char; 256], HashMap<char, u8>) {
    let printable = (0x21u32..=0x7e)
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect::<std::collections::HashSet<u32>>();
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

fn array<'v>(metadata: &'v BTreeMap<String, Value>, key: &str) -> &'v [Value] {
    metadata
        .get(key)
        .and_then(Value::array)
        .unwrap_or_else(|| panic!("this gguf holds no {key}"))
}
