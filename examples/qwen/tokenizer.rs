use neura::gguf::Value;
use std::collections::HashMap;

pub struct Tokenizer {
    tokens: Vec<String>,
    plain: Vec<bool>,
    ids: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    control: Vec<(String, u32)>,
    alphabet: Vec<char>,
    bytes: HashMap<char, u8>,
}

impl Tokenizer {
    pub fn read(metadata: &std::collections::BTreeMap<String, Value>) -> Self {
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
        let mut control = Vec::new();
        let mut plain = Vec::with_capacity(tokens.len());
        for (at, token) in tokens.iter().enumerate() {
            ids.insert(token.clone(), at as u32);
            let special = matches!(kinds.get(at), Some(&2) | Some(&3));
            plain.push(special);
            if special {
                control.push((token.clone(), at as u32));
            }
        }
        control.sort_by_key(|held| std::cmp::Reverse(held.0.len()));
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
            plain,
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
        let held = text.as_bytes();
        let mut at = 0usize;
        while at < held.len() {
            if let Some((token, id, width)) = self.control_at(text, at) {
                let _ = token;
                ids.push(id);
                at += width;
                continue;
            }
            let (piece, width) = self.pretoken(held, at);
            ids.extend(self.merge(&piece));
            at += width;
        }
        ids
    }

    pub fn decode(&self, id: u32) -> &str {
        &self.tokens[id as usize]
    }

    pub fn piece(&self, id: u32) -> Vec<u8> {
        let held = self.decode(id);
        if self.plain[id as usize] {
            return held.as_bytes().to_vec();
        }
        held.chars()
            .map(|character| self.bytes.get(&character).copied().unwrap_or(b'?'))
            .collect()
    }

    fn control_at(&self, text: &str, at: usize) -> Option<(String, u32, usize)> {
        let rest = &text[at..];
        self.control
            .iter()
            .find(|(token, _)| rest.starts_with(token.as_str()))
            .map(|(token, id)| (token.clone(), *id, token.len()))
    }

    fn pretoken(&self, bytes: &[u8], at: usize) -> (String, usize) {
        let held = |index: usize| bytes.get(index).copied();
        if held(at) == Some(b'\'') {
            for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
                let end = at + 1 + suffix.len();
                if end <= bytes.len()
                    && bytes[at + 1..end]
                        .iter()
                        .zip(suffix.as_bytes())
                        .all(|(left, right)| left.eq_ignore_ascii_case(right))
                {
                    return (self.spell(&bytes[at..end]), end - at);
                }
            }
        }
        let letter = |byte: u8| byte.is_ascii_alphabetic() || byte >= 0x80;
        if at + 1 < bytes.len() && !letter(bytes[at]) && held(at + 1).is_some_and(letter) {
            let end = self.letters(bytes, at + 1);
            return (self.spell(&bytes[at..end]), end - at);
        }
        if held(at).is_some_and(letter) {
            let end = self.letters(bytes, at);
            return (self.spell(&bytes[at..end]), end - at);
        }
        if held(at).is_some_and(|byte| byte.is_ascii_digit()) {
            return (self.spell(&bytes[at..at + 1]), 1);
        }
        if held(at) == Some(b' ') && held(at + 1).is_some_and(|byte| self.punct(byte)) {
            let end = self.run(bytes, at + 1, |byte| self.punct(byte));
            let end = self.newlines(bytes, end);
            return (self.spell(&bytes[at..end]), end - at);
        }
        if held(at).is_some_and(|byte| self.punct(byte)) {
            let end = self.run(bytes, at, |byte| self.punct(byte));
            let end = self.newlines(bytes, end);
            return (self.spell(&bytes[at..end]), end - at);
        }
        if held(at).is_some_and(|byte| byte == b'\n' || byte == b'\r') {
            let end = self.run(bytes, at, |byte| byte == b'\n' || byte == b'\r');
            return (self.spell(&bytes[at..end]), end - at);
        }
        if held(at).is_some_and(|byte| byte == b' ' || byte == b'\t') {
            let end = self.run(bytes, at, |byte| byte == b' ' || byte == b'\t');
            let end = if end == bytes.len() {
                end
            } else {
                let mut held = end;
                while held > at + 1 && bytes[held - 1] == b' ' {
                    held -= 1;
                }
                held
            };
            return (self.spell(&bytes[at..end]), end - at);
        }
        (self.spell(&bytes[at..at + 1]), 1)
    }

    fn letters(&self, bytes: &[u8], at: usize) -> usize {
        self.run(bytes, at, |byte| byte.is_ascii_alphabetic() || byte >= 0x80)
    }

    fn punct(&self, byte: u8) -> bool {
        !byte.is_ascii_alphanumeric() && !byte.is_ascii_whitespace() && byte < 0x80
    }

    fn run(&self, bytes: &[u8], at: usize, accept: impl Fn(u8) -> bool) -> usize {
        let mut end = at;
        while end < bytes.len() && accept(bytes[end]) {
            end += 1;
        }
        end
    }

    fn newlines(&self, bytes: &[u8], at: usize) -> usize {
        self.run(bytes, at, |byte| byte == b'\n' || byte == b'\r')
    }

    fn spell(&self, bytes: &[u8]) -> String {
        bytes
            .iter()
            .map(|byte| self.alphabet[*byte as usize])
            .collect()
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

fn byte_alphabet() -> (Vec<char>, HashMap<char, u8>) {
    let printable = (0x21u32..=0x7e)
        .chain(0xa1..=0xac)
        .chain(0xae..=0xff)
        .collect::<std::collections::HashSet<u32>>();
    let mut table = vec![char::from_u32(0).expect("a code point"); 256];
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

fn array<'v>(metadata: &'v std::collections::BTreeMap<String, Value>, key: &str) -> &'v [Value] {
    metadata
        .get(key)
        .and_then(Value::array)
        .unwrap_or_else(|| panic!("this gguf holds no {key}"))
}
