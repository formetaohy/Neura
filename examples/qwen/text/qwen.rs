use crate::text::unicode;
use neura::tokenizer::Rule;

pub(crate) struct Qwen;

impl Rule for Qwen {
    fn pieces<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let held = text.chars().collect::<Vec<char>>();
        let mut offsets = vec![0usize];
        for character in &held {
            let last = *offsets
                .last()
                .expect("a piece opens at the first character of the text");
            offsets.push(last + character.len_utf8());
        }
        let letterish = |at: usize| held.get(at).is_some_and(|held| unicode::letterish(*held));
        let whitespace = |at: usize| held.get(at).is_some_and(|held| held.is_whitespace());
        let lower = |at: usize| {
            held.get(at)
                .and_then(|held| held.to_lowercase().next())
                .unwrap_or(char::MAX)
        };
        let punctuation = |character: char| {
            !character.is_whitespace()
                && !unicode::letterish(character)
                && !unicode::is_number(character)
        };
        let mut pieces = Vec::new();
        let mut at = 0usize;
        while at < held.len() {
            let start = at;
            let cpt = held[at];
            if cpt == '\'' && matches!(lower(at + 1), 's' | 't' | 'm' | 'd') {
                at += 2;
                pieces.push(&text[offsets[start]..offsets[at]]);
                continue;
            }
            if cpt == '\''
                && matches!(
                    (lower(at + 1), lower(at + 2)),
                    ('r', 'e') | ('v', 'e') | ('l', 'l')
                )
            {
                at += 3;
                pieces.push(&text[offsets[start]..offsets[at]]);
                continue;
            }
            if !(cpt == '\r' || cpt == '\n' || unicode::is_number(cpt))
                && (unicode::letterish(cpt) || letterish(at + 1))
            {
                at += 1;
                while letterish(at) {
                    at += 1;
                }
                pieces.push(&text[offsets[start]..offsets[at]]);
                continue;
            }
            if unicode::is_number(cpt) {
                at += 1;
                pieces.push(&text[offsets[start]..offsets[at]]);
                continue;
            }
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
                pieces.push(&text[offsets[start]..offsets[at]]);
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
                pieces.push(&text[offsets[start]..offsets[at]]);
                continue;
            }
            if run > 1 && at + run < held.len() {
                at += run - 1;
                pieces.push(&text[offsets[start]..offsets[at]]);
                continue;
            }
            if run > 0 {
                at += run;
                pieces.push(&text[offsets[start]..offsets[at]]);
                continue;
            }
            at += 1;
            pieces.push(&text[offsets[start]..offsets[at]]);
        }
        pieces
    }
}
