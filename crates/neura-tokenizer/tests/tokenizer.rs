use neura_gguf::Gguf;
use neura_tokenizer::{Rule, Text, Tokenizer, Vocabulary};

#[path = "support/mod.rs"]
mod support;

use support::{Meta, build};

struct PerCharacter;

impl Rule for PerCharacter {
    fn pieces<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = Vec::new();
        let mut at = 0usize;
        for character in text.chars() {
            let next = at + character.len_utf8();
            pieces.push(&text[at..next]);
            at = next;
        }
        pieces
    }
}

struct PerSpace;

impl Rule for PerSpace {
    fn pieces<'a>(&self, text: &'a str) -> Vec<&'a str> {
        text.split(' ').collect()
    }
}

struct Whole;

impl Rule for Whole {
    fn pieces<'a>(&self, text: &'a str) -> Vec<&'a str> {
        vec![text]
    }
}

fn refuses<T>(action: impl FnOnce() -> T) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn archive(tokens: &[&str], kinds: &[u32], merges: &[&str]) -> Gguf {
    let bytes = build(
        &[
            ("tokenizer.ggml.model", Meta::Text("gpt2")),
            ("tokenizer.ggml.tokens", Meta::Texts(tokens)),
            ("tokenizer.ggml.token_type", Meta::Words(kinds)),
            ("tokenizer.ggml.merges", Meta::Texts(merges)),
        ],
        &[],
    );
    Gguf::decode(&bytes)
}

#[test]
fn a_byte_reads_back_as_the_token_it_spells() {
    let held = ["A", "!", "\u{120}", "\u{10a}", "\u{121}"];
    let source = archive(&held, &[1, 1, 1, 1, 1], &[]);
    let tokenizer = Tokenizer::new(Vocabulary::read(&source), PerCharacter);
    let ids = tokenizer.encode("A! \n\u{7f}");
    assert_eq!(ids, vec![0, 1, 2, 3, 4]);
    let mut bytes = Vec::new();
    for id in ids {
        bytes.extend(tokenizer.piece(id));
    }
    assert_eq!(
        bytes,
        "A! \n\u{7f}".as_bytes(),
        "the byte alphabet spells every byte of the text it reads",
    );
}

#[test]
fn a_piece_of_several_bytes_rounds_through_the_alphabet() {
    let source = archive(
        &["\u{c3}", "\u{a9}", "\u{c3}\u{a9}"],
        &[1, 1, 1],
        &["\u{c3} \u{a9}"],
    );
    let tokenizer = Tokenizer::new(Vocabulary::read(&source), Whole);
    assert_eq!(tokenizer.encode("\u{e9}"), vec![2]);
    assert_eq!(tokenizer.piece(2), "\u{e9}".as_bytes().to_vec());
}

#[test]
fn a_special_token_stands_apart_from_the_text_around_it() {
    let source = archive(&["<|x|>", "a", "b", "ab"], &[3, 1, 1, 1], &["a b"]);
    let tokenizer = Tokenizer::new(Vocabulary::read(&source), Whole);
    assert_eq!(tokenizer.encode("<|x|>ab"), vec![0, 3]);
    assert_eq!(tokenizer.encode("a<|x|>b"), vec![1, 0, 2]);
    assert_eq!(tokenizer.piece(3), b"ab".to_vec());
    assert_eq!(tokenizer.piece(0), b"<|x|>".to_vec());
}

#[test]
fn the_lowest_ranked_pair_merges_first() {
    let source = archive(
        &["a", "b", "c", "bc", "abc"],
        &[1, 1, 1, 1, 1],
        &["b c", "a bc"],
    );
    let tokenizer = Tokenizer::new(Vocabulary::read(&source), Whole);
    assert_eq!(tokenizer.encode("abc"), vec![4]);
    assert_eq!(tokenizer.encode("aabc"), vec![0, 4]);
    assert_eq!(tokenizer.vocabulary(), 5);
    assert_eq!(tokenizer.id("abc"), Some(4));
    assert_eq!(tokenizer.id("d"), None);
}

#[test]
fn the_rule_of_the_caller_decides_where_the_text_splits() {
    let source = archive(&["hello", "world"], &[1, 1], &[]);
    let tokenizer = Tokenizer::new(Vocabulary::read(&source), PerSpace);
    assert_eq!(tokenizer.encode("hello world"), vec![0, 1]);
    assert!(refuses(|| tokenizer.encode("hello  world")));
    let whole = Tokenizer::new(Vocabulary::read(&source), Whole);
    assert!(refuses(|| whole.encode("hello world")));
}

#[test]
fn a_piece_no_token_spells_is_refused() {
    let source = archive(&["a"], &[1], &[]);
    let tokenizer = Tokenizer::new(Vocabulary::read(&source), PerCharacter);
    assert!(refuses(|| tokenizer.encode("b")));
    assert!(refuses(|| tokenizer.piece(1)));
}

#[test]
fn a_tokenizer_that_is_not_byte_level_bpe_is_refused() {
    let bytes = build(
        &[
            ("tokenizer.ggml.model", Meta::Text("llama")),
            ("tokenizer.ggml.tokens", Meta::Texts(&["a"])),
            ("tokenizer.ggml.token_type", Meta::Words(&[1])),
            ("tokenizer.ggml.merges", Meta::Texts(&[])),
        ],
        &[],
    );
    let source = Gguf::decode(&bytes);
    assert!(refuses(|| Vocabulary::read(&source)));
}

#[test]
fn a_merge_that_names_one_piece_is_refused() {
    let source = archive(&["a", "b", "ab"], &[1, 1, 1], &["ab"]);
    assert!(refuses(|| Vocabulary::read(&source)));
}

#[test]
fn text_waits_for_the_bytes_a_character_stands_on() {
    let mut text = Text::new();
    assert_eq!(text.push("We".as_bytes()), "We");
    assert_eq!(text.push(&[0xe4, 0xb8]), "");
    assert_eq!(text.push(&[0x96, 0x21]), "\u{4e16}!");
    assert_eq!(text.push(&[0xff]), "\u{fffd}");
    assert_eq!(text.push(&[0xf0, 0x9f]), "");
    assert_eq!(text.flush(), "\u{fffd}");
}
