use neura_abi::Element;
use neura_decoder::Decoder;
use neura_gguf::Gguf;
use neura_graph::{Graph, Shape};
use neura_model::qwen35::{Config, Model, Tokenizer};

#[path = "support/mod.rs"]
mod support;

use support::{Meta, build, singles};

const HIDDEN: u64 = 8;
const HEADS: u64 = 2;
const KEY_HEADS: u64 = 1;
const WIDTH: u64 = 4;
const FEED_FORWARD: u64 = 16;
const LINEAR_HEADS: u64 = 2;
const LINEAR_WIDTH: u64 = 4;
const KERNEL: u64 = 4;
const LAYERS: u64 = 2;
const CONTEXT: u32 = 16;
const END_OF_TURN: &str = "<|im_end|>";

fn refuses<T>(action: impl FnOnce() -> T) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn alphabet(byte: u8) -> char {
    let printable = matches!(byte, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
    if printable {
        return char::from(byte);
    }
    let extra = (0..byte)
        .filter(|held| !matches!(held, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff))
        .count();
    char::from_u32(256 + extra as u32).expect("a byte spells a character of its own")
}

fn spelled(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| alphabet(*byte)).collect()
}

struct Corpus {
    tokens: Vec<String>,
    kinds: Vec<u32>,
}

impl Corpus {
    fn of(tokens: Vec<String>) -> Self {
        let mut kinds = vec![1u32; tokens.len()];
        kinds[tokens.len() - 1] = 3;
        Self { tokens, kinds }
    }

    fn bytes() -> Self {
        Self::of(
            (0x21u8..=0x7e)
                .map(|byte| char::from(byte).to_string())
                .chain([END_OF_TURN.to_owned()])
                .collect(),
        )
    }

    fn end(&self) -> u32 {
        (self.tokens.len() - 1) as u32
    }

    fn decode(&self, arch: &str, tensors: &[(String, Vec<u64>, Vec<u8>)], with_eos: bool) -> Gguf {
        let held = self
            .tokens
            .iter()
            .map(String::as_str)
            .collect::<Vec<&str>>();
        let mut metadata = vec![
            ("general.architecture", Meta::Text(arch)),
            ("qwen35.block_count", Meta::Unsigned32(LAYERS as u32)),
            ("qwen35.context_length", Meta::Unsigned32(64)),
            ("qwen35.embedding_length", Meta::Unsigned32(HIDDEN as u32)),
            (
                "qwen35.feed_forward_length",
                Meta::Unsigned32(FEED_FORWARD as u32),
            ),
            (
                "qwen35.attention.head_count",
                Meta::Unsigned32(HEADS as u32),
            ),
            (
                "qwen35.attention.head_count_kv",
                Meta::Unsigned32(KEY_HEADS as u32),
            ),
            (
                "qwen35.attention.key_length",
                Meta::Unsigned32(WIDTH as u32),
            ),
            ("qwen35.rope.freq_base", Meta::Single(1_000_000.0)),
            (
                "qwen35.rope.dimension_count",
                Meta::Unsigned32(WIDTH as u32),
            ),
            ("qwen35.full_attention_interval", Meta::Unsigned32(2)),
            (
                "qwen35.attention.layer_norm_rms_epsilon",
                Meta::Single(1e-6),
            ),
            ("qwen35.ssm.conv_kernel", Meta::Unsigned32(KERNEL as u32)),
            (
                "qwen35.ssm.state_size",
                Meta::Unsigned32(LINEAR_WIDTH as u32),
            ),
            (
                "qwen35.ssm.group_count",
                Meta::Unsigned32(LINEAR_HEADS as u32),
            ),
            (
                "qwen35.ssm.time_step_rank",
                Meta::Unsigned32(LINEAR_HEADS as u32),
            ),
            (
                "qwen35.ssm.inner_size",
                Meta::Unsigned32((LINEAR_HEADS * LINEAR_WIDTH) as u32),
            ),
            ("tokenizer.ggml.model", Meta::Text("gpt2")),
            ("tokenizer.ggml.pre", Meta::Text("qwen35")),
            ("tokenizer.ggml.tokens", Meta::Texts(&held)),
            ("tokenizer.ggml.token_type", Meta::Words(&self.kinds)),
            ("tokenizer.ggml.merges", Meta::Texts(&[])),
        ];
        if with_eos {
            metadata.push(("tokenizer.ggml.eos_token_id", Meta::Unsigned32(self.end())));
        }
        let tensors = tensors
            .iter()
            .map(|(name, shape, payload)| {
                (name.as_str(), 0u32, shape.as_slice(), payload.as_slice())
            })
            .collect::<Vec<(&str, u32, &[u64], &[u8])>>();
        Gguf::decode(&build(&metadata, &tensors))
    }
}

struct Weights {
    tensors: Vec<(String, Vec<u64>, Vec<u8>)>,
}

impl Weights {
    fn new() -> Self {
        Self {
            tensors: Vec::new(),
        }
    }

    fn matrix(&mut self, name: &str, rows: u64, columns: u64) {
        self.hold(name, vec![columns, rows]);
    }

    fn vector(&mut self, name: &str, width: u64) {
        self.hold(name, vec![width]);
    }

    fn hold(&mut self, name: &str, shape: Vec<u64>) {
        let elements = shape.iter().product();
        self.tensors
            .push((name.to_owned(), shape, singles(elements)));
    }

    fn model(vocabulary: u64) -> Self {
        let mut held = Self::new();
        let linear = LINEAR_HEADS * LINEAR_WIDTH;
        held.matrix("token_embd.weight", vocabulary, HIDDEN);
        held.vector("output_norm.weight", HIDDEN);
        held.vector("blk.0.attn_norm.weight", HIDDEN);
        held.matrix("blk.0.attn_qkv.weight", linear * 3, HIDDEN);
        held.matrix("blk.0.attn_gate.weight", linear, HIDDEN);
        held.matrix("blk.0.ssm_out.weight", HIDDEN, linear);
        held.matrix("blk.0.ssm_conv1d.weight", linear * 3, KERNEL);
        held.matrix("blk.0.ssm_alpha.weight", LINEAR_HEADS, HIDDEN);
        held.matrix("blk.0.ssm_beta.weight", LINEAR_HEADS, HIDDEN);
        held.vector("blk.0.ssm_a", LINEAR_HEADS);
        held.vector("blk.0.ssm_dt.bias", LINEAR_HEADS);
        held.vector("blk.0.ssm_norm.weight", LINEAR_WIDTH);
        held.vector("blk.0.post_attention_norm.weight", HIDDEN);
        held.matrix("blk.0.ffn_gate.weight", FEED_FORWARD, HIDDEN);
        held.matrix("blk.0.ffn_up.weight", FEED_FORWARD, HIDDEN);
        held.matrix("blk.0.ffn_down.weight", HIDDEN, FEED_FORWARD);
        held.vector("blk.1.attn_norm.weight", HIDDEN);
        held.matrix("blk.1.attn_q.weight", HEADS * WIDTH * 2, HIDDEN);
        held.matrix("blk.1.attn_k.weight", KEY_HEADS * WIDTH, HIDDEN);
        held.matrix("blk.1.attn_v.weight", KEY_HEADS * WIDTH, HIDDEN);
        held.matrix("blk.1.attn_output.weight", HIDDEN, HEADS * WIDTH);
        held.vector("blk.1.attn_q_norm.weight", WIDTH);
        held.vector("blk.1.attn_k_norm.weight", WIDTH);
        held.vector("blk.1.post_attention_norm.weight", HIDDEN);
        held.matrix("blk.1.ffn_gate.weight", FEED_FORWARD, HIDDEN);
        held.matrix("blk.1.ffn_up.weight", FEED_FORWARD, HIDDEN);
        held.matrix("blk.1.ffn_down.weight", HIDDEN, FEED_FORWARD);
        held
    }
}

#[test]
fn a_config_reads_the_metadata_qwen35_weighs() {
    let corpus = Corpus::bytes();
    let source = corpus.decode(
        "qwen35",
        &Weights::model(corpus.tokens.len() as u64).tensors,
        true,
    );
    let config = Config::read(source.metadata(), corpus.tokens.len());
    assert_eq!(config.layers, LAYERS as u32);
    assert_eq!(config.hidden, HIDDEN as u32);
    assert_eq!(config.feed_forward, FEED_FORWARD as u32);
    assert_eq!(config.heads, HEADS as u32);
    assert_eq!(config.key_heads, KEY_HEADS as u32);
    assert_eq!(config.head_width, WIDTH as u32);
    assert_eq!(config.rope_base, 1_000_000.0);
    assert_eq!(config.rope_width, WIDTH as u32);
    assert_eq!(config.interval, 2);
    assert_eq!(config.linear_heads, LINEAR_HEADS as u32);
    assert_eq!(config.linear_width, LINEAR_WIDTH as u32);
    assert_eq!(config.conv_kernel, KERNEL as u32);
    assert_eq!(config.vocabulary, corpus.tokens.len() as u32);
    assert_eq!(config.context, 64);
    assert_eq!(config.floor, 1e-6);
    assert!(!config.attends(0) && config.attends(1));
    assert_eq!(
        config.mixed_width(),
        (LINEAR_HEADS * LINEAR_WIDTH * 3) as u32,
    );
    let other = corpus.decode("llama", &[], true);
    assert!(refuses(|| Config::read(other.metadata(), 1)));
}

#[test]
fn the_tokenizer_reads_the_split_the_end_and_the_vocabulary_of_qwen35() {
    let corpus = Corpus::bytes();
    let source = corpus.decode(
        "qwen35",
        &Weights::model(corpus.tokens.len() as u64).tensors,
        true,
    );
    let tokenizer = Tokenizer::read(&source);
    assert_eq!(tokenizer.vocabulary(), corpus.tokens.len());
    assert_eq!(tokenizer.end_of_turn(), corpus.end());
    assert_eq!(tokenizer.id(END_OF_TURN), Some(corpus.end()));
    let held = corpus.decode(
        "qwen35",
        &Weights::model(corpus.tokens.len() as u64).tensors,
        false,
    );
    assert_eq!(
        Tokenizer::read(&held).end_of_turn(),
        corpus.end(),
        "a gguf that names no end of turn still holds the token one stands on",
    );
}

#[test]
fn the_split_walks_the_ways_the_text_of_a_qwen_walks() {
    let corpus = Corpus::of(
        [
            spelled(b"Hello"),
            spelled(b" world"),
            spelled(b"don"),
            spelled(b"'t"),
            spelled(b"1"),
            spelled(b"2"),
            spelled(b"3"),
            spelled(b"4"),
            spelled(b"!!!!"),
            spelled(b"\n\n"),
            spelled(b"a"),
            spelled(b"b"),
            spelled("\u{4e2d}\u{6587}".as_bytes()),
        ]
        .into_iter()
        .chain([END_OF_TURN.to_owned()])
        .collect(),
    );
    let source = corpus.decode("qwen35", &[], false);
    let tokenizer = Tokenizer::read(&source);
    let split = |text: &str| {
        tokenizer
            .encode(text)
            .iter()
            .map(|id| tokenizer.piece(*id))
            .collect::<Vec<Vec<u8>>>()
    };
    assert_eq!(
        split("Hello world"),
        vec![b"Hello".to_vec(), b" world".to_vec()],
    );
    assert_eq!(split("don't"), vec![b"don".to_vec(), b"'t".to_vec()]);
    assert_eq!(
        split("1234"),
        vec![b"1".to_vec(), b"2".to_vec(), b"3".to_vec(), b"4".to_vec(),],
    );
    assert_eq!(split("!!!!"), vec![b"!!!!".to_vec()]);
    assert_eq!(
        split("a\n\nb"),
        vec![b"a".to_vec(), b"\n\n".to_vec(), b"b".to_vec()],
    );
    assert_eq!(
        split("\u{4e2d}\u{6587}"),
        vec!["\u{4e2d}\u{6587}".as_bytes().to_vec()],
    );
}

#[test]
fn a_model_assembles_the_layers_the_file_weighs() {
    let corpus = Corpus::bytes();
    let source = corpus.decode(
        "qwen35",
        &Weights::model(corpus.tokens.len() as u64).tensors,
        true,
    );
    let tokenizer = Tokenizer::read(&source);
    let config = Config::read(source.metadata(), tokenizer.vocabulary());
    let graph = Graph::new();
    let model = Model::new(&graph, &source, &config, CONTEXT);
    let token = graph.input(Shape::matrix(1, 1), Element::Single);
    let position = graph.input(Shape::matrix(1, 1), Element::Single);
    let logits = model.step(&graph, token, position);
    assert_eq!(
        graph.shape(logits).dims(),
        [1, 1, 1, corpus.tokens.len() as u32],
    );
    let states = model.states();
    assert_eq!(states.len(), 2 * LAYERS as usize);
    assert_eq!(
        graph.shape(states[0]).dims(),
        [
            1,
            1,
            KERNEL as u32 - 1,
            (LINEAR_HEADS * LINEAR_WIDTH * 3) as u32
        ],
    );
    assert_eq!(
        graph.shape(states[1]).dims(),
        [
            LINEAR_HEADS as u32,
            1,
            LINEAR_WIDTH as u32,
            LINEAR_WIDTH as u32
        ],
    );
    assert_eq!(
        graph.shape(states[2]).dims(),
        [KEY_HEADS as u32, 1, CONTEXT, WIDTH as u32],
    );
    let decoder = Decoder::greedy(
        &graph,
        token,
        position,
        logits,
        states,
        tokenizer.end_of_turn(),
    );
    assert_eq!(graph.shape(decoder.chosen()).dims(), [1, 1, 1, 1]);
    assert_eq!(decoder.end(), corpus.end());
}
