use neura::gguf::Value as Held;
use std::collections::BTreeMap;

pub struct Config {
    pub layers: u32,
    pub hidden: u32,
    pub feed_forward: u32,
    pub heads: u32,
    pub key_heads: u32,
    pub head_width: u32,
    pub rope_base: f32,
    pub rope_width: u32,
    pub interval: u32,
    pub linear_heads: u32,
    pub linear_width: u32,
    pub conv_kernel: u32,
    pub vocabulary: u32,
    pub context: u32,
    pub floor: f32,
    pub end: u32,
}

impl Config {
    pub fn read(metadata: &BTreeMap<String, Held>, vocabulary: usize) -> Self {
        let arch = text(metadata, "general.architecture");
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let linear_heads = integer(metadata, &key("ssm.group_count"));
        let linear_width = integer(metadata, &key("ssm.state_size"));
        let value_heads = integer(metadata, &key("ssm.time_step_rank"));
        assert_eq!(
            value_heads, linear_heads,
            "this gguf weighs {value_heads} value heads against {linear_heads} key heads, and this model holds one value head beside every key head",
        );
        let inner = integer(metadata, &key("ssm.inner_size"));
        assert_eq!(
            inner,
            linear_heads * linear_width,
            "this gguf spans {inner} numbers of linear attention where {linear_heads} heads of {linear_width} span {}",
            linear_heads * linear_width,
        );
        let end = metadata
            .get("tokenizer.ggml.eos_token_id")
            .and_then(Held::integer)
            .map_or_else(
                || {
                    array(metadata, "tokenizer.ggml.tokens")
                        .iter()
                        .position(|token| token.text() == Some("<|im_end|>"))
                        .unwrap_or_else(|| {
                            panic!("this gguf names no end of turn, and holds no <|im_end|>")
                        }) as u32
                },
                |end| end as u32,
            );
        Self {
            layers: integer(metadata, &key("block_count")),
            hidden: integer(metadata, &key("embedding_length")),
            feed_forward: integer(metadata, &key("feed_forward_length")),
            heads: integer(metadata, &key("attention.head_count")),
            key_heads: integer(metadata, &key("attention.head_count_kv")),
            head_width: integer(metadata, &key("attention.key_length")),
            rope_base: number(metadata, &key("rope.freq_base")),
            rope_width: integer(metadata, &key("rope.dimension_count")),
            interval: integer(metadata, &key("full_attention_interval")),
            linear_heads,
            linear_width,
            conv_kernel: integer(metadata, &key("ssm.conv_kernel")),
            vocabulary: vocabulary as u32,
            context: integer(metadata, &key("context_length")),
            floor: number(metadata, &key("attention.layer_norm_rms_epsilon")),
            end,
        }
    }

    pub fn attends(&self, layer: u32) -> bool {
        (layer + 1).is_multiple_of(self.interval)
    }

    pub fn linear_width_total(&self) -> u32 {
        self.linear_heads * self.linear_width
    }

    pub fn mixed_width(&self) -> u32 {
        self.linear_width_total() * 3
    }
}

pub fn text(metadata: &BTreeMap<String, Held>, key: &str) -> String {
    metadata
        .get(key)
        .and_then(Held::text)
        .unwrap_or_else(|| panic!("this gguf holds no {key}"))
        .to_owned()
}

fn integer(metadata: &BTreeMap<String, Held>, key: &str) -> u32 {
    metadata
        .get(key)
        .and_then(Held::integer)
        .unwrap_or_else(|| panic!("this gguf holds no {key}")) as u32
}

fn number(metadata: &BTreeMap<String, Held>, key: &str) -> f32 {
    metadata
        .get(key)
        .and_then(Held::number)
        .unwrap_or_else(|| panic!("this gguf holds no {key}")) as f32
}

fn array<'v>(metadata: &'v BTreeMap<String, Held>, key: &str) -> &'v [Held] {
    metadata
        .get(key)
        .and_then(Held::array)
        .unwrap_or_else(|| panic!("this gguf holds no {key}"))
}
