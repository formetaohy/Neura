use neura::gguf::Metadata;

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
}

impl Config {
    pub fn read(metadata: Metadata<'_>, vocabulary: usize) -> Self {
        let arch = metadata.text("general.architecture");
        assert_eq!(
            arch, "qwen35",
            "this model weighs the tensors of the qwen35 architecture, and this gguf names {arch} as its general.architecture",
        );
        let key = |suffix: &str| format!("{arch}.{suffix}");
        let linear_heads = metadata.word(&key("ssm.group_count"));
        let linear_width = metadata.word(&key("ssm.state_size"));
        let value_heads = metadata.word(&key("ssm.time_step_rank"));
        assert_eq!(
            value_heads, linear_heads,
            "this gguf weighs {value_heads} value heads against {linear_heads} key heads, and this model holds one value head beside every key head",
        );
        let inner = metadata.word(&key("ssm.inner_size"));
        assert_eq!(
            inner,
            linear_heads * linear_width,
            "this gguf spans {inner} numbers of linear attention where {linear_heads} heads of {linear_width} span {}",
            linear_heads * linear_width,
        );
        Self {
            layers: metadata.word(&key("block_count")),
            hidden: metadata.word(&key("embedding_length")),
            feed_forward: metadata.word(&key("feed_forward_length")),
            heads: metadata.word(&key("attention.head_count")),
            key_heads: metadata.word(&key("attention.head_count_kv")),
            head_width: metadata.word(&key("attention.key_length")),
            rope_base: metadata.single(&key("rope.freq_base")),
            rope_width: metadata.word(&key("rope.dimension_count")),
            interval: metadata.word(&key("full_attention_interval")),
            linear_heads,
            linear_width,
            conv_kernel: metadata.word(&key("ssm.conv_kernel")),
            vocabulary: u32::try_from(vocabulary)
                .expect("a vocabulary of tokens fits the numbers a token id holds"),
            context: metadata.word(&key("context_length")),
            floor: metadata.single(&key("attention.layer_norm_rms_epsilon")),
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
