# QWEN

Runs a Qwen3.5 gguf through Neura, printing the answer to a prompt one token at a time. Every
model-specific piece — the metadata it reads, the tensor names it declares, the `qwen35`
pre-tokenizer and its byte-level merges, gated delta net, the full attention layers — lives in this
example; the framework only offers the graph, the formats and the loader.

```sh
cargo run --release --example qwen -- <model.gguf> "What is the capital of France?" 48
```

The argument list is the model file, the prompt, and how many tokens to generate (48 by default).
A prompt that begins with `raw:` is handed to the model without the chat template around it; every
other prompt is wrapped in the template the gguf names, its empty think block included. Tokens are
chosen greedily, and generation ends at the end of turn or at the token count.

`Qwen3.5-0.8B-Q4_K_M.gguf` and the other mainstream quants of that model (Q4_K_M, Q5_K_M, Q6_K,
Q8_0) load as they are: the example declares every weight in the layout the file holds, so nothing
is repacked on the host.

This is a demonstration of the framework, not a serving stack: 24 layers run as one single-token
step graph whose kv caches, convolution states and gated delta net states stay resident on the
device, so generation costs one submission per token.
