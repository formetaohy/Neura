# QWEN

Runs a Qwen3.5 gguf through Neura, printing the answer to a prompt one token at a time.

```sh
cargo run --release --example qwen -- <model.gguf> "What is the capital of France?" 48
```

The arguments are the model file, the prompt, and how many tokens to generate (48 by default). A
prompt that begins with `raw:` reaches the model without the chat template around it; every other
prompt is wrapped in the template the gguf holds, its empty think block included. Tokens are chosen
greedily, and generation ends at the end of turn or at the token count.

`Qwen3.5-0.8B-Q4_K_M.gguf` and the other mainstream quants of that model (Q4_K_M, Q5_K_M, Q6_K,
Q8_0) load as they are, without repacking on the host. The example runs a context of `min(1024, the
model's own)` and builds the model once over one store as a family of two programs: a fill program
that walks eight rows of a chunk at a time, and an answer program that walks one row and chooses
the token that follows it, sharing the weights, the kv caches, the convolution states and the
gated delta net states every row advances. A prompt walks through the fill program while more
than a chunk of it remains; the rows it leaves over, together with every token it answers with,
walk through the answer program one row at a time, so the choice the generation follows always
stands in the one place that program writes it.
