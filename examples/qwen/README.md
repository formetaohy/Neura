# QWEN

Runs a Qwen3.5 gguf through Neura, printing the answer to a prompt one token at a time.

```sh
cargo run --release --example qwen -- <model.gguf> "What is the capital of France?" 48
```

The arguments are the model file, the prompt, how many tokens to generate (48 by default), and how
many sequences answer the prompt together (1 by default). A prompt that begins with `raw:` reaches
the model without the chat template around it; every other prompt is wrapped in the template the
gguf holds, its empty think block included. Tokens are chosen greedily, and generation ends at the
end of turn or at the token count.

```sh
cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf "What is the capital of France?" 48 8
```

`Qwen3.5-0.8B-Q4_K_M.gguf` and the other mainstream quants of that model (Q4_K_M, Q5_K_M, Q6_K,
Q8_0) load as they are, without repacking on the host. The example runs a context of `min(1024, the
model's own)` and builds one answer program whose `batch` planes each hold a sequence: the weights
are shared by every plane, while the kv caches, the convolution states and the gated delta net
states advance one plane at a time. Each plane carries its own cursor and mask, so a prompt is
prefilled token by token as the other planes answer their own tokens. Admitting a sequence runs a
small reset program that zeroes the kv cache, the convolution state and the gated delta net state
of its plane before the first prompt token; the other planes keep their state.

Every run compiles the plan this device measured for the answer and reset programs, and a device
that has measured none walks the profile it balances on. Setting `NEURA_TUNE=1` measures every plan
the device offers against the weights the programs walk, keeps the fastest of them, and writes the
choice into the artifact cache, so the run that measured it and every run after it walk the
measured plan.

```sh
NEURA_TUNE=1 cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf "What is the capital of France?" 48
```
