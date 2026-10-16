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
model's own)` and builds the model once over one store as a family of five programs: fill programs
that walk 256, 128, 32 or 8 rows of a chunk at a time, and an answer program that walks one row and
chooses the token that follows it, sharing the weights, the kv caches, the convolution states and
the gated delta net states every row advances. A prompt walks through the largest fill program the
rows it has left admit, then the next largest, and so on; the rows it leaves over, together with
every token it answers with, walk through the answer program one row at a time, so the choice the
generation follows always stands in the one place that program writes it.

Every run compiles the plan this device measured for the programs, and a device that has
measured none walks the profile it balances on. Setting `NEURA_TUNE=1` measures every plan the
device offers against the weights the programs walk, keeps the fastest of them, and writes the
choice into the artifact cache, so the run that measured it and every run after it walk the
measured plan.

```sh
NEURA_TUNE=1 cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf "What is the capital of France?" 48
```
