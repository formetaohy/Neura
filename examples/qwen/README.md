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
Q8_0) load as they are, without repacking on the host. The example runs 1024 tokens of context by
default (`NEURA_CONTEXT` names another length, capped at the model's own) and builds one answer
program whose `batch` planes each hold a sequence: the weights
are shared by every plane, while the kv caches, the convolution states and the gated delta net
states advance one plane at a time. Each plane carries its own cursor and mask. A run of one
sequence fills a prompt in the largest chunk of the ladder `32, 8` that leaves a token for the
answer, and every plane of a batch fills its own prompt in the same ladder; a plane whose prompt
no longer fills a chunk walks its remaining tokens one at a time. Admitting a sequence
runs a small reset program that zeroes the kv cache, the convolution state and the gated delta net
state of its plane before the first prompt token; the other planes keep their state.

Setting `NEURA_TURNS` to user prompts separated by `|` answers them as one conversation. The first
turn admits its whole rendered prompt; every later turn suspends the plane after its answer and
resumes it with only the rendered user message it appends, so the kv cache, convolution states and
gated delta net states of the earlier turns stay in place instead of being prefilled again. When
the next turn no longer fits the context, the example drops the oldest user and assistant messages
and restarts the plane with the shortened conversation; a restart zeroes the plane and prefills the
whole shortened prompt. A conversation answers one sequence at a time and does not combine with
`NEURA_PREFIX`.

```sh
NEURA_TURNS="What is the capital of France?|What river runs through it?|How many people live there?" cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf ignored 64
```

Setting `NEURA_PREFIX=1` fills the prompt except its last token once, on a prefix plane, and every
sequence of the batch copies that plane's kv cache, convolution states and gated delta net states
before it walks its own last token. The copy is a state copy, not a replay: the sequences answer the
same tokens they answer without the shared prefix, sampled runs included, while the prompt is
computed once instead of once a plane.

```sh
NEURA_PREFIX=1 cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf "What is the capital of France?" 48 8
```

Every full-attention layer keeps its keys and values in a ring whose window is `NEURA_WINDOW`
tokens; unset, the window is the whole context. A cursor past the window overwrites the oldest
slots, so attention reads the last window of tokens while the gated delta net layers keep their
recurrent state, and the kv cache stops growing with the conversation. A window at least as long
as the context is the exact full attention the model was trained with; a shorter window is a
sliding window and answers may differ. `NEURA_WINDOW` therefore lets a long conversation continue
past the memory a full cache would need, while `NEURA_CONTEXT` still bounds each sequence.

```sh
NEURA_CONTEXT=8192 NEURA_WINDOW=1024 cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf "What is the capital of France?" 48 8
```

Setting `NEURA_KV=half` stores the keys and values of every full-attention layer in half precision
and keeps the convolution and gated delta net states at single precision. The cache takes half the
memory and every attention step reads half the bytes. An 8192-token context of eight sequences does
not fit the default 2GB heap in single precision and fits at half, so a long conversation can keep
exact full attention instead of a sliding window. Attention answers from rounded keys and values,
and a long run may choose a token a single precision run does not.

```sh
NEURA_KV=half NEURA_CONTEXT=8192 cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf "What is the capital of France?" 48 8
```

Every run compiles the plan this device measured for the answer, fill and reset programs, and a
device that has measured none walks the profile it balances on. Setting `NEURA_TUNE=1` measures
every plan the device offers against the weights the programs walk, keeps the fastest of them, and
writes the choice into the artifact cache, so the run that measured it and every run after it walk
the measured plan.

```sh
NEURA_TUNE=1 cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf "What is the capital of France?" 48
```
