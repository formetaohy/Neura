# Qwen

Runs a Qwen3.5 gguf as two characters with different personalities and prints their conversation as it happens.

Run from the repository root:

```sh
cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf 8
```

The arguments are the model file and the number of replies. Ada opens the conversation, and Ada and Ben alternate for the requested number of replies. Their personalities live only in the system prompts, so both characters share one set of weights and no extra weights are loaded.

Each character keeps its own kv cache, convolution states and gated delta net states. After a reply the example suspends that character, and the next turn resumes it with only the other character's new line, so earlier turns are not prefilled again. When a character's context fills, the example drops its oldest exchange and restarts that plane with the shortened transcript.

The example runs 1024 tokens of context or the model's own context if smaller, fills prompts in chunks of 32 and 8, and stops a reply at the end of turn or after 64 tokens.
