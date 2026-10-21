# Qwen

Runs a Qwen3.5 gguf as two characters with different personalities that take turns talking.

Run from the repository root:

```sh
cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf 8
```

The arguments are the model file and the number of replies. Ada speaks first, and Ada and Ben alternate.
