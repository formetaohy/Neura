# Qwen

Runs a Qwen3.5 gguf as two characters with different personalities that take turns talking.

Run from the repository root:

```sh
cargo run --release --example qwen -- Qwen3.5-0.8B-Q8_0.gguf 8
```

The arguments are the model file and the number of replies. Ada speaks first, and Ada and Ben alternate.

Set `NEURA_FRAME_US` to advance the model within a frame of that many microseconds, for example `NEURA_FRAME_US=2000`.

Set `NEURA_RESIDENT_MB` to hold that many megabytes of weights on the device and stream the rest from host memory, for example `NEURA_RESIDENT_MB=640`.
