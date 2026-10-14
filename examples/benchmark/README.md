# Benchmark

Measures every device path the framework walks: scheduler fixed costs, element, decision, movement, ragged, narrow and wide products, attention, vision, and training steps, each with plan structure, submit and step timing, and a derived rate.

Run from the repository root:

```sh
cargo run --release --example benchmark
```

Run one section by naming it:

```sh
cargo run --release --example benchmark -- attention
```

The first run compiles each section through the platform shader compiler; the artifact cache serves every later run.
