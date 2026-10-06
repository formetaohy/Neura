# Train

Train an MLP to map four synthetic observations to two actions using mean squared error and AdamW.

Run from the repository root:

```sh
cargo run --release --example train
```

The program builds a `4 → 32 → 32 → 2` MLP and trains it on 256 synthetic samples for 400 steps. The graph includes the forward pass, automatic differentiation, and optimizer updates, and is compiled once for repeated execution.

The output includes loss values, task and wave counts, tensor and weight memory usage, host-side timing, and predicted actions alongside their targets.
