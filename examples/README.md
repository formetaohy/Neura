# Examples

These examples demonstrate training and performance measurement with Neura. Run the commands below from the repository root with Rust installed and a device supported by Neura's native graphics backends available.

| Example | Purpose |
| --- | --- |
| [train.rs](train.rs) | Train an MLP to map four synthetic observations to two actions using mean squared error and AdamW. |
| [benchmark.rs](benchmark.rs) | Measure elementwise operations, MLP training and inference, and matrix multiplication; compare execution profiles and device-tuned selection. |

## Training

```sh
cargo run --release --example train
```

Builds a `4 → 32 → 32 → 2` MLP and trains it on 256 synthetic samples for 400 steps. The graph includes the forward pass, automatic differentiation, and optimizer updates, and is compiled once for repeated execution.

The output includes loss values, task and wave counts, tensor and weight memory usage, host-side timing, and predicted actions alongside their targets.

## Benchmark

```sh
cargo run --release --example benchmark
```

Measures:

- Elementwise multiplication followed by ReLU over 262,144 elements.
- MLP training at different batch sizes and network depths.
- Batched MLP inference with action selection via `argmax`.
- Matrix multiplication with mixed shapes.
- A `1024 × 1024 × 1024` matrix product across the device's execution profiles, followed by `Runtime::tune`.

The output includes device information, workgroup sizes, matrix multiplication tiles, task and wave counts, host-side submission time, device execution time, and matrix multiplication throughput. Measurements use warmup runs and report averages; results depend on the device and its current load and clock state.
