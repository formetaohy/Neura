# Benchmark

Measure elementwise operations, MLP training and inference, and matrix multiplication; compare execution profiles and device-tuned selection.

Run from the repository root:

```sh
cargo run --release --example benchmark
```

The program measures:

- Elementwise multiplication followed by ReLU over 262,144 elements.
- MLP training at different batch sizes and network depths.
- Batched MLP inference with action selection via `argmax`.
- Matrix multiplication with mixed shapes.
- A `1024 × 1024 × 1024` matrix product across the device's execution profiles, followed by `Runtime::tune`.

The output includes device information, workgroup sizes, matrix multiplication tiles, task and wave counts, host-side submission time, device execution time, and matrix multiplication throughput. Measurements use warmup runs and report averages; results depend on the device and its current load and clock state.
