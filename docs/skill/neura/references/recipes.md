# Neura recipes

Every snippet below is drawn from a file that ships with the crates. The path after each one is where the full version — including the numbers it is checked against — lives. Recipes 1 to 11 were compiled and run against neura 0.4.0 before they were written down: 1 to 9 and 11 on the default backend (DX12 on Windows, with Vulkan asked for explicitly in recipe 10), so a snippet that fails on your machine is a version or a shape question, not a guess. Adapt the snippet, then compile it and run it on the machine that will own it.

## 1. Train a model end to end

`neura/examples/train.rs` is the whole program: open a device, build a graph, compile it once, then write a batch and run per step.

```rust
use neura::{AdamW, Element, Graph, Init, Mlp, Runtime, RuntimeRequest, Shape, mse_loss};

let runtime = pollster::block_on(Runtime::open(RuntimeRequest::default()))
    .expect("a device to train on");
let graph = Graph::new();
let model = Mlp::new(
    &graph,
    "model",
    &[4, 32, 32, 2],
    Init::Uniform { low: -0.25, high: 0.25 },
    Element::Single,
);
let observations = graph.input(Shape::matrix(256, 4), Element::Single);
let targets = graph.input(Shape::matrix(256, 2), Element::Single);
let prediction = model.forward(&graph, observations);
graph.retain(prediction);
let loss = mse_loss(&graph, prediction, targets);
let gradients = graph.backward(loss);
let mut optimizer = AdamW::new(&graph, "optimizer", 0.005, 0.9, 0.999, 1e-8, 0.0);
optimizer.track_all(&graph, &model.parameters());
optimizer.step(&graph, &gradients);

let weights = runtime.weights(&graph);
let program = runtime.compile(&graph, &weights);
runtime.write(&program, observations, &observation_data);
runtime.write(&program, targets, &target_data);
for step in 0..400 {
    runtime.run(&program);
    if step % 100 == 0 {
        println!("step {step}: loss {}", runtime.read(&program, loss)[0]);
    }
}
```

`graph.retain(prediction)` is what makes the prediction readable later; the loss is read back, so it may be retained too. The optimizer's `step` is part of the graph, so every `run` performs one training step.

## 2. Inference only

No `backward`, no optimizer, so `runtime.weights` and `runtime.compile` are all the setup left.

```rust
let graph = Graph::new();
let model = Mlp::new(&graph, "model", &[4, 32, 2], Init::Uniform { low: -0.1, high: 0.1 }, Element::Single);
let observation = graph.input(Shape::matrix(1, 4), Element::Single);
let action = model.forward(&graph, observation);
graph.retain(action);

let checkpoint = Checkpoint::decode(&std::fs::read("policy.safetensors")?);
let weights = runtime.load(&graph, &checkpoint);
runtime.restore(&weights, &checkpoint);
let program = runtime.compile(&graph, &weights);

runtime.write(&program, observation, &[0.0, 1.0, 0.5, 0.25]);
runtime.run(&program);
let action = runtime.read(&program, action);
```

A batch of one is a fixed shape; a batch that changes is recipe 3. Route: `neura-runtime/tests/execution.rs`, `neura-runtime/tests/persist.rs`.

## 3. A batch that changes size, one compiled program

Declare an upper bound with a free extent, attach it to the row axis (`Shape::matrix` puts rows on axis 2), and bind the live length before each run. The bound is frozen at compile time; nothing may exceed it.

```rust
let bound = 64;
let batch = graph.free(bound);
let observations = graph.input(Shape::matrix(bound, 4).freed(&[(2, batch)]), Element::Single);
let targets = graph.input(Shape::matrix(bound, 2).freed(&[(2, batch)]), Element::Single);
let prediction = model.forward(&graph, observations);
graph.retain(prediction);
let loss = mse_loss(&graph, prediction, targets);
// ... backward, optimizer, weights, compile as in recipe 1 ...

for live in [bound, 12, 1] {
    runtime.bind(&program, &[live]);
    runtime.write(&program, observations, &data(live * 4));
    runtime.write(&program, targets, &data(live * 2));
    runtime.run(&program);
    let value = runtime.read(&program, loss)[0];
}
```

Bindings are positional, in the order the graph named its free extents. An extent whose count the device computes (a ragged axis, `Graph::counted` with a device value) is *not* in that list. Route: `neura-runtime/tests/extent.rs`, `neura-nn/tests/training.rs`.

## 4. Variable-length sequences (packed, ragged)

`graph.ragged(bound, lengths)` closes variable lengths into an offsets table on the device; every packed tensor is `[1, 1, bound, width]` with that ragged extent on axis 2.

```rust
const PLANES: u32 = 4;
const BOUND: u32 = 16;
const WIDTH: u32 = 8;

let lengths = graph.input(Shape::vector(PLANES), Element::Single);
let ragged = graph.ragged(BOUND, lengths);
let packed = Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]);
let queries = graph.gradient_input(packed, Element::Single);
let keys = graph.gradient_input(packed, Element::Single);
let values = graph.gradient_input(packed, Element::Single);
let out = graph.attention(
    queries,
    keys,
    values,
    AttentionOptions {
        scale: 1.0 / (WIDTH as f32).sqrt(),
        causal: true,
        origin: None,
        segments: Some(ragged.offsets),
        reach: None,
        query_segments: None,
    },
);
graph.retain(out);
graph.retain(ragged.offsets);
```

`graph.gradient_input` is the input whose gradient you want (a packed token stream is usually one). `Graph::rows(ragged)` gives the per-row `{plane, position}` tables when you need to scatter back into per-plane tensors, and `Graph::segment_sum(value, ragged)` sums each plane. Quantities that change per batch (the lengths, not the bound) are host writes.

Writing follows the declared bound and reading follows the live extents: `runtime.write` of a packed tensor takes `bound * width` numbers, while `runtime.read` returns the rows the lengths close — lengths `[4, 3, 2, 1]` over a bound of 16 read back as `10 * width` numbers.

Route: `neura-runtime/tests/varlen.rs` carries the full forward-and-backward reference, `neura-runtime/tests/ragged.rs` the padded comparison, `neura-graph/tests/segment.rs` the per-plane sums.

## 5. A KV cache with a sliding window

`graph.resident` holds storage beside the weights that survives every run; the cache is written by the graph itself, and the cursor tells attention where the new token sits.

```rust
const PLANES: u32 = 4;      // heads times batches: the cache planes
const CAPACITY: u32 = 256;
const WIDTH: u32 = 64;
const REACH: u32 = 128;     // a sliding window; drop it for the whole cache

let keys = graph.resident(Shape::of([PLANES, 1, CAPACITY, WIDTH]), Element::Single);
let values = graph.resident(Shape::of([PLANES, 1, CAPACITY, WIDTH]), Element::Single);
let cursor = graph.input(Shape::of([PLANES, 1, 1, 1]), Element::Single);
let slot = graph.input(Shape::of([PLANES, 1, 1, 1]), Element::Single);
let row = graph.input(Shape::of([PLANES, 1, 1, WIDTH]), Element::Single);
let query = graph.input(Shape::of([PLANES, 1, 1, WIDTH]), Element::Single);
graph.write_into(keys, slot, row);
graph.write_into(values, slot, row);
let out = graph.attention(
    query,
    keys,
    values,
    AttentionOptions {
        scale: 1.0 / (WIDTH as f32).sqrt(),
        causal: true,
        origin: Some(cursor),
        segments: None,
        reach: Some(REACH),
        query_segments: None,
    },
);
graph.retain(out);
```

Each step the host writes the absolute token `position` as the cursor and `position % CAPACITY` as the slot; a window (`reach`) requires a causal mask and makes the cache a ring, while `reach: None` keeps it linear and needs no wrapping. `query_segments` is the chunked-prefill variant: a query block that carries its own offsets beside the cache's. Route: `neura-runtime/tests/decode.rs`, `neura-runtime/tests/chunked.rs`, `neura-runtime/tests/attention.rs`.

## 6. Save and load

Only named tensors — parameters and state — travel in a checkpoint, and the container is safetensors. Names come from the layer constructors (`Mlp::new(&graph, "model", ..)`) or from `named_parameter`/`named_state`.

```rust
let checkpoint = runtime.checkpoint(&weights);
std::fs::write("model.safetensors", checkpoint.bytes())?;

let bytes = std::fs::read("model.safetensors")?;
let checkpoint = Checkpoint::decode(&bytes);
let weights = runtime.load(&graph, &checkpoint);
runtime.restore(&weights, &checkpoint);
let program = runtime.compile(&graph, &weights);
```

`runtime.restore(&weights, &checkpoint)` on a store you already hold is the way to roll back to a baseline, optimizer state included. A rebuilt graph with the same parameter layout can reuse a store through `runtime.rebind(&weights, &rebuilt_graph)`. Route: `neura-runtime/tests/persist.rs`, `neura-runtime/tests/safetensors.rs`, `neura-runtime/tests/rebind.rs`, `neura-nn/tests/state.rs`.

## 7. Half precision and quantized weights

Everything is declared per tensor, and the host still speaks `f32`.

```rust
let model = Mlp::new(&graph, "model", &[4, 32, 2], init, Element::Half);
let dense = Linear::quantized(&graph, "dense", 32, 8, quantum, init, Element::Single);
let blocks = Linear::block_quantized(&graph, "blocks", 32, 8, init, Element::Single, Element::Int4);
```

`Element::Half` and `Bfloat16` are plain storage formats; `Int8`, `Int4`, `Fp8E4M3`, `Fp8E5M2` and `Fp4E2M1` are quantized, carrying a scale per tensor or a scale table per block (`neura-abi/src/element.rs` names the block sizes). `program.span(value)` reports the payload and the table a tensor occupies. Route: `neura-nn/tests/precision.rs`, `neura-runtime/tests/precision.rs`, `neura-graph/tests/quantization.rs`.

## 8. Overlap readback with the next step

Two readback slots exist by default; a `Readout` holds the run it was pulled from, so the host can start the next step before touching the numbers.

```rust
runtime.run(&program);
let first = runtime.pull(&program, &[loss]);
runtime.write(&program, observations, &next_batch);
runtime.run(&program);
let previous = runtime.collect(first);          // the first step's loss
let current = runtime.read(&program, loss);     // the second step's
```

`runtime.read` is `pull` plus `collect` in one call; use it everywhere except the step where latency matters. Route: `neura-runtime/tests/execution.rs`.

## 9. Write your own optimizer

A step is one expression per parameter, authored after `backward`.

```rust
let gradients = graph.backward(loss);
let descent = graph.fill(Shape::scalar(), -0.05);
for parameter in model.parameters() {
    graph.add_into(parameter, graph.mul(gradients.of(parameter), descent));
}
```

`add_into`, `mul_into`, `copy_into`, `write_into` and `scatter_into` are the in-place writers; `graph.clip`-style tricks go through `gradients.clip(&graph, threshold)`. The built-in `Sgd` and `AdamW` are written this way, `AdamW` keeping its moments in `named_state`. Route: `neura-runtime/tests/rebind.rs`, `neura-nn/tests/optimizer.rs`.

## 10. Pick a backend, an adapter or a heap size

```rust
use neura::{AdapterPolicy, Backends, GpuRequest, PowerPreference};

let request = RuntimeRequest {
    gpu: GpuRequest {
        backends: Backends::VULKAN,
        adapter: AdapterPolicy::Power(PowerPreference::LowPower),
        artifacts: Some(std::path::PathBuf::from(".neura-artifacts")),
        ..GpuRequest::default()
    },
    heap_bytes: 256 << 20,
    ..RuntimeRequest::default()
};
let runtime = pollster::block_on(Runtime::open(request))?;
println!("{}", runtime.context().adapter_info().name);
```

`Backends` combines `VULKAN`, `METAL` and `DX12`; `GpuRequest::default()` requests the high-performance adapter with the adapter's own limits. `artifacts` caches compiled device programs across runs. `neura-gpu` also exports `LimitsPolicy`, `Limits`, `Capability`, `CooperativeMatrix` and the raw `Device`/`Queue` if you need them directly. The heap is fixed at open and holds every tensor of a plan, so size it from `program.heap_bytes()`/`program.tensor_bytes()` rather than by guessing. Route: `neura-runtime/tests/memory.rs`, `neura-gpu/tests/adapter.rs`.

## 11. Read what a plan decided

```rust
println!(
    "{} tasks in {} waves, {} workgroups, {} tensors; {} bytes of tensors, {} of weights, {} of heap",
    program.task_count(),
    program.wave_count(),
    program.workgroups(),
    program.value_count(),
    program.tensor_bytes(),
    program.weights().bytes(),
    program.heap_bytes(),
);
```

`program.tiles()` and `program.matmul_geometries()` show the matrix-multiply shapes the planner chose; `runtime.tune(&graph, &weights)` measures candidates and returns a program, and `runtime.measure(&program)` reports seconds per run. Route: `neura-runtime/tests/timing.rs`, `neura-runtime/tests/cooperative.rs`.

## 12. Add a device operation

Five places, in order, and one proof:

1. `neura-abi/src/kind.rs` — add the op to the `kinds!` table (entry point, device modules, geometry, prelude/chain, whether it reads a cursor).
2. `neura-kernel/src/task/<name>.rs` — the device body, registered in `install` in `neura-kernel/src/lib.rs`; `neura-macro` gives `#[module]`, `#[kernel]` and `expression!`, and `BINDINGS` declares the buffers. Copy the task nearest to yours and change it.
3. `neura-plan/src/lower.rs` — the task grid; `neura-plan/src/region.rs` — what the task reads and writes; `neura-plan/src/encode.rs` — the invariant that keeps the plan honest.
4. `neura-graph/src/op.rs` — the graph-level entry point; `neura-graph/src/autodiff.rs` — its gradient.
5. `neura-kernel/tests/kernel.rs` walks every kind for every element; add a runtime test beside the feature with numbers from a plain-Rust reference.

A pure elementwise op needs no new kind: `neura-pointwise/src/op.rs` holds the vocabulary and the gradient formulas, and `neura-precision/src/lib.rs` holds storage formats.

## 13. Prove the numbers

A program with a free extent runs every binding, so one test at the bound proves nothing. Build the same computation with fixed shapes, and compare at the bound, a middle length and one row.

```rust
fn reference(input: &[f32], rows: u32) -> Vec<f32> { /* plain Rust */ }

for live in [bound, 7, 1] {
    runtime.bind(&program, &[live]);
    runtime.write(&program, input, &data(live * 4));
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &reference(&data(live * 4), live), 1e-5);
}
```

Tolerances used across the framework's tests: `1e-6` for f32 against f32, `1e-5` to `1e-3` for gradient checks, `1e-2` for half precision. `neura-runtime/tests/support/mod.rs` holds the `assert_close` they share.
