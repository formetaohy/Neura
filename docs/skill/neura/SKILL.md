---
name: neura
description: Write, train, run and extend Rust code against the Neura GPU deep-learning framework (crates `neura`, `neura-graph`, `neura-nn`, `neura-runtime`, `neura-gpu`, `neura-abi`, `neura-plan`). Use when a project depends on neura and you need its API, a working training or inference loop, dynamic batch or token counts, variable-length (ragged) sequences, attention with a KV cache, checkpoints, mixed precision or quantized weights, or a new device operation.
---

# Neura

Neura is a GPU-driven deep-learning framework for games: DX12, Vulkan and Metal behind one Rust API, no Python and no autograd tape at run time.

The crates carry no comments and no doc comments by design. Names, types, tests and this skill are the whole contract. Never guess an API: every name you write must come from a `lib.rs` you have read or a file you are looking at. If your search finds no name, it does not exist — say so instead of inventing one.

This skill is written for the 0.4 series. Read the version out of the `Cargo.toml` you locate in step 1 and quote it whenever behaviour surprises you.

## Step 1 — locate the dependency source on this machine

Every published crate ships its source, tests included. Find it before writing anything:

```bash
cargo metadata --format-version 1 | grep -o '"manifest_path":"[^"]*neura[^"]*"'
```

- The output is JSON, so on Windows every separator arrives as `\\`; unescape before opening a path.
- PowerShell: `cargo metadata --format-version 1 | ConvertFrom-Json | % packages | ? { $_.name -like 'neura*' } | % manifest_path`
- The directory beside that `Cargo.toml` is the crate root: read `<root>/src/**`, `<root>/tests/*.rs`, and `<root>/examples/*.rs` on the `neura` crate.
- The registry copy is read-only and pinned to one version. `cargo tree -i neura` and `cargo tree -p neura-graph` show which versions are in play.
- If your tools cannot read outside the project directory, run `cargo vendor` once and read `vendor/neura-*/` instead.

## Step 2 — read the public surface first (about 140 lines)

Nine `lib.rs` files list every public name a consumer can touch. Read them before anything else; the facade tells you what you may name, the owning crate is where the behaviour lives.

| crate | holds |
| --- | --- |
| `neura` | the facade, `src/lib.rs`; it re-exports most of what you need, start here |
| `neura-graph` | `Graph`, `Value`, `Shape`, `Window`, `Pool`, `Init`, `AttentionOptions`, `Gradients`, `Ragged`, `Rows`, `Prefixes`, `Compacted`, `Residency`, `TaskInfo`, `ValueInfo`, `GraphSnapshot`, `GraphStamp`, `Revision` |
| `neura-nn` | layers `Linear`, `Conv2d`, `Embedding`, `LayerNorm`, `RmsNorm`, `GroupNorm`, `Mlp`, `MultiHeadAttention`, `Adapter`, `HeadShape`; losses `mse_loss`, `cross_entropy`, `policy_loss`; optimizers `Sgd`, `AdamW`, `Moments` |
| `neura-runtime` | `Runtime`, `RuntimeRequest`, `Program`, `Weights`, `Checkpoint`, `Readout`, `Run`, `Span`, `Placement`, `Budget`, `Profile` |
| `neura-plan` | `Layout`, `Region`, `Seed`, and the invariants of a plan in `src/encode.rs` |
| `neura-abi` | `Element`, `Store`, the `Kind` vocabulary (`KINDS`), `DeviceModule`, `Geometry`, the device record tables, `MAX_RANK`, `EXACT_WALK_LIMIT` |
| `neura-gpu` | `Device`, `Queue`, `GpuContext`, `GpuRequest`, `GpuUnavailable`, `AdapterInfo`, `AdapterPolicy`, `AdapterId`, `Backend`, `Backends`, `PowerPreference`, `DeviceType`, `Limits`, `LimitsPolicy`, `Capability`, `GpuBuffer`, `PipelineHandle`, `BindGroup`, `Submission`, `Readback`, `ArtifactCache` |
| `neura-kernel` | one device body per `Kind`, and the binding table |
| `neura-shader` | `ComputeProgram`, `Module`, and the HLSL / MSL / SPIR-V writers |

## Step 3 — the five nouns and the order they are built

`Graph<'g>` builds a value graph; `Value<'g>` is one tensor node of it. `Graph::backward` derives a `Gradients<'g>` and appends the backward tasks to the same graph. `Runtime` owns the device, the heap and the readback ring. `Runtime::compile` freezes the current graph revision into a `Program` that runs the whole thing — forward, backward and the optimizer's in-place updates — on every `Runtime::run`.

The canonical shape of a program:

```rust
use neura::{AdamW, Element, Graph, Init, Mlp, Runtime, RuntimeRequest, Shape, mse_loss};

fn main() {
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
    let (observation_data, target_data) = session(256);
    runtime.write(&program, observations, &observation_data);
    runtime.write(&program, targets, &target_data);
    for step in 0..400 {
        runtime.run(&program);
        if step % 100 == 0 {
            println!("step {step}: loss {}", runtime.read(&program, loss)[0]);
        }
    }
}
```

`session` is the data function that `examples/train.rs` of the `neura` crate carries beside this program. The rules that order the code, all enforced by panics:

1. Author the whole graph — inputs, parameters, forward, `backward`, optimizer step — then compile once. `backward` runs once per graph, and no leaf may be updated in place before it.
2. `graph.retain(value)` every tensor the host reads back. A derived tensor nothing reads afterwards has its storage reused by a later task of the plan, and `Runtime::read` refuses it.
3. `input` holds host-written storage, `parameter` and the layer constructors hold weights, `state` holds training state, `resident` holds read-only storage beside the weights, everything else is derived. Gradients reach parameters and state, never inputs.
4. `Element::Single` is `f32`. Pass `Element::Half` (or a quantized element) when throughput or size matters; `Runtime::write` and `Runtime::read` always speak `f32` and pack per element and scale.

## Step 4 — recipes

`references/recipes.md` holds verified snippets for training, forward-only inference, dynamic batch or token counts, variable-length attention, a KV cache, checkpoints, mixed precision, pipelined readback, a hand-written optimizer, backend selection, and adding a device operation. Take the one that matches the task and adapt it. Do not write an equivalent from memory: the tests beside it are the reference numbers.

## Step 5 — traps

`references/traps.md` lists the ways a Neura program compiles, runs, and still reports wrong numbers or panics late, each with the test file on this machine that pins it. Read it before claiming success on anything involving dynamic extents, attention masks, in-place updates, checkpoints or quantized elements.

## Step 6 — where to look for behaviour

| question | read |
| --- | --- |
| what a layer computes | `<root>/src/layer.rs` of `neura-nn` |
| what an op asserts about shapes | `<root>/src/op.rs` of `neura-graph` |
| what the device does for a kind | `<root>/src/task/<name>.rs` of `neura-kernel`, routed by `Kind::entry` in `neura-abi/src/kind.rs` |
| which numbers are expected | `<root>/tests/*.rs`; the file name names the feature family |
| how a plan lays a walk out | `<root>/src/lower.rs` of `neura-plan` |
| what is checked before a program runs | `<root>/src/encode.rs` of `neura-plan` |
| what is asserted at run time | the `assert!` messages in `neura-runtime/src/runtime/mod.rs` and `neura-graph/src/graph.rs` |

Feature families worth knowing by name, all under `neura-runtime/tests/` unless noted: `execution.rs`, `memory.rs`, `reuse.rs`, `view.rs`, `extent.rs`, `rebind.rs`, `revision.rs`, `attention.rs`, `decode.rs`, `ragged.rs`, `varlen.rs`, `chunked.rs`, `rope.rs`, `segment.rs`, `moe.rs`, `convolution.rs`, `pooling.rs`, `predicate.rs`, `precision.rs`, `persist.rs`, `safetensors.rs`, `interop.rs`, `timing.rs`, plus `neura-nn/tests/{training,gradient,optimizer,state,normalization,attention,precision,adapter}.rs` and `neura-graph/tests/{shape,op,gradient,freeze,quantization,prefix,window,segment,grouped}.rs`.

## Step 7 — extending the framework

Adding an operation touches five places, in this order; read each one before changing it:

1. Vocabulary — `neura-abi/src/kind.rs`, the `kinds!` table: one row per op with its entry point, device modules, geometry, prelude/chain flags and whether it reads a cursor. Nothing else knows an op exists until it is here.
2. Device body — `neura-kernel/src/task/<name>.rs`, registered in the `install` function of `neura-kernel/src/lib.rs`. `neura-macro` provides the `#[module]`, `#[kernel]` and `expression!` authoring forms used there; read an existing task beside the one you add.
3. Planning — `neura-plan/src/lower.rs` for the task grid, `neura-plan/src/region.rs` for the read and write regions each task touches, `neura-plan/src/encode.rs` for the invariant that keeps the plan honest.
4. Gradient — `neura-graph/src/autodiff.rs`, and `neura-graph/src/op.rs` for the graph-level entry point.
5. Proof — `neura-kernel/tests/kernel.rs` walks every kind for every element; the runtime family test beside the feature proves the numbers. Do not add a kind without a test that pins its numbers.

Pure elementwise work does not need a new kind: the vocabulary and the gradient formulas of a pointwise operation live in `neura-pointwise/src/op.rs`, and storage formats live in `neura-precision/src/lib.rs`.

## Portable use

`SKILL.md` is the entry point; `references/*.md` are plain markdown. Copy this whole directory into a project's `.agents/skills/` (or any skill directory) to make the skill discoverable, and point a tool that does not understand skills at `SKILL.md` first.
