# Neura concepts

Read after locating the crate source (`SKILL.md`, step 1). Every claim below is observable in that source; the file names are relative to each crate root.

## One graph, one revision

`Graph<'g>` is a builder and a task list. Every operation call appends tasks and returns a `Value<'g>` naming the output.

- `Value::id()`, `Value::shape()`, and on the graph: `shape`, `element`, `scale`, `trains`, `carries_gradient`, `name_of`, `value_count`, `task_count`.
- `graph.stamp()` and `graph.revision()` answer "has this graph changed"; `Revision::is_current()` and `Program::is_current()` answer it for a compiled program. `Runtime::run`, `bind`, `write` and `read` all assert the program is still current, so author first and compile after. A rebuild plus `runtime.rebind` is what keeps a weight store alive across graphs.
- Storage has one of six identities (`Residency`):
  - `Input` — host-written tensors (`graph.input`), read every run.
  - `Parameter` — weights (`graph.parameter`, `graph.named_parameter`, the layer constructors). `graph.freeze(&[..])` takes one out of training, and must run before anything reads it.
  - `State` — training state (`graph.state`, `graph.named_state`), the place optimizer moments and counters live.
  - `Resident` — storage beside the weights that no gradient reaches (`graph.resident`), the pattern a KV cache uses.
  - `Derived` — every op output.
  - `View` — an alias of another tensor's storage (`permute`, `reshape`, `slice`, `trim`, `detach`).
- `graph.retain(value)` marks the storage owner as un-reusable, which is what makes a tensor readable after a run. `Graph::detach` cuts a value out of autodiff; `Graph::trim(value, axis, count)` returns a view of the first `count` numbers of an axis; `Graph::length(value, axis)` returns the live length of an axis as a scalar value the device reads.

## Shape algebra

- Four axes at most (`MAX_RANK`), right-aligned: `Shape::of([rows, columns])` is `[1, 1, rows, columns]`, `Shape::matrix(rows, columns)` and `Shape::vector(n)` are the same thing spelled shorter, `Shape::scalar()` is `[1, 1, 1, 1]`.
- `dims()`, `elements()`, `rows()` = `dims[0] * dims[1] * dims[2]`, `columns()` = `dims[3]`, `strides()`, `is_scalar()`, `free(axis)`, `dynamic()`.
- Dimensions must be positive, and a tensor may not exceed the index space the device addresses (the assert in `Shape::from_axes` names the ceiling).
- Free extents make one program serve many lengths: `graph.free(bound)` names a slot with an upper bound, `Shape::freed(&[(axis, free)])` attaches it to a shape (the declared dimension must equal the bound), `graph.counted(bound, count)` also names a device-authored count, and `graph.author(free, count)` attaches one later.
- `Runtime::bind(&program, &extents)` sets the live length of each free extent, in the order the graph named them. A binding only ever shortens: exceeding the bound panics, and writes are checked against the live length. `Graph::length(value, axis)` reads a live length back as a value on the device.
- A static operand beside a free extent is legal when its dimension is `1` or the bound; anything else is refused while the graph is built. This is why a normalization with a dynamic width works and a mispaired table does not.
- Ragged axes pack variable-length rows: `graph.ragged(bound, lengths)` takes `lengths` as `[planes]` (a flat vector, or `[heads, batch]`) and returns `Ragged { extent, offsets }`, where `offsets` is a `[planes + 1, 1]` table closed on the device by a prefix sum and `extent` is a counted free extent over the packed rows. `graph.rows(ragged)` turns it into per-row `{plane, position}` tables; `graph.prefix_sum` and `graph.compact(mask)` are the neighbouring primitives. A tensor that walks a ragged extent is packed: rank 4, `[1, 1, bound, width]`, with the ragged extent on its second-to-last axis. Everything else about a ragged axis is refused while the graph is built — see `neura-graph/tests/ragged.rs`, `neura-graph/tests/segment.rs`.

## Autodiff

- `graph.backward(loss)` returns `Gradients<'g>`. `gradients.of(value)` panics for a value no gradient reaches; `gradients.clip(&graph, threshold)` rescales the whole set; `gradients.recompute(|graph| ..)` re-runs a region in the backward pass instead of keeping its output (activation checkpointing).
- In-place writers: `add_into`, `mul_into`, `copy_into`, `write_into(target, indices, updates)`, `scatter_into`, plus `graph.quantize`. They are what an optimizer is made of, and they must be authored after `backward` — a graph is differentiated before any of its leaves is updated in place.
- There is one ordering for landing a gradient: the contribution is first reduced to the shape of the tensor that receives it, then mapped into the layout of the tensor that owns the storage. A view receives its gradient through the tensor it aliases, so `gradients.of(view)` and `gradients.of(owner)` are the same value.

## Weights, programs and runs

- `runtime.weights(&graph)` opens a `Weights<'r>` store sized and laid out for that graph's parameters and state. One store serves every program of one model; `runtime.rebind(&weights, &graph)` points it at a rebuilt graph with the same layout.
- `runtime.compile(&graph, &weights)` freezes the current revision into a `Program<'r>`, compiling device kernels the first time a plan appears. `runtime.precompile(&graph, profile)` compiles without building a program; `runtime.compile_with(.., profile)` and `runtime.compile_chosen(.., profile, &[(Product, MatmulTile)])` take a plan choice; `runtime.tune(&graph, &weights)` measures candidates and returns the winner.
- `runtime.profiles()`, `runtime.default_profile()`, `runtime.budget()`, `runtime.capability()`, `runtime.measure(&program)` expose what the device offers and what a plan costs.
- A program reports `task_count`, `step_count`, `wave_count`, `workgroups`, `value_count`, `work`, `tensor_bytes`, `arena_bytes`, `resident_bytes`, `heap_bytes`, `device_bytes`, `weights()`, `dynamic()`, `updates_weights()`, `carries_authored()`, `tiles()`, `matmul_geometries()`, `readable(value)`, `span(value)`.
- `runtime.run(&program)` submits the whole plan and returns a `Run`, whose `seconds()` measures that submission. Every run re-executes forward, backward and any in-place update in the plan.
- Host traffic: `runtime.write(&program, value, &[f32])`, `runtime.read(&program, value) -> Vec<f32>`, `runtime.read_many(..)`, and the asynchronous pair `runtime.pull(&program, &[values]) -> Readout` / `runtime.collect(readout)`. A `Readout` holds the run it was pulled from, so one step of readback can overlap the next run; `runtime.readback_slots()` is how many may be in flight (`RuntimeRequest::readback_bytes` sizes them).
- `Span`, from `program.span(value)`, describes where a tensor lives: payload bytes and offset, the table that follows for a quantized element, element count and scale.

## Checkpoints

- `runtime.checkpoint(&weights)` captures every *named* tensor — parameters and state — into a `Checkpoint`. `checkpoint.bytes()` and `Checkpoint::decode(&bytes)` are the container, whose layout is safetensors; `checkpoint.tensors()`, `names()`, `tensor(name)` inspect it.
- `runtime.load(&graph, &checkpoint)` opens a store for a graph whose named tensors match; `runtime.restore(&weights, &checkpoint)` writes a checkpoint into an existing store, before or between runs. Half-precision stores round-trip bit for bit (`neura-runtime/tests/persist.rs::a_half_store_round_trips_bit_for_bit`).
- Names come from layer constructors (`Mlp::new(&graph, "model", ..)` names `model.layers.0.weight` and so on) or from `named_parameter`/`named_state`. An unnamed parameter refuses a checkpoint; a graph whose names do not match refuses a load.

## Device layer, for kernels you write yourself

`neura-gpu`: `Device::open(&GpuRequest)`, `Queue::of(&device)`, `GpuBuffer::new/write/write_at/read`, `BufferBinding`, `GpuContext::declare(ComputeProgram) -> PipelineHandle`, `PipelineHandle::bind_group(&[Binding])`, `Submission::new/dispatch/submit` with `SubmissionIndex`, `Readback` for host-visible rounds, `ArtifactCache` and `GpuRequest::artifacts` for caching compiled artifacts, `Limits`, `Capability`, `CooperativeMatrix`.

`neura-shader`: build a `Module` (globals, functions, blocks, instructions), then a `ComputeProgram` for a `Backend` — HLSL for DX12, MSL for Metal, SPIR-V for Vulkan — through `hlsl`, `msl` and `spirv`. `reflect` and `describe` report the bindings a module needs, and `MAX_BINDING_BYTES` is the ceiling one binding may span.

## What a plan guarantees

`neura-plan` turns a graph into a task grid. `Layout` is the byte layout of tensors and weights, `Region` is a range of one store, `Seed` is the initial content of a parameter. `Plan::compile` asserts its own invariants in `encode.rs`, and `region.rs` narrows what each task reads and writes; `hazard.rs` orders tasks that touch the same storage. When a Neura program produces wrong numbers, the invariants there are what a maintainer will ask you about — a graph that cannot be planned fails at compile with a message naming the task.
