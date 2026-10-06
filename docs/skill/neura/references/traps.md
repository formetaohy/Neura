# Neura traps

Neura fails fast: a wrong program panics with a sentence that names the value and the axis. What follows are the panics you will meet, what they mean, the pattern that avoids them, and the file on this machine that pins the behaviour. Read the file before you argue with the message — the tests are the contract.

## 1. `Runtime::read` or `write` refuses a tensor

> a temporary whose storage a later task of the plan reuses, or a view that walks a layout the storage does not

A derived tensor is alive only as long as a later task reads it: the planner may hand the storage of such a temporary to a later task, in which case the host can no longer read it. A permuted view cannot be addressed row by row through its storage, so the host can neither read nor write it.

Fix: `graph.retain(value)` anything the host reads back. Read and write the tensor that owns the storage, or fold the view into a derived tensor first (an elementwise op, `sum_rows`, a matmul) and retain that.

Evidence: `neura-runtime/tests/execution.rs::a_reclaimed_temporary_is_refused_and_a_retained_one_reads_back`; `neura-runtime/tests/view.rs::a_permuted_view_reads_back_through_the_axes_it_names` shows a reshape reading back and a permutation refusing.

## 2. `backward` runs once, and in-place updates come after

> a graph is differentiated once; build another graph for a second pass
> a graph is differentiated before any of its leaves is updated in place

An optimizer step is part of the graph. Calling `graph.backward` after `add_into`/`write_into`, or twice on one graph, is refused.

Fix: author forward, then `backward`, then the optimizer step, then compile. A second pass over the same weights means a second program over a rebuilt graph, with `runtime.rebind` keeping the store.

Evidence: `neura-graph/src/autodiff.rs`; `neura-runtime/tests/rebind.rs`; `neura-nn/tests/optimizer.rs`.

## 3. A gradient that reaches nothing

> no gradient reaches {shape}, and the gradient of a view reaches the tensor that owns its storage
> the loss derives from no parameter

`Gradients::of` panics for a value no gradient flows to: a frozen parameter, a constant, an input that is not a `gradient_input`, or a loss built only from those.

Fix: use `graph.gradient_input` for inputs whose gradients you want, `graph.trains(value)` to ask before assuming, and `graph.carries_gradient(id)` when iterating.

Evidence: `neura-graph/tests/gradient.rs`, `neura-graph/tests/freeze.rs`, `neura-graph/tests/prefix.rs`.

## 4. `freeze` must come before the tasks that read the parameter

> a parameter is frozen before the tasks that read it are authored

Freezing changes whether the parameter takes part in training, so it is decided before the graph that reads it exists.

Fix: freeze the parameters you want constant right after building the layers, before any op reads them.

Evidence: `neura-graph/src/graph.rs::freeze`; `neura-graph/tests/freeze.rs`.

## 5. A compiled program belongs to one graph revision

> the program is not current

`Runtime::run`, `bind`, `write` and `read` all assert `Program::is_current()`. Any graph call after `runtime.compile` — a new op, a `retain`, another `backward` — moves the revision and invalidates the program.

Fix: compile last. Keep the graph object alive for the whole training loop, and express a changing batch as a free extent rather than as a rebuilt graph.

Evidence: `neura-runtime/tests/revision.rs`, `neura-runtime/tests/precompile.rs`.

## 6. Free extents: the bound is a compile-time ceiling

> writing N numbers into a tensor of M numbers

A free extent declares an upper bound when the graph is built. `Runtime::bind` lowers the live length and nothing may exceed the bound; write and read sizes follow the *binding*, not the bound.

Fix: bind before writing, and size host data from the live length. Bindings are positional, in the order the graph named the extents; a count the device authors (a ragged axis's token length, `Graph::counted` with a device value) is not in that list at all.

A write is sized by the declared bound and a read by the live extents: a packed tensor of bound 16 over live lengths 4, 3, 2 and 1 takes 16 x width numbers to write and reads back as 10 x width.

Evidence: `neura-runtime/tests/extent.rs`, `neura-runtime/tests/varlen.rs`, `neura-plan/src/encode.rs::host_slots`.

## 7. `Shape::freed` is exact

> axis {axis} of {dims} declares {n} numbers where the free extent bound at {bound} holds them

The declared dimension must equal the bound of the free extent, the shape has four axes at most, and every dimension is positive. `Shape::matrix(rows, columns)` places `rows` on axis 2, so a free row axis is `freed(&[(2, free)])`, not axis 0.

Evidence: `neura-graph/src/shape.rs`, `neura-graph/tests/shape.rs`, `neura-graph/tests/extent.rs`.

## 8. A static operand beside a free extent holds one number or the bound

An elementwise operand may be a literal (`dim 1`) or as long as the bound the walk can reach; anything in between is refused while the graph is built.

Fix: broadcast a table to the bound, or slice it to the live length. A normalization over a dynamic width works because its scale holds one number.

Evidence: `neura-graph/tests/extent.rs`; `neura-graph/src/shape.rs::covers_a_walk`.

## 9. Ragged axes are packed, and their sums are per plane

- A tensor that walks a ragged extent is `[1, 1, bound, width]` with that extent on axis 2. Anything else refuses while the graph is built.
- `Graph::sum_axis` and `sum_rows` refuse a registered ragged axis: summing a packed axis mixes rows of different planes. Use `Graph::segment_sum(value, ragged)`, which gives one number per plane, or `Graph::rows(ragged)` to scatter per-plane numbers back onto rows.
- `graph.ragged` caps a bound at `EXACT_WALK_LIMIT` (2^24); the lengths are what the host writes per batch.

Evidence: `neura-graph/tests/ragged.rs`, `neura-graph/tests/segment.rs`, `neura-runtime/tests/ragged_sum.rs`, `neura-runtime/tests/rows.rs`.

## 10. Attention asks for its scale, its mask and its cursor explicitly

- `scale` must be finite and non-zero: a zero scale weighs every score to nothing and is refused.
- `reach` (a sliding window) requires `causal: true`; a window over keys the mask leaves behind is refused.
- `origin` is the cursor. It is required for a ring cache and refused for a packed query axis; with a window the token position comes from the cursor, not from the row index.
- `query_segments` expresses chunked prefill (a query block shorter or longer than its key plane) and needs `segments` as well.

Evidence: `neura-graph/src/op.rs::attention`; `neura-runtime/tests/attention.rs`, `neura-runtime/tests/decode.rs`, `neura-runtime/tests/chunked.rs`, `neura-runtime/tests/varlen.rs`.

## 11. A KV cache is device storage, not a host write

`graph.resident` takes storage beside the weights; tasks write it (`write_into`), and it is not a parameter, not trainable and not checkpointed. Step the cache by writing the position as the cursor and `position % capacity` as the slot.

Evidence: `neura-runtime/tests/decode.rs`.

## 12. Checkpoints are named

> a checkpoint names every tensor it holds, and an unnamed parameter carries no name

Layer constructors and `named_parameter`/`named_state` name tensors. `runtime.checkpoint` refuses an unnamed parameter; `runtime.load` refuses a graph whose named tensors differ; `runtime.restore` writes a checkpoint into a store you already hold. Half-precision tensors round-trip bit for bit.

Evidence: `neura-runtime/tests/persist.rs`, `neura-runtime/tests/safetensors.rs`, `neura-nn/tests/state.rs`.

## 13. Host traffic is always f32, packed per element and scale

`Runtime::write` takes `&[f32]` and `Runtime::read` returns `Vec<f32>` whatever the storage element is; packing and the scale table are the runtime's business. `program.span(value)` reports what a tensor occupies, payload and table.

Evidence: `neura-runtime/src/runtime/mod.rs::write`; `neura-runtime/tests/precision.rs`, `neura-runtime/tests/extent.rs`.

## 14. The readback ring is finite

> every readback of the runtime was in flight

Two readback slots exist by default. A `Readout` you never `collect` holds its slot, and the next `pull` panics once they run out.

Fix: collect, or use `runtime.read`, or size the ring with `RuntimeRequest::readback_bytes` when you want more steps in flight.

Evidence: `neura-runtime/tests/execution.rs::a_pull_without_a_collect_runs_out_of_readbacks`.

## 15. The heap is a decision, not a surprise

`Runtime::open` takes `heap_bytes` (16 MB by default) and asserts it fits one storage binding of the device. Every tensor of a plan lives in that heap, and the weight store must fit one binding too.

Fix: read `program.heap_bytes()`, `program.tensor_bytes()` and `program.weights().bytes()` and raise `heap_bytes`; on a small device, split the model or lower the batch bound instead of retrying.

Evidence: `neura-runtime/tests/memory.rs`; `neura-runtime/src/runtime/mod.rs::of_context`.

## 16. One program, many bindings: verify more than the bound

A graph with a free extent is one program serving every length — that is the point, and the reason a single test at the bound proves nothing about the shapes a batch will actually take. Compare against a fixed-shape reference at the bound, a middle length and one row, for the forward pass and for every gradient you train with.

Evidence: `neura-runtime/tests/extent.rs` (the pattern, for matmul, convert, softmax, normalization, products), `neura-nn/tests/training.rs`.

## 17. Float comparisons need a tolerance

Framework tests use `assert_close` with `1e-6` for f32 against f32, up to `1e-2` for half precision, and looser slack for gradients. Exact equality is only used where the value is exact by construction (a mask, an index, a permutation).

Evidence: `neura-runtime/tests/support/mod.rs`, `neura-graph/tests/predicate.rs`.
