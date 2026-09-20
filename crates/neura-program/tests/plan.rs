use neura_abi::{
    KIND_BINARY, KIND_EXPAND, KIND_MATMUL, KIND_SUM_CHUNK, KIND_SUM_TO, KIND_UNARY,
    KIND_UNARY_GRAD, StepRecord, TaskRecord,
};
use neura_program::{Encoding, Graph, Init, Shape};
use std::mem::size_of;

const ALIGNMENT: u64 = 256;

fn encoding(graph: &Graph) -> Encoding {
    graph.encode(ALIGNMENT)
}

fn records<T: bytemuck::AnyBitPattern>(bytes: &[u8], width: usize) -> Vec<T> {
    bytes
        .chunks_exact(width)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

fn tape(encoding: &Encoding) -> Vec<TaskRecord> {
    records(encoding.tasks(), size_of::<TaskRecord>())
}

fn steps(encoding: &Encoding) -> Vec<StepRecord> {
    records(encoding.steps(), size_of::<StepRecord>())
}

fn kinds(encoding: &Encoding) -> Vec<u32> {
    tape(encoding).iter().map(|task| task.kind).collect()
}

fn wave_of(encoding: &Encoding, task: usize) -> u32 {
    encoding
        .waves()
        .iter()
        .position(|end| task < *end as usize)
        .expect("every task belongs to a wave") as u32
}

fn writers(encoding: &Encoding, value: u32) -> Vec<usize> {
    tape(encoding)
        .iter()
        .enumerate()
        .filter(|(_, task)| task.out == value)
        .map(|(index, _)| index)
        .collect()
}

fn readers(encoding: &Encoding, value: u32) -> Vec<usize> {
    let tape = tape(encoding);
    let steps = steps(encoding);
    tape.iter()
        .enumerate()
        .filter(|(_, task)| {
            task.a == value
                || task.b == value
                || task.c == value
                || (task.chain..task.chain + task.steps)
                    .any(|step| steps[step as usize].operand == value)
        })
        .map(|(index, _)| index)
        .collect()
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_chain_of_rectifiers_fuses_into_one_task() {
    let graph = Graph::new();
    let mut value = graph.parameter(Shape::vector(4), Init::Zero);
    for _ in 0..4 {
        value = graph.relu(value);
    }
    let encoding = encoding(&graph);
    assert_eq!(encoding.task_count(), 1);
    assert_eq!(encoding.wave_count(), 1);
    assert_eq!(encoding.step_count(), 3);
    assert_eq!(value.shape().elements(), 4);
}

#[test]
fn a_dense_layer_fuses_its_epilogue_into_the_product() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 8), Init::Zero);
    let bias = graph.parameter(Shape::vector(8), Init::Zero);
    let data = graph.input(Shape::matrix(3, 4));
    let activated = graph.relu(graph.add(graph.matmul(data, weight), bias));
    let encoding = encoding(&graph);
    assert_eq!(encoding.task_count(), 1);
    assert_eq!(encoding.wave_count(), 1);
    assert_eq!(kinds(&encoding), vec![KIND_MATMUL]);
    let tape = tape(&encoding);
    assert_eq!(
        tape[0].steps, 2,
        "the bias and the rectifier ride the product"
    );
    let steps = steps(&encoding);
    assert_eq!(steps[tape[0].chain as usize].operand, bias.id());
    assert_eq!(steps[tape[0].chain as usize + 1].operand, u32::MAX);
    assert_eq!(tape[0].out, activated.id());
}

#[test]
fn a_value_two_tasks_read_stays_on_the_tape() {
    let graph = Graph::new();
    let data = graph.parameter(Shape::vector(8), Init::Zero);
    let squared = graph.mul(data, data);
    let encoding = encoding(&graph);
    assert_eq!(encoding.task_count(), 1);
    assert_eq!(kinds(&encoding), vec![KIND_BINARY]);
    assert_eq!(readers(&encoding, data.id()), vec![0]);
    assert_eq!(tape(&encoding)[0].out, squared.id());
}

#[test]
fn a_retained_value_is_never_folded_away() {
    let graph = Graph::new();
    let data = graph.parameter(Shape::vector(8), Init::Zero);
    let doubled = graph.mul(data, graph.fill(Shape::vector(8), 2.0));
    graph.retain(doubled);
    let activated = graph.relu(doubled);
    graph.retain(activated);
    let encoding = encoding(&graph);
    let reader = writers(&encoding, activated.id());
    assert_eq!(reader.len(), 1, "the rectifier keeps a task of its own");
    assert_eq!(kinds(&encoding)[reader[0]], KIND_UNARY);
    assert!(
        readers(&encoding, doubled.id()).contains(&reader[0]),
        "the rectifier reads the value the graph retains",
    );
}

#[test]
fn independent_tasks_share_a_wave() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(64), Init::Zero);
    let right = graph.parameter(Shape::vector(64), Init::Zero);
    let sum = graph.add(left, right);
    let product = graph.mul(left, right);
    graph.retain(sum);
    graph.retain(product);
    let out = graph.add(sum, product);
    graph.retain(out);
    let encoding = encoding(&graph);
    assert_eq!(encoding.task_count(), 3);
    assert_eq!(encoding.wave_count(), 2);
    assert_eq!(encoding.waves(), &[2, 3]);
    assert_eq!(out.shape(), Shape::vector(64));
}

#[test]
fn a_chain_of_temporaries_holds_one_tensor() {
    let graph = Graph::new();
    let mut value = graph.parameter(Shape::vector(256), Init::Zero);
    for _ in 0..16 {
        value = graph.relu(value);
    }
    let encoding = encoding(&graph);
    assert_eq!(encoding.task_count(), 1);
    assert_eq!(
        encoding.arena_bytes(),
        256 * 4 + 256 * 4,
        "the parameter and the value the fused chain produces",
    );
}

#[test]
fn a_plan_sizes_its_own_arena() {
    let small = Graph::new();
    let parameter = small.parameter(Shape::vector(256), Init::Zero);
    small.relu(parameter);
    let large = Graph::new();
    let parameter = large.parameter(Shape::vector(4096), Init::Zero);
    large.relu(parameter);
    let small = encoding(&small);
    let large = encoding(&large);
    assert_eq!(small.arena_bytes(), 256 * 4 + 256 * 4);
    assert_eq!(large.arena_bytes(), 4096 * 4 + 4096 * 4);
    assert!(
        small.arena_bytes() < large.arena_bytes(),
        "a wider tensor asks for a wider arena",
    );
}

#[test]
fn a_matmul_needs_a_shared_depth() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(2, 3), Init::Zero);
    let right = graph.parameter(Shape::matrix(2, 4), Init::Zero);
    assert!(
        refuses(|| {
            let _ = graph.matmul(left, right);
        }),
        "a matmul of a 2x3 by a 2x4 was accepted",
    );
}

#[test]
fn elementwise_operands_must_meet() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(2, 3), Init::Zero);
    let right = graph.parameter(Shape::matrix(2, 4), Init::Zero);
    assert!(
        refuses(|| {
            let _ = graph.add(left, right);
        }),
        "an add of a 2x3 and a 2x4 was accepted",
    );
}

#[test]
fn a_matmul_tiles_its_output() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(64, 32), Init::Zero);
    let right = graph.parameter(Shape::matrix(32, 48), Init::Zero);
    let out = graph.matmul(left, right);
    assert_eq!(out.shape(), Shape::matrix(64, 48));
    let encoding = encoding(&graph);
    let tiles: u32 = 4 * 3;
    assert_eq!(
        encoding.task_count(),
        tiles.div_ceil(neura_program::MATMUL_TILES_PER_TASK),
    );
    assert_eq!(kinds(&encoding).len(), encoding.task_count() as usize);
    assert!(kinds(&encoding).iter().all(|kind| *kind == KIND_MATMUL));
}

#[test]
fn a_backward_pass_reaches_every_parameter() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 8), Init::Zero);
    let bias = graph.parameter(Shape::vector(8), Init::Zero);
    let input = graph.input(Shape::matrix(16, 4));
    let dense = graph.matmul(input, weight);
    let shifted = graph.add(dense, bias);
    let activated = graph.relu(shifted);
    let loss = graph.sum(activated);
    let grads = graph.backward(loss);
    assert_eq!(grads.of(weight).shape(), Shape::matrix(4, 8));
    assert_eq!(grads.of(bias).shape(), Shape::vector(8));

    let kinds = kinds(&encoding(&graph));
    assert!(
        kinds.contains(&KIND_MATMUL),
        "the weight gradient is a matmul"
    );
    assert!(
        kinds.contains(&KIND_UNARY_GRAD),
        "the rectifier contributes a mask"
    );
    assert!(
        kinds.contains(&KIND_SUM_CHUNK),
        "the loss reduces through partial sums"
    );
    assert!(
        kinds.contains(&KIND_EXPAND),
        "the loss gradient spreads over the tensor"
    );
    assert_eq!(kinds.iter().filter(|kind| **kind == KIND_SUM_TO).count(), 1);
}

#[test]
fn a_parameter_updated_in_place_feeds_the_tasks_that_follow() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(8), Init::Zero);
    let input = graph.input(Shape::vector(8));
    let loss = graph.sum(graph.mul(input, weight));
    let grads = graph.backward(loss);
    let scaled = graph.mul(grads.of(weight), graph.fill(Shape::scalar(), -0.1));
    graph.add_into(weight, scaled);
    let read_back = graph.relu(weight);
    graph.retain(read_back);
    let encoding = encoding(&graph);
    let update = writers(&encoding, weight.id());
    assert!(!update.is_empty(), "the update writes the parameter");
    let reader = writers(&encoding, read_back.id());
    assert_eq!(reader.len(), 1, "the reader holds a task of its own");
    assert!(
        update
            .iter()
            .all(|task| wave_of(&encoding, *task) < wave_of(&encoding, reader[0])),
        "a reader observes the update it follows",
    );
}

#[test]
fn an_update_in_place_never_shares_a_wave_with_its_readers() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(8), Init::Zero);
    let input = graph.input(Shape::vector(8));
    let read = graph.mul(input, weight);
    let loss = graph.sum(read);
    let grads = graph.backward(loss);
    graph.add_into(weight, grads.of(weight));
    let read_back = graph.relu(weight);
    graph.retain(read_back);
    let encoding = encoding(&graph);
    let update = writers(&encoding, weight.id());
    assert_eq!(update.len(), 1, "the update writes the parameter once");
    let before = wave_of(&encoding, writers(&encoding, read.id())[0]);
    let after = wave_of(&encoding, writers(&encoding, read_back.id())[0]);
    assert!(
        before < wave_of(&encoding, update[0]),
        "an update in place follows every reader of the value it replaces",
    );
    assert!(
        after > wave_of(&encoding, update[0]),
        "a reader that follows the update observes it",
    );
}

#[test]
fn a_graph_updated_in_place_is_refused_a_later_backward() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero);
    let loss = graph.sum(graph.relu(weight));
    graph.add_into(weight, graph.fill(Shape::vector(4), 0.5));
    assert!(
        refuses(|| {
            let _ = graph.backward(loss);
        }),
        "a loss was differentiated after its parameter had been updated",
    );
}

#[test]
fn a_view_shares_the_storage_of_its_source() {
    let graph = Graph::new();
    let matrix = graph.parameter(Shape::matrix(8, 4), Init::Zero);
    let transposed = graph.transpose(matrix);
    assert_eq!(transposed.shape(), Shape::matrix(4, 8));
    let encoding = encoding(&graph);
    assert_eq!(encoding.span(matrix).bytes, 32 * 4);
    assert!(
        refuses(|| {
            let _ = encoding.span(transposed);
        }),
        "a transposed view was handed its own storage",
    );
}

#[test]
fn a_softmax_row_keeps_its_shape_and_gradient() {
    let graph = Graph::new();
    let logits = graph.parameter(
        Shape::matrix(16, 8),
        Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
    );
    let probabilities = graph.softmax(logits);
    assert_eq!(probabilities.shape(), Shape::matrix(16, 8));
    let grads = graph.backward(graph.sum(probabilities));
    assert_eq!(grads.of(logits).shape(), Shape::matrix(16, 8));
}

#[test]
fn a_graph_is_differentiated_once() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero);
    let loss = graph.sum(graph.relu(weight));
    graph.backward(loss);
    assert!(
        refuses(|| {
            let _ = graph.backward(loss);
        }),
        "a graph was differentiated twice",
    );
}

#[test]
fn a_loss_that_derives_from_no_parameter_is_refused() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4));
    let loss = graph.sum(graph.relu(data));
    assert!(
        refuses(|| {
            let _ = graph.backward(loss);
        }),
        "a loss without parameters was differentiated",
    );
}

#[test]
fn a_plan_holds_every_value_and_the_seed_of_every_parameter() {
    let graph = Graph::new();
    let weight = graph.parameter(
        Shape::matrix(4, 4),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    );
    let input = graph.input(Shape::matrix(2, 4));
    let out = graph.softmax(graph.matmul(input, weight));
    graph.backward(graph.sum(out));
    let encoding = encoding(&graph);
    assert_eq!(encoding.value_count() as usize, graph.value_count());
    assert!(encoding.arena_bytes() > 0);
    assert!(encoding.work() > 0);
    let seed = encoding
        .initial()
        .iter()
        .find(|(offset, _)| *offset == encoding.span(weight).offset)
        .expect("the weight carries its initial samples");
    assert_eq!(seed.1.len(), 16);
}

#[test]
fn a_sum_wider_than_two_levels_is_refused() {
    let graph = Graph::new();
    let wide = graph.parameter(
        Shape::vector(neura_program::REDUCE_TILE * neura_program::REDUCE_TILE + 1),
        Init::Zero,
    );
    assert!(
        refuses(|| {
            let _ = graph.sum(wide);
        }),
        "a sum whose partial sums need a third level was accepted",
    );
}

#[test]
fn a_non_scalar_loss_is_refused() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero);
    assert!(
        refuses(|| {
            let _ = graph.backward(weight);
        }),
        "a vector was accepted as a loss",
    );
}

#[test]
fn every_tensor_a_plan_names_lies_inside_its_arena() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(8, 4), Init::Zero);
    let data = graph.input(Shape::matrix(2, 8));
    let out = graph.softmax(graph.add(
        graph.matmul(data, weight),
        graph.fill(Shape::vector(4), 1.0),
    ));
    graph.retain(out);
    let grads = graph.backward(graph.sum(out));
    graph.retain(grads.of(weight));
    let encoding = encoding(&graph);
    for value in [weight, data, out, grads.of(weight)] {
        let span = encoding.span(value);
        assert!(
            span.offset + span.bytes <= encoding.arena_bytes(),
            "tensor {} of {} bytes at {} leaves the {} byte arena",
            value.id(),
            span.bytes,
            span.offset,
            encoding.arena_bytes(),
        );
        assert_eq!(
            span.offset % ALIGNMENT,
            0,
            "a tensor lies off the block grid"
        );
    }
}
