use neura_abi::{
    KIND_BINARY, KIND_EXPAND, KIND_MATMUL, KIND_SUM_CHUNK, KIND_SUM_TO, KIND_UNARY,
    KIND_UNARY_GRAD, TaskRecord,
};
use neura_program::{Graph, Init, Shape};

const ARENA: u64 = 1 << 20;
const ALIGNMENT: u64 = 256;

fn tape(graph: &Graph) -> Vec<TaskRecord> {
    let encoding = graph.encode(ALIGNMENT, ARENA);
    bytemuck::cast_slice::<u8, TaskRecord>(encoding.tasks()).to_vec()
}

fn kinds(graph: &Graph) -> Vec<u32> {
    tape(graph).iter().map(|task| task.kind).collect()
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_chain_draws_one_wave_per_dependency() {
    let graph = Graph::new();
    let mut value = graph.parameter(Shape::vector(4), Init::Zero);
    for _ in 0..4 {
        value = graph.relu(value);
    }
    let encoding = graph.encode(ALIGNMENT, ARENA);
    assert_eq!(encoding.task_count(), 4);
    assert_eq!(encoding.wave_count(), 4);
    assert_eq!(value.shape().elements(), 4);
}

#[test]
fn independent_tasks_share_a_wave() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(64), Init::Zero);
    let right = graph.parameter(Shape::vector(64), Init::Zero);
    let sum = graph.add(left, right);
    let product = graph.mul(left, right);
    let out = graph.add(sum, product);
    let encoding = graph.encode(ALIGNMENT, ARENA);
    assert_eq!(encoding.task_count(), 3);
    assert_eq!(encoding.wave_count(), 2);
    assert_eq!(encoding.waves(), &[2, 3]);
    assert_eq!(out.shape(), Shape::vector(64));
}

#[test]
fn a_chain_of_temporaries_reuses_one_storage() {
    let graph = Graph::new();
    let mut value = graph.parameter(Shape::vector(256), Init::Zero);
    for _ in 0..16 {
        value = graph.relu(value);
    }
    let encoding = graph.encode(ALIGNMENT, ARENA);
    assert_eq!(
        encoding.arena_bytes(),
        256 * 4 + 256 * 4,
        "a chain of sixteen temporaries holds one intermediate at a time",
    );
}

#[test]
fn a_capacity_short_of_the_graph_is_refused() {
    let graph = Graph::new();
    let first = graph.parameter(Shape::vector(4096), Init::Zero);
    let second = graph.parameter(Shape::vector(4096), Init::Zero);
    let sum = graph.add(first, second);
    let product = graph.mul(first, second);
    graph.add(sum, product);
    let required = graph.encode(ALIGNMENT, ARENA).arena_bytes();
    assert!(
        refuses(|| {
            let _ = graph.encode(ALIGNMENT, required - 1);
        }),
        "an arena one byte short of the graph was accepted",
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
    let encoding = graph.encode(ALIGNMENT, ARENA);
    let tiles: u32 = 4 * 3;
    assert_eq!(
        encoding.task_count(),
        tiles.div_ceil(neura_program::MATMUL_TILES_PER_TASK),
    );
    assert_eq!(kinds(&graph).len(), encoding.task_count() as usize);
    assert!(kinds(&graph).iter().all(|kind| *kind == KIND_MATMUL));
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

    let kinds = kinds(&graph);
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
    let encoding = graph.encode(ALIGNMENT, ARENA);
    let records = tape(&graph);
    let update = records
        .iter()
        .position(|task| task.out == weight.id() && task.kind == KIND_BINARY)
        .expect("the update task is on the tape");
    let reader = records
        .iter()
        .position(|task| task.out == read_back.id() && task.kind == KIND_UNARY)
        .expect("the reader task is on the tape");
    assert!(update < reader, "a reader follows the update it observes");
    assert!(encoding.wave_count() >= 2);
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
fn an_update_in_place_never_shares_a_wave_with_its_readers() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(8), Init::Zero);
    let input = graph.input(Shape::vector(8));
    let loss = graph.sum(graph.mul(input, weight));
    let grads = graph.backward(loss);
    graph.add_into(weight, grads.of(weight));
    let encoding = graph.encode(ALIGNMENT, ARENA);
    let records = tape(&graph);
    let update = records
        .iter()
        .position(|task| task.out == weight.id() && task.kind == KIND_BINARY)
        .expect("the update task is on the tape");
    let wave = encoding
        .waves()
        .iter()
        .position(|end| update < *end as usize)
        .expect("the update task belongs to a wave");
    assert_eq!(
        wave as u32,
        encoding.wave_count() - 1,
        "an update in place must follow every reader",
    );
}

#[test]
fn a_view_shares_the_storage_of_its_source() {
    let graph = Graph::new();
    let matrix = graph.parameter(Shape::matrix(8, 4), Init::Zero);
    let transposed = graph.transpose(matrix);
    assert_eq!(transposed.shape(), Shape::matrix(4, 8));
    let encoding = graph.encode(ALIGNMENT, ARENA);
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
    let encoding = graph.encode(ALIGNMENT, ARENA);
    assert_eq!(encoding.value_count() as usize, graph.value_count());
    assert!(encoding.arena_bytes() <= ARENA);
    assert!(encoding.work() > 0);
    let seed = encoding
        .initial()
        .iter()
        .find(|(offset, _)| *offset == encoding.span(weight).offset)
        .expect("the weight carries its initial samples");
    assert_eq!(seed.1.len(), 16);
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
