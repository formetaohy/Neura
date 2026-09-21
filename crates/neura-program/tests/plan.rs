use neura_abi::op;
use neura_abi::{
    Kind, NARROW, PROFILES, Placement, Precision, Profile, StepRecord, TaskRecord, WIDE, WORD_BYTES,
};
use neura_program::{Encoding, Graph, Init, Shape, Store};
use std::mem::size_of;

const ALIGNMENT: u64 = 256;
const PLACEMENT: Placement = Placement::new(1 << 16, 1 << 18);

fn encoding(graph: &Graph) -> Encoding {
    encoding_with(graph, NARROW)
}

fn encoding_with(graph: &Graph, profile: Profile) -> Encoding {
    graph.encode(ALIGNMENT, profile, Precision::Single)
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

fn kinds(encoding: &Encoding) -> Vec<Kind> {
    tape(encoding)
        .iter()
        .map(|task| Kind::of(task.kind))
        .collect()
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
    assert_eq!(graph.task_count(), 4);
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
    assert_eq!(kinds(&encoding), vec![Kind::Matmul]);
    let tape = tape(&encoding);
    assert_eq!(
        tape[0].steps, 2,
        "the bias and the rectifier ride the product"
    );
    let steps = steps(&encoding);
    assert_eq!(steps[tape[0].chain as usize].op, op::ADD);
    assert_eq!(steps[tape[0].chain as usize].operand, bias.id());
    assert_eq!(steps[tape[0].chain as usize + 1].op, op::RELU);
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
    assert_eq!(kinds(&encoding), vec![Kind::Binary]);
    assert_eq!(readers(&encoding, data.id()), vec![0]);
    assert_eq!(tape(&encoding)[0].out, squared.id());
}

#[test]
fn a_value_two_tasks_read_keeps_its_storage_from_the_output_of_either() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(64), Init::Zero);
    let right = graph.parameter(Shape::vector(64), Init::Zero);
    let product = graph.mul(left, right);
    let negated = graph.neg(product);
    let squared = graph.mul(product, product);
    let encoding = encoding(&graph);
    let reader = writers(&encoding, negated.id())[0];
    let taker = writers(&encoding, squared.id())[0];
    assert_eq!(
        wave_of(&encoding, reader),
        wave_of(&encoding, taker),
        "both readers of a value share a wave",
    );
    assert_ne!(
        encoding.span(product, PLACEMENT).offset,
        encoding.span(squared, PLACEMENT).offset,
        "the storage of a value a second task reads in the same wave was handed to its reader",
    );
}

#[test]
fn a_value_one_task_reads_hands_its_storage_to_that_task() {
    let graph = Graph::new();
    let data = graph.parameter(Shape::vector(64), Init::Zero);
    let scaled = graph.mul(data, graph.fill(Shape::vector(64), 2.0));
    let squared = graph.mul(scaled, scaled);
    let encoding = encoding(&graph);
    assert_eq!(
        encoding.span(scaled, PLACEMENT).offset,
        encoding.span(squared, PLACEMENT).offset,
        "the only reader of a value takes the storage it consumed",
    );
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
    assert_eq!(kinds(&encoding)[reader[0]], Kind::Unary);
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
        256 * 4,
        "the arena holds only the value the fused chain produces",
    );
    assert_eq!(
        graph.layout(ALIGNMENT, Precision::Single).weights().words(),
        256,
        "the parameter lives in the weight store, not in the arena",
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
    assert_eq!(small.arena_bytes(), 256 * 4);
    assert_eq!(large.arena_bytes(), 4096 * 4);
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
fn a_matmul_takes_the_widest_tile_that_still_fills_the_device() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(1024, 16), Init::Zero);
    let right = graph.parameter(Shape::matrix(16, 1024), Init::Zero);
    let blocked = graph.parameter(Shape::matrix(16, 48), Init::Zero);
    let balanced = graph.matmul(left, right);
    let ragged = graph.matmul(left, blocked);
    graph.retain(balanced);
    graph.retain(ragged);
    let encoding = encoding_with(&graph, WIDE);
    let ladder = encoding.profile().ladder();
    let tiles = encoding.tiles();
    let tasks = tape(&encoding);
    let balanced_task = *tasks
        .iter()
        .find(|task| task.out == balanced.id())
        .expect("the 1024x1024 product holds a task");
    assert_eq!(
        tiles[balanced_task.geometry as usize], ladder[2],
        "a 1024x1024 product is tiled as widely as its profile divides it",
    );
    let ragged_task = *tasks
        .iter()
        .find(|task| task.out == ragged.id())
        .expect("the 1024x48 product holds a task");
    assert_eq!(
        tiles[ragged_task.geometry as usize], ladder[0],
        "a product no wide tile divides takes the narrowest tile of its profile",
    );
    assert_eq!(
        tiles,
        &[ladder[0], ladder[2]],
        "a device program carries only the tiles its tape names",
    );
    assert_eq!(
        encoding.matmul_geometries(),
        vec![(ladder[0], 192), (ladder[2], 256)],
        "a tape reports the geometry of every task it hands the device",
    );
    assert_eq!(
        encoding.work(),
        256 * ladder[2].tile_work() + 192 * ladder[0].tile_work(),
        "a plan accounts the tile work it dispatches",
    );
    assert_ne!(
        encoding_with(&graph, NARROW).tiles(),
        encoding_with(&graph, WIDE).tiles(),
        "a profile decides which tiles a product is tiled with",
    );
    assert!(
        encoding_with(&graph, WIDE).task_count() < encoding_with(&graph, NARROW).task_count(),
        "a wider tile hands the device fewer, larger tasks",
    );
}

#[test]
fn a_wide_op_hands_the_device_a_bounded_number_of_tasks() {
    for (elements, tasks) in [(32u32, 1u32), (4096, 2), (65536, 32), (1 << 20, 256)] {
        let graph = Graph::new();
        let data = graph.input(Shape::vector(elements));
        graph.relu(data);
        let encoding = encoding(&graph);
        assert_eq!(
            encoding.task_count(),
            tasks,
            "a rectifier over {elements} elements hands the device {tasks} tasks",
        );
    }
}

#[test]
fn every_profile_plans_the_same_values() {
    for profile in PROFILES {
        let graph = Graph::new();
        let weight = graph.parameter(Shape::matrix(4, 8), Init::Zero);
        let bias = graph.parameter(Shape::vector(8), Init::Zero);
        let data = graph.input(Shape::matrix(2, 4));
        let out = graph.relu(graph.add(graph.matmul(data, weight), bias));
        let encoding = encoding_with(&graph, *profile);
        assert_eq!(
            encoding.value_count() as usize,
            graph.value_count(),
            "{profile:?} publishes values a graph without a reduction holds",
        );
        assert_eq!(encoding.span(out, PLACEMENT).elements, 16);
    }
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
        kinds.contains(&Kind::Matmul),
        "the weight gradient is a matmul"
    );
    assert!(
        kinds.contains(&Kind::Partial),
        "the rectifier contributes a mask"
    );
    assert!(
        kinds.contains(&Kind::SumChunk),
        "the loss reduces through partial sums"
    );
    assert!(
        kinds.contains(&Kind::Broadcast),
        "the loss gradient spreads the scalar over the tensor"
    );
    assert_eq!(
        kinds.iter().filter(|kind| **kind == Kind::SumAxis).count(),
        1,
        "the bias gradient folds the rows it was spread over",
    );
}

#[test]
fn a_reduction_folds_through_as_many_levels_as_it_takes() {
    let graph = Graph::new();
    let wide = graph.parameter(Shape::vector(1 << 21), Init::Zero);
    let loss = graph.sum(wide);
    let encoding = encoding(&graph);
    let reductions = kinds(&encoding)
        .iter()
        .filter(|kind| **kind == Kind::SumChunk)
        .count();
    assert_eq!(
        reductions, 257,
        "a sum folds a chunk of 8192 elements at a time until one scalar stands",
    );
    assert_eq!(
        encoding.wave_count(),
        2,
        "every level of the fold is a wave"
    );
    assert_eq!(loss.shape(), Shape::scalar());
    assert!(
        encoding.value_count() as usize > graph.value_count(),
        "a folded reduction publishes its partial sums",
    );
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
    assert_eq!(encoding.span(matrix, PLACEMENT).elements, 32);
    assert!(
        refuses(|| {
            let _ = encoding.span(transposed, PLACEMENT);
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
    assert!(encoding.value_count() as usize >= graph.value_count());
    assert!(encoding.arena_bytes() > 0);
    assert!(encoding.work() > 0);
    let layout = graph.layout(ALIGNMENT, Precision::Single);
    let seed = layout
        .uploads()
        .iter()
        .find(|(address, _)| {
            layout.weight_bytes(PLACEMENT, *address) == encoding.span(weight, PLACEMENT).offset
        })
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

#[test]
fn every_tensor_a_plan_names_lies_inside_its_region() {
    for profile in PROFILES {
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
        let encoding = encoding_with(&graph, *profile);
        let layout = graph.layout(ALIGNMENT, Precision::Single);
        for value in [weight, data, out, grads.of(weight)] {
            let span = encoding.span(value, PLACEMENT);
            let bytes = u64::from(span.elements) * WORD_BYTES;
            let (base, limit) = match span.store {
                Store::Tensors => (PLACEMENT.tensors() * WORD_BYTES, encoding.tensor_bytes()),
                Store::Weights => (PLACEMENT.weights() * WORD_BYTES, layout.weights().bytes()),
            };
            assert!(
                span.offset - base + bytes <= limit,
                "tensor {} of {} bytes at {} leaves the {} bytes of its {:?}",
                value.id(),
                bytes,
                span.offset - base,
                limit,
                span.store,
            );
            assert_eq!(
                (span.offset - base) % ALIGNMENT,
                0,
                "a tensor lies off the block grid"
            );
        }
    }
}

#[test]
fn a_folded_operand_remembers_which_side_of_its_consumer_it_took() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(4), Init::Zero);
    let right = graph.parameter(Shape::vector(4), Init::Zero);
    let bias = graph.input(Shape::vector(4));
    let difference = graph.sub(bias, graph.mul(left, right));
    let quotient = graph.div(graph.mul(right, left), bias);
    graph.retain(difference);
    graph.retain(quotient);
    let encoding = encoding(&graph);
    assert_eq!(encoding.task_count(), 2);
    let tape = tape(&encoding);
    let steps = steps(&encoding);
    assert_eq!(tape[0].op, op::MUL);
    let folded_into_a_subtraction = steps[tape[0].chain as usize];
    assert_eq!(folded_into_a_subtraction.op, op::SUB);
    assert_eq!(folded_into_a_subtraction.operand, bias.id());
    assert_eq!(
        folded_into_a_subtraction.swapped, 1,
        "a product folded into a subtraction is its subtrahend",
    );
    assert_eq!(tape[1].op, op::MUL);
    let folded_into_a_quotient = steps[tape[1].chain as usize];
    assert_eq!(folded_into_a_quotient.op, op::DIV);
    assert_eq!(folded_into_a_quotient.operand, bias.id());
    assert_eq!(
        folded_into_a_quotient.swapped, 0,
        "a product folded into a quotient is its dividend",
    );
}

#[test]
fn a_partial_reads_back_the_result_its_formula_names() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero);
    let scale = graph.parameter(Shape::vector(4), Init::Zero);
    let activated = graph.tanh(weight);
    let scaled = graph.mul(activated, scale);
    let loss = graph.sum(scaled);
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(weight));
    let encoding = encoding(&graph);
    let tape = tape(&encoding);
    let tangent = tape
        .iter()
        .find(|task| task.op == op::TANH)
        .expect("the tangent keeps a task of its own while the product differentiates it");
    assert_eq!(tangent.out, activated.id());
    let partial = tape
        .iter()
        .find(|task| task.op == op::TANH && Kind::of(task.kind) == Kind::Partial)
        .expect("the tangent leaves a partial behind");
    assert_eq!(
        partial.a,
        activated.id(),
        "a tangent partial reads the value its own op produced, not the operand it was handed",
    );
    assert_eq!(partial.b, neura_abi::NO_VALUE);
    assert_eq!(partial.slot, 0);
    assert_eq!(partial.c, gradients.of(activated).id());
    assert_eq!(partial.out, gradients.of(weight).id());
}

#[test]
fn a_partial_reads_back_the_operands_its_formula_names() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero);
    let other = graph.parameter(Shape::vector(4), Init::Zero);
    let magnitude = graph.abs(weight);
    let scaled = graph.mul(magnitude, other);
    let loss = graph.sum(scaled);
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(weight));
    graph.retain(gradients.of(other));
    let encoding = encoding(&graph);
    let tape = tape(&encoding);
    let absolute = tape
        .iter()
        .find(|task| task.op == op::ABS && Kind::of(task.kind) == Kind::Partial)
        .expect("a magnitude sign reaches the operand it was taken from");
    assert_eq!(
        absolute.a,
        weight.id(),
        "a magnitude partial has to read the operand it differentiates",
    );
    assert_eq!(absolute.b, neura_abi::NO_VALUE);
    let products = tape
        .iter()
        .filter(|task| task.op == op::MUL && Kind::of(task.kind) == Kind::Partial)
        .collect::<Vec<_>>();
    assert_eq!(products.len(), 2, "each tracked factor carries a partial");
    for (slot, partial) in products.iter().enumerate() {
        assert_eq!(
            partial.a,
            neura_abi::NO_VALUE,
            "a product descends through the factor it did not differentiate",
        );
        assert_eq!(
            partial.b,
            if slot == 0 {
                other.id()
            } else {
                magnitude.id()
            },
        );
        assert_eq!(partial.slot, slot as u32);
    }
}
