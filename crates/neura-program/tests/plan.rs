use neura_abi::{Kind, Placement, StepRecord, Store, TaskRecord, WORD_BYTES};
use neura_graph::{Graph, Init, Shape};
use neura_op as op;
use neura_precision::Precision;
use neura_profile::{Budget, Profile};

const DEVICE: Budget = Budget::of(1024, 48 << 10);

fn narrow() -> Profile {
    Profile::derive(Budget::BASELINE)[0]
}

fn wide() -> Profile {
    *Profile::derive(Budget::BASELINE).last().expect("a profile")
}

fn every_profile() -> Vec<Profile> {
    Profile::derive(DEVICE)
}
use neura_program::{Encoding, Layout};
use std::mem::size_of;

const ALIGNMENT: u64 = 256;
const PLACEMENT: Placement = Placement::new(1 << 16, 1 << 18);

fn encoding(graph: &Graph) -> Encoding {
    encoding_with(graph, narrow())
}

fn encoding_with(graph: &Graph, profile: Profile) -> Encoding {
    Encoding::of(graph, ALIGNMENT, profile, Precision::Single, Vec::new())
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

fn segment_of(encoding: &Encoding, task: usize) -> usize {
    encoding
        .segments()
        .partition_point(|segment| segment.first as usize <= task)
        - 1
}

fn follows(encoding: &Encoding, before: usize, after: usize) -> bool {
    let segment = segment_of(encoding, before);
    let dispatch_before = dispatch_of(encoding, before);
    let dispatch_after = dispatch_of(encoding, after);
    dispatch_before < dispatch_after
        || (segment == segment_of(encoding, after)
            && before as u32 - encoding.segments()[segment].first
                < after as u32 - encoding.segments()[segment].first)
}

fn dispatch_of(encoding: &Encoding, task: usize) -> u32 {
    let mut at = 0usize;
    for (dispatch, record) in encoding.dispatches().iter().enumerate() {
        let count = encoding.segments()
            [record.first_segment as usize..(record.first_segment + record.segments) as usize]
            .iter()
            .map(|segment| segment.count as usize)
            .sum::<usize>();
        if task < at + count {
            return dispatch as u32;
        }
        at += count;
    }
    panic!("every task belongs to a dispatch")
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
    assert_eq!(encoding.dispatch_count(), 1);
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
    assert_eq!(encoding.dispatch_count(), 1);
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
        dispatch_of(&encoding, reader),
        dispatch_of(&encoding, taker),
        "both readers of a value share a dispatch",
    );
    assert_ne!(
        encoding.span(product, PLACEMENT).offset,
        encoding.span(squared, PLACEMENT).offset,
        "the storage of a value a second task reads in the same dispatch was handed to its reader",
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
fn independent_tasks_share_a_dispatch() {
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
    assert_eq!(encoding.dispatch_count(), 2);
    assert_eq!(encoding.dispatches()[0].segments, 2);
    assert_eq!(encoding.dispatches()[1].segments, 1);
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
        Layout::of(&graph, ALIGNMENT, Precision::Single)
            .weights()
            .words(),
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
fn a_product_takes_the_tile_that_stages_the_fewest_loads_for_its_shape() {
    let graph = Graph::new();
    let tall = graph.parameter(Shape::matrix(1024, 32), Init::Zero);
    let weight = graph.parameter(Shape::matrix(32, 64), Init::Zero);
    let blocked = graph.parameter(Shape::matrix(32, 48), Init::Zero);
    let balanced = graph.matmul(tall, weight);
    let ragged = graph.matmul(tall, blocked);
    graph.retain(balanced);
    graph.retain(ragged);
    let encoding = encoding_with(&graph, wide());
    let tiles = encoding.tiles();
    let tasks = tape(&encoding);
    let balanced_task = *tasks
        .iter()
        .find(|task| task.out == balanced.id())
        .expect("the 1024x64 product holds a task");
    let balanced_tile = tiles[balanced_task.geometry as usize];
    assert_eq!(
        (balanced_tile.rows(), balanced_tile.columns()),
        (64, 64),
        "a product whose shape divides takes the tile that multiplies the most elements per staged load",
    );
    assert_eq!(balanced_tile.registers(), 16);
    let ragged_task = *tasks
        .iter()
        .find(|task| task.out == ragged.id())
        .expect("the 1024x48 product holds a task");
    assert_eq!(
        tiles[ragged_task.geometry as usize], balanced_tile,
        "a product no tile divides is masked by the tile its shape pays the least for",
    );
    assert_eq!(
        encoding.tiles(),
        [balanced_tile],
        "a program carries the tiles its products name and no other",
    );
    assert_eq!(
        encoding.matmul_geometries(),
        vec![(balanced_tile, 32)],
        "a tape reports the geometry of every task it hands the device",
    );
    assert_eq!(
        encoding.work(),
        32 * balanced_tile.tile_work(),
        "a plan accounts the tile work it dispatches",
    );
    assert_ne!(
        encoding_with(&graph, narrow()).tiles(),
        encoding_with(&graph, wide()).tiles(),
        "a profile decides which tiles a product is tiled with",
    );
    assert!(
        encoding_with(&graph, wide()).task_count() < encoding_with(&graph, narrow()).task_count(),
        "a wider tile hands the device fewer, larger tasks",
    );
}

#[test]
fn a_product_whose_output_is_narrow_splits_its_depth_across_tasks() {
    let graph = Graph::new();
    let narrow = graph.parameter(Shape::matrix(8, 4096), Init::Zero);
    let weight = graph.parameter(Shape::matrix(4096, 32), Init::Zero);
    let out = graph.matmul(narrow, weight);
    graph.retain(out);
    let encoding = encoding_with(&graph, wide());
    let tape = tape(&encoding);
    assert_eq!(out.shape(), Shape::matrix(8, 32));
    let products = tape
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::Matmul)
        .collect::<Vec<_>>();
    let folds = tape
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::MatmulFold)
        .collect::<Vec<_>>();
    assert_eq!(folds.len(), 1);
    let tile = encoding.tiles()[products[0].geometry as usize];
    let tiles =
        out.shape().rows().div_ceil(tile.rows()) * out.shape().columns().div_ceil(tile.columns());
    let splits = folds[0].splits;
    assert!(
        splits > 1,
        "a product of {tiles} tiles fills no device without splitting its depth",
    );
    assert_eq!(products.len() as u32, tiles * splits);
    let partials = folds[0].a;
    assert_eq!(
        folds[0].out,
        out.id(),
        "the fold writes the product its graph names"
    );
    assert_ne!(partials, out.id(), "the fold reads partials of its own");
    let mut filled = vec![0u32; splits as usize];
    for task in &products {
        assert_eq!(task.splits, splits);
        assert_eq!(task.out, partials);
        assert_eq!(task.count, 1);
        filled[task.slot as usize] += 1;
    }
    assert!(
        filled.iter().all(|count| *count == tiles),
        "every split fills every tile of the product",
    );
    assert_eq!(
        folds[0].count,
        out.shape().elements(),
        "the fold writes one element per task",
    );
    assert_eq!(
        dispatch_of(&encoding, tape.len() - 1),
        encoding.dispatch_count() - 1,
        "the fold reads every slot after the last slot is written",
    );
}

#[test]
fn a_split_product_hands_its_epilogue_to_the_fold() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4096, 32), Init::Zero);
    let bias = graph.parameter(Shape::vector(32), Init::Zero);
    let data = graph.input(Shape::matrix(8, 4096));
    let out = graph.relu(graph.add(graph.matmul(data, weight), bias));
    graph.retain(out);
    let encoding = encoding_with(&graph, wide());
    let tape = tape(&encoding);
    let folds = tape
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::MatmulFold)
        .collect::<Vec<_>>();
    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0].out, out.id());
    assert_eq!(
        folds[0].steps, 2,
        "the bias and the rectifier ride the fold"
    );
    let steps = steps(&encoding);
    assert_eq!(steps[folds[0].chain as usize].op, op::ADD);
    assert_eq!(steps[folds[0].chain as usize].operand, bias.id());
    assert_eq!(steps[folds[0].chain as usize + 1].op, op::RELU);
    for task in tape
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::Matmul)
    {
        assert_eq!(
            task.steps, 0,
            "a product that splits its depth applies no epilogue of its own",
        );
    }
}

#[test]
fn a_wide_op_hands_the_device_a_bounded_number_of_tasks() {
    for (elements, tasks) in [(32u32, 1u32), (4096, 2), (65536, 32), (1 << 20, 512)] {
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
    for profile in every_profile() {
        let graph = Graph::new();
        let weight = graph.parameter(Shape::matrix(4, 8), Init::Zero);
        let bias = graph.parameter(Shape::vector(8), Init::Zero);
        let data = graph.input(Shape::matrix(2, 4));
        let out = graph.relu(graph.add(graph.matmul(data, weight), bias));
        let encoding = encoding_with(&graph, profile);
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
fn a_broadcast_product_folds_its_gradient_back_to_the_shape_of_its_operand() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([4, 1, 3, 2]), Init::Zero);
    let right = graph.parameter(Shape::of([1, 5, 2, 3]), Init::Zero);
    let product = graph.matmul(left, right);
    assert_eq!(product.shape(), Shape::of([4, 5, 3, 3]));
    let grads = graph.backward(graph.sum(product));
    assert_eq!(
        grads.of(left).shape(),
        left.shape(),
        "a gradient of a broadcast operand carries the shape of that operand",
    );
    assert_eq!(grads.of(right).shape(), right.shape());
    assert_eq!(
        kinds(&encoding(&graph))
            .iter()
            .filter(|kind| **kind == Kind::SumAxis)
            .count(),
        2,
        "one fold returns each operand to the rows its product spread",
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
        encoding.dispatch_count(),
        2,
        "every level of the fold is a dispatch"
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
            .all(|task| follows(&encoding, *task, reader[0])),
        "a reader observes the update it follows",
    );
}

#[test]
fn an_update_in_place_follows_every_reader_of_the_value_it_replaces() {
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
    let before = writers(&encoding, read.id())[0];
    let after = writers(&encoding, read_back.id())[0];
    assert!(
        follows(&encoding, before, update[0]),
        "an update in place follows every reader of the value it replaces",
    );
    assert!(
        follows(&encoding, update[0], after),
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
    let layout = Layout::of(&graph, ALIGNMENT, Precision::Single);
    let seed = layout
        .seeds()
        .iter()
        .find(|seed| {
            layout.weight_bytes(PLACEMENT, seed.address())
                == encoding.span(weight, PLACEMENT).offset
        })
        .expect("the weight carries its sampler");
    assert_eq!(seed.elements(), 16);
    assert_eq!(
        seed.init(),
        Init::Uniform {
            low: -0.5,
            high: 0.5
        }
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
fn every_tensor_a_plan_names_lies_inside_its_region() {
    for profile in every_profile() {
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
        let encoding = encoding_with(&graph, profile);
        let layout = Layout::of(&graph, ALIGNMENT, Precision::Single);
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

#[test]
fn a_write_takes_its_turn_after_every_write_it_follows() {
    let graph = Graph::new();
    let state = graph.resident(Shape::vector(4));
    let deep = {
        let mut value = graph.relu(graph.input(Shape::vector(4)));
        for _ in 0..5 {
            value = graph.relu(value);
        }
        value
    };
    let shallow = graph.fill(Shape::vector(4), 7.0);
    graph.add_into(state, deep);
    graph.copy_into(state, shallow);
    let out = graph.relu(state);
    graph.retain(state);
    graph.retain(out);
    let encoding = encoding(&graph);
    let updates = writers(&encoding, state.id());
    assert_eq!(updates.len(), 2, "two tasks update the resident tensor");
    let reader = writers(&encoding, out.id())[0];
    assert!(
        follows(&encoding, updates[0], updates[1]),
        "the copy reaches the tensor the sum already updated",
    );
    assert!(
        updates
            .iter()
            .all(|update| follows(&encoding, *update, reader)),
        "the reader reaches the tensor after every update to it",
    );
}

#[test]
fn a_chain_of_single_task_levels_rides_one_segment() {
    let graph = Graph::new();
    let mut value = graph.input(Shape::vector(64));
    for _ in 1..8 {
        value = graph.relu(graph.mul(value, value));
    }
    let out = graph.relu(value);
    graph.retain(out);
    let encoding = encoding(&graph);
    assert_eq!(encoding.task_count(), 7);
    assert_eq!(
        encoding.dispatch_count(),
        1,
        "a chain of single task levels leaves one dispatch",
    );
    assert_eq!(
        encoding.dispatches()[0].segments,
        1,
        "one workgroup carries the whole chain",
    );
    assert_eq!(encoding.segments().len(), 1);
    for task in 0..encoding.task_count() as usize {
        assert_eq!(segment_of(&encoding, task), 0);
        assert_eq!(dispatch_of(&encoding, task), 0);
    }
}

#[test]
fn a_fold_reads_a_leaf_no_later_than_the_task_it_lands_behind() {
    let graph = Graph::new();
    let state = graph.resident(Shape::vector(4));
    let bias = graph.parameter(Shape::vector(4), Init::Zero);
    let read = graph.mul(state, bias);
    let patch = graph.fill(Shape::vector(4), 7.0);
    graph.copy_into(state, patch);
    let out = graph.relu(read);
    graph.retain(out);
    let encoding = encoding(&graph);
    let writer = writers(&encoding, out.id());
    assert_eq!(writer.len(), 1);
    assert_eq!(
        kinds(&encoding)[writer[0]],
        Kind::Unary,
        "the rectifier kept a task of its own behind the write of the leaf it reads",
    );
    assert!(readers(&encoding, read.id()).contains(&writer[0]));
}

#[test]
fn a_fold_reaches_past_a_task_the_chain_does_not_read() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(8), Init::Zero);
    let right = graph.parameter(Shape::vector(8), Init::Zero);
    let product = graph.mul(left, right);
    let filler = graph.fill(Shape::vector(8), 1.0);
    let out = graph.add(product, filler);
    graph.retain(out);
    let encoding = encoding(&graph);
    assert_eq!(
        encoding.task_count(),
        2,
        "the product folded into the sum it feeds and left the filler on the tape",
    );
    assert_eq!(kinds(&encoding), vec![Kind::Fill, Kind::Binary]);
    let tape = tape(&encoding);
    assert_eq!(tape[1].out, out.id());
    assert_eq!(tape[1].steps, 1);
}

#[test]
fn a_fold_keeps_the_storage_a_view_reads_written() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(2, 3), Init::Zero);
    let right = graph.parameter(Shape::matrix(2, 3), Init::Zero);
    let product = graph.mul(left, right);
    let doubled = graph.add(product, graph.fill(Shape::matrix(2, 3), 1.0));
    let rows = graph.sum_rows(graph.transpose(product));
    graph.retain(doubled);
    graph.retain(rows);
    let encoding = encoding(&graph);
    assert_eq!(
        writers(&encoding, product.id()).len(),
        1,
        "the product a view reads kept the task that writes it",
    );
    assert_eq!(encoding.task_count(), 3);
}

#[test]
fn an_update_in_place_reads_the_tensor_it_writes_through_its_own_layout() {
    let graph = Graph::new();
    let table = graph.resident(Shape::matrix(4, 4));
    let patch = graph.parameter(Shape::matrix(4, 4), Init::Zero);
    graph.add_into(table, patch);
    assert!(
        refuses(|| {
            graph.add_into(table, graph.transpose(table));
        }),
        "an update in place accepted a transposed view of the tensor it writes",
    );
    graph.add_into(table, graph.transpose(graph.transpose(table)));
}

#[test]
fn a_shape_the_device_cannot_address_is_refused_before_it_is_built() {
    let largest = Shape::of([46340, 46340]);
    assert_eq!(largest.elements(), 2_147_395_600);
    assert_eq!(largest.dims(), [1, 1, 46340, 46340]);
    assert_eq!(largest.rows(), 46340);
    assert_eq!(largest.strides(), [0, 0, 46340, 1]);
    assert!(refuses(|| {
        let _ = Shape::of([65536, 65536]);
    }));
    assert!(refuses(|| {
        let _ = Shape::of([4096, 1024, 1024]);
    }));
    assert!(refuses(|| {
        let _ = Shape::of([65536, 1, 1, 1]).combined(Shape::of([1, 1, 1, 65536]));
    }));
    assert!(refuses(|| {
        let _ = Shape::of([0]);
    }));
    assert!(refuses(|| {
        let _ = Shape::of([1, 1, 1, 1, 1]);
    }));
    assert_eq!(Shape::of([3, 4, 5]).reduced(2).elements(), 15);
}
