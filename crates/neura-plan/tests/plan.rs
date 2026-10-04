use neura_abi::{
    Element, Kind, PatchRecord, Placement, StepRecord, Store, TaskRecord, ValueRecord, WORD_BYTES,
};
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};
use neura_pointwise as op;
use neura_profile::{
    AttentionTile, Budget, CooperativeMatrix, CooperativeTile, MatmulStrategy, MatmulTile, Profile,
};

const DEVICE: Budget = Budget::of(1024, 48 << 10);

fn narrow() -> Profile {
    Profile::derive(Budget::BASELINE, None)[0]
}

fn wide() -> Profile {
    *Profile::derive(Budget::BASELINE, None)
        .last()
        .expect("a profile")
}

fn every_profile() -> Vec<Profile> {
    Profile::derive(DEVICE, None)
}
use neura_plan::{Layout, Plan, Product};
use std::mem::size_of;

const ALIGNMENT: u64 = 256;
const PLACEMENT: Placement = Placement::new(1 << 16, 1 << 18);

fn plan(graph: &Graph) -> Plan {
    plan_with(graph, narrow())
}

fn plan_with(graph: &Graph, profile: Profile) -> Plan {
    Plan::of(graph, ALIGNMENT, profile)
}

fn records<T: bytemuck::AnyBitPattern>(bytes: &[u8], width: usize) -> Vec<T> {
    bytes
        .chunks_exact(width)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

fn tasks(plan: &Plan) -> Vec<TaskRecord> {
    records(plan.tasks(), size_of::<TaskRecord>())
}

fn steps(plan: &Plan) -> Vec<StepRecord> {
    records(plan.steps(), size_of::<StepRecord>())
}

fn kinds(plan: &Plan) -> Vec<Kind> {
    tasks(plan).iter().map(|task| Kind::of(task.kind)).collect()
}

fn segment_of(plan: &Plan, task: usize) -> usize {
    plan.segments()
        .partition_point(|segment| segment.first as usize <= task)
        - 1
}

fn patched_slots(plan: &Plan, patch: PatchRecord) -> Vec<u32> {
    plan.patch_list()[patch.slots as usize..(patch.slots + patch.slots_count) as usize].to_vec()
}

fn follows(plan: &Plan, before: usize, after: usize) -> bool {
    let segment = segment_of(plan, before);
    wave_of(plan, before) < wave_of(plan, after)
        || (segment == segment_of(plan, after)
            && before as u32 - plan.segments()[segment].first
                < after as u32 - plan.segments()[segment].first)
}

fn wave_of(plan: &Plan, task: usize) -> u32 {
    tasks(plan)[task].wave
}

fn writers(plan: &Plan, value: u32) -> Vec<usize> {
    tasks(plan)
        .iter()
        .enumerate()
        .filter(|(_, task)| task.out == value)
        .map(|(index, _)| index)
        .collect()
}

fn readers(plan: &Plan, value: u32) -> Vec<usize> {
    let steps = steps(plan);
    let tasks = tasks(plan);
    tasks
        .iter()
        .enumerate()
        .filter(|(_, task)| {
            task.a == value
                || task.b == value
                || task.c == value
                || task.origin == value
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
    let mut value = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    for _ in 0..4 {
        value = graph.relu(value);
    }
    assert_eq!(graph.task_count(), 4);
    let plan = plan(&graph);
    assert_eq!(plan.task_count(), 1);
    assert_eq!(plan.wave_count(), 1);
    assert_eq!(plan.step_count(), 3);
    assert_eq!(value.shape().elements(), 4);
}

#[test]
fn a_dense_layer_fuses_its_epilogue_into_the_product() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 8), Init::Zero, Element::Single);
    let bias = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let data = graph.input(Shape::matrix(3, 4), Element::Single);
    let activated = graph.relu(graph.add(graph.matmul(data, weight), bias));
    let plan = plan(&graph);
    assert_eq!(plan.task_count(), 1);
    assert_eq!(plan.wave_count(), 1);
    assert_eq!(kinds(&plan), vec![Kind::Matmul]);
    let steps = steps(&plan);
    let tasks = tasks(&plan);
    assert_eq!(
        tasks[0].steps, 2,
        "the bias and the rectifier ride the product"
    );
    assert_eq!(steps[tasks[0].chain as usize].op, op::ADD);
    assert_eq!(steps[tasks[0].chain as usize].operand, bias.id());
    assert_eq!(steps[tasks[0].chain as usize + 1].op, op::RELU);
    assert_eq!(steps[tasks[0].chain as usize + 1].operand, u32::MAX);
    assert_eq!(tasks[0].out, activated.id());
}

#[test]
fn a_value_two_tasks_read_stays_on_the_tape() {
    let graph = Graph::new();
    let data = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let squared = graph.mul(data, data);
    let plan = plan(&graph);
    assert_eq!(plan.task_count(), 1);
    assert_eq!(kinds(&plan), vec![Kind::Binary]);
    assert_eq!(readers(&plan, data.id()), vec![0]);
    assert_eq!(tasks(&plan)[0].out, squared.id());
}

#[test]
fn a_value_two_tasks_read_keeps_its_storage_from_the_output_of_either() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(64), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::vector(64), Init::Zero, Element::Single);
    let product = graph.mul(left, right);
    let negated = graph.neg(product);
    let squared = graph.mul(product, product);
    let plan = plan(&graph);
    let reader = writers(&plan, negated.id())[0];
    let taker = writers(&plan, squared.id())[0];
    assert_eq!(
        wave_of(&plan, reader),
        wave_of(&plan, taker),
        "both readers of a value share a wave",
    );
    assert_ne!(
        plan.span(product, PLACEMENT).offset,
        plan.span(squared, PLACEMENT).offset,
        "the storage of a value a second task reads in the same wave was handed to its reader",
    );
}

#[test]
fn a_value_one_task_reads_hands_its_storage_to_that_task() {
    let graph = Graph::new();
    let data = graph.parameter(Shape::vector(64), Init::Zero, Element::Single);
    let scaled = graph.mul(data, graph.fill(Shape::vector(64), 2.0));
    let squared = graph.mul(scaled, scaled);
    let plan = plan(&graph);
    assert_eq!(
        plan.span(scaled, PLACEMENT).offset,
        plan.span(squared, PLACEMENT).offset,
        "the only reader of a value takes the storage it consumed",
    );
}

#[test]
fn a_retained_value_is_never_folded_away() {
    let graph = Graph::new();
    let data = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let doubled = graph.mul(data, graph.fill(Shape::vector(8), 2.0));
    graph.retain(doubled);
    let activated = graph.relu(doubled);
    graph.retain(activated);
    let plan = plan(&graph);
    let reader = writers(&plan, activated.id());
    assert_eq!(reader.len(), 1, "the rectifier keeps a task of its own");
    assert_eq!(kinds(&plan)[reader[0]], Kind::Unary);
    assert!(
        readers(&plan, doubled.id()).contains(&reader[0]),
        "the rectifier reads the value the graph retains",
    );
}

#[test]
fn a_consumer_closes_the_producers_it_reads_into_one_segment() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(64), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::vector(64), Init::Zero, Element::Single);
    let sum = graph.add(left, right);
    let product = graph.mul(left, right);
    graph.retain(sum);
    graph.retain(product);
    let out = graph.add(sum, product);
    graph.retain(out);
    let plan = plan(&graph);
    assert_eq!(plan.task_count(), 3);
    assert_eq!(
        plan.wave_count(),
        1,
        "a wave the device cannot fill hands its work to the wave it reads",
    );
    assert_eq!(plan.segments().len(), 1);
    let sum_task = writers(&plan, sum.id())[0];
    let product_task = writers(&plan, product.id())[0];
    let consumer = writers(&plan, out.id())[0];
    assert!(follows(&plan, sum_task, consumer));
    assert!(follows(&plan, product_task, consumer));
    assert_eq!(out.shape(), Shape::vector(64));
}

#[test]
fn a_chain_of_temporaries_holds_one_tensor() {
    let graph = Graph::new();
    let mut value = graph.parameter(Shape::vector(256), Init::Zero, Element::Single);
    for _ in 0..16 {
        value = graph.relu(value);
    }
    let plan = plan(&graph);
    assert_eq!(plan.task_count(), 1);
    assert_eq!(
        plan.arena_bytes(),
        256 * 4,
        "the arena holds only the value the fused chain produces",
    );
    assert_eq!(
        Layout::of(&graph, ALIGNMENT).weights().words(),
        256,
        "the parameter lives in the weight store, not in the arena",
    );
}

#[test]
fn a_plan_sizes_its_own_arena() {
    let small = Graph::new();
    let parameter = small.parameter(Shape::vector(256), Init::Zero, Element::Single);
    small.relu(parameter);
    let large = Graph::new();
    let parameter = large.parameter(Shape::vector(4096), Init::Zero, Element::Single);
    large.relu(parameter);
    let small = plan(&small);
    let large = plan(&large);
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
    let left = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(2, 4), Init::Zero, Element::Single);
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
    let left = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(2, 4), Init::Zero, Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.add(left, right);
        }),
        "an add of a 2x3 and a 2x4 was accepted",
    );
}

#[test]
fn a_product_takes_the_tile_that_spends_the_least_on_the_workgroups_it_fills() {
    let graph = Graph::new();
    let tall = graph.parameter(Shape::matrix(1024, 32), Init::Zero, Element::Single);
    let weight = graph.parameter(Shape::matrix(32, 64), Init::Zero, Element::Single);
    let blocked = graph.parameter(Shape::matrix(32, 48), Init::Zero, Element::Single);
    let balanced = graph.matmul(tall, weight);
    let ragged = graph.matmul(tall, blocked);
    graph.retain(balanced);
    graph.retain(ragged);
    let plan = plan_with(&graph, wide());
    let tiles = plan.tiles();
    let tasks = tasks(&plan);
    let balanced_task = *tasks
        .iter()
        .find(|task| task.out == balanced.id())
        .expect("the 1024x64 product holds a task");
    let balanced_tile = tiles[balanced_task.geometry as usize];
    let products = tasks
        .iter()
        .filter(|task| task.out == balanced.id())
        .count() as u32;
    assert_eq!(
        (balanced_tile.rows(), balanced_tile.columns()),
        (16, 16),
        "a product fills every workgroup its profile offers with the tile that walks the fewest registers for it",
    );
    assert!(
        products >= wide().workgroups(),
        "a product fills every workgroup its profile runs at once with at least one tile",
    );
    assert_eq!(balanced_tile.registers(), 1);
    let ragged_task = *tasks
        .iter()
        .find(|task| task.out == ragged.id())
        .expect("the 1024x48 product holds a task");
    assert_eq!(
        tiles[ragged_task.geometry as usize], balanced_tile,
        "a product no tile divides is masked by the tile its shape pays the least for",
    );
    let ragged_products = tasks.iter().filter(|task| task.out == ragged.id()).count() as u32;
    assert_eq!(
        plan.matmul_geometries(),
        vec![(balanced_tile, products + ragged_products)],
        "a plan reports the geometry of every task it hands the device",
    );
    assert_eq!(
        plan.work(),
        u64::from(products + ragged_products) * balanced_tile.tile_work(),
        "a plan accounts the tile work it schedules",
    );
    assert_eq!(
        plan.tiles(),
        wide().tiles(),
        "a plan carries every tile of the profile it compiles for",
    );
    assert_ne!(
        plan_with(&graph, narrow()).tiles(),
        plan_with(&graph, wide()).tiles(),
        "a profile decides which tiles a product is tiled with",
    );
}

#[test]
fn a_product_whose_output_is_narrow_splits_its_depth_across_tasks() {
    let graph = Graph::new();
    let narrow = graph.parameter(Shape::matrix(8, 4096), Init::Zero, Element::Single);
    let weight = graph.parameter(Shape::matrix(4096, 32), Init::Zero, Element::Single);
    let out = graph.matmul(narrow, weight);
    graph.retain(out);
    let plan = plan_with(&graph, wide());
    let tasks = tasks(&plan);
    assert_eq!(out.shape(), Shape::matrix(8, 32));
    let products = tasks
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::Matmul)
        .collect::<Vec<_>>();
    let folds = tasks
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::MatmulFold)
        .collect::<Vec<_>>();
    assert_eq!(folds.len(), 1);
    let tile = plan.tiles()[products[0].geometry as usize];
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
        wave_of(&plan, tasks.len() - 1),
        plan.wave_count() - 1,
        "the fold reads every slot after the last slot is written",
    );
}

#[test]
fn a_fold_leaves_the_wave_that_already_fills_the_device() {
    let graph = Graph::new();
    let narrow = graph.parameter(Shape::matrix(8, 4096), Init::Zero, Element::Single);
    let weight = graph.parameter(Shape::matrix(4096, 32), Init::Zero, Element::Single);
    let out = graph.matmul(narrow, weight);
    graph.retain(out);
    let plan = plan_with(&graph, wide());
    let tasks = tasks(&plan);
    let fold = tasks
        .iter()
        .position(|task| Kind::of(task.kind) == Kind::MatmulFold)
        .expect("a product that splits its depth carries a fold");
    assert_ne!(
        wave_of(&plan, fold),
        wave_of(&plan, 0),
        "a fold never lengthens the wave its own partials already fill",
    );
}

#[test]
fn a_split_product_hands_its_epilogue_to_the_fold() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4096, 32), Init::Zero, Element::Single);
    let bias = graph.parameter(Shape::vector(32), Init::Zero, Element::Single);
    let data = graph.input(Shape::matrix(8, 4096), Element::Single);
    let out = graph.relu(graph.add(graph.matmul(data, weight), bias));
    graph.retain(out);
    let plan = plan_with(&graph, wide());
    let tasks = tasks(&plan);
    let folds = tasks
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::MatmulFold)
        .collect::<Vec<_>>();
    assert_eq!(folds.len(), 1);
    assert_eq!(folds[0].out, out.id());
    assert_eq!(
        folds[0].steps, 2,
        "the bias and the rectifier ride the fold"
    );
    let steps = steps(&plan);
    assert_eq!(steps[folds[0].chain as usize].op, op::ADD);
    assert_eq!(steps[folds[0].chain as usize].operand, bias.id());
    assert_eq!(steps[folds[0].chain as usize + 1].op, op::RELU);
    for task in tasks
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
        let data = graph.input(Shape::vector(elements), Element::Single);
        graph.relu(data);
        let plan = plan(&graph);
        assert_eq!(
            plan.task_count(),
            tasks,
            "a rectifier over {elements} elements hands the device {tasks} tasks",
        );
    }
}

#[test]
fn every_profile_plans_the_same_values() {
    for profile in every_profile() {
        let graph = Graph::new();
        let weight = graph.parameter(Shape::matrix(4, 8), Init::Zero, Element::Single);
        let bias = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
        let data = graph.input(Shape::matrix(2, 4), Element::Single);
        let out = graph.relu(graph.add(graph.matmul(data, weight), bias));
        let plan = plan_with(&graph, profile);
        assert_eq!(
            plan.value_count() as usize,
            graph.value_count(),
            "{profile:?} publishes values a graph without a reduction holds",
        );
        assert_eq!(plan.span(out, PLACEMENT).elements, 16);
    }
}

#[test]
fn a_backward_pass_reaches_every_parameter() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 8), Init::Zero, Element::Single);
    let bias = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let input = graph.input(Shape::matrix(16, 4), Element::Single);
    let dense = graph.matmul(input, weight);
    let shifted = graph.add(dense, bias);
    let activated = graph.relu(shifted);
    let loss = graph.sum(activated);
    let grads = graph.backward(loss);
    assert_eq!(grads.of(weight).shape(), Shape::matrix(4, 8));
    assert_eq!(grads.of(bias).shape(), Shape::vector(8));

    let kinds = kinds(&plan(&graph));
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
    let gradient = grads.of(bias).id();
    let folded = tasks(&plan(&graph))
        .into_iter()
        .filter(|task| Kind::of(task.kind) == Kind::SumAxis)
        .collect::<Vec<_>>();
    assert!(
        !folded.is_empty() && folded.iter().all(|task| task.out == gradient),
        "the bias gradient folds the rows it was spread over",
    );
    assert_eq!(
        folded.iter().map(|task| task.count).sum::<u32>(),
        bias.shape().elements(),
        "a fold hands every row of its gradient a task",
    );
}

#[test]
fn a_broadcast_product_folds_its_gradient_back_to_the_shape_of_its_operand() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([4, 1, 3, 2]), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::of([1, 5, 2, 3]), Init::Zero, Element::Single);
    let product = graph.matmul(left, right);
    assert_eq!(product.shape(), Shape::of([4, 5, 3, 3]));
    let grads = graph.backward(graph.sum(product));
    assert_eq!(
        grads.of(left).shape(),
        left.shape(),
        "a gradient of a broadcast operand carries the shape of that operand",
    );
    assert_eq!(grads.of(right).shape(), right.shape());
    let folded = tasks(&plan(&graph))
        .into_iter()
        .filter(|task| Kind::of(task.kind) == Kind::SumAxis)
        .collect::<Vec<_>>();
    for gradient in [grads.of(left), grads.of(right)] {
        assert_eq!(
            folded
                .iter()
                .filter(|task| task.out == gradient.id())
                .map(|task| task.count)
                .sum::<u32>(),
            gradient.shape().elements(),
            "one fold returns each operand to the rows its product spread",
        );
    }
}

#[test]
fn a_reduction_folds_through_as_many_levels_as_it_takes() {
    let graph = Graph::new();
    let wide = graph.parameter(Shape::vector(1 << 21), Init::Zero, Element::Single);
    let loss = graph.sum(wide);
    let plan = plan(&graph);
    let reductions = kinds(&plan)
        .iter()
        .filter(|kind| **kind == Kind::SumChunk)
        .count();
    assert_eq!(
        reductions, 257,
        "a sum folds a chunk of 8192 elements at a time until one scalar stands",
    );
    assert_eq!(plan.wave_count(), 2, "every level of the fold is a wave");
    assert_eq!(loss.shape(), Shape::scalar());
    assert!(
        plan.value_count() as usize > graph.value_count(),
        "a folded reduction publishes its partial sums",
    );
}

#[test]
fn a_parameter_updated_in_place_feeds_the_tasks_that_follow() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let input = graph.input(Shape::vector(8), Element::Single);
    let loss = graph.sum(graph.mul(input, weight));
    let grads = graph.backward(loss);
    let scaled = graph.mul(grads.of(weight), graph.fill(Shape::scalar(), -0.1));
    graph.add_into(weight, scaled);
    let read_back = graph.relu(weight);
    graph.retain(read_back);
    let plan = plan(&graph);
    let update = writers(&plan, weight.id());
    assert!(!update.is_empty(), "the update writes the parameter");
    let reader = writers(&plan, read_back.id());
    assert_eq!(reader.len(), 1, "the reader holds a task of its own");
    assert!(
        update.iter().all(|task| follows(&plan, *task, reader[0])),
        "a reader observes the update it follows",
    );
}

#[test]
fn an_update_in_place_follows_every_reader_of_the_value_it_replaces() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let input = graph.input(Shape::vector(8), Element::Single);
    let read = graph.mul(input, weight);
    graph.retain(read);
    let loss = graph.sum(read);
    let grads = graph.backward(loss);
    graph.add_into(weight, grads.of(weight));
    let read_back = graph.relu(weight);
    graph.retain(read_back);
    let plan = plan(&graph);
    let update = writers(&plan, weight.id());
    assert_eq!(update.len(), 1, "the update writes the parameter once");
    let before = writers(&plan, read.id())[0];
    let after = writers(&plan, read_back.id())[0];
    assert!(
        follows(&plan, before, update[0]),
        "an update in place follows every reader of the value it replaces",
    );
    assert!(
        follows(&plan, update[0], after),
        "a reader that follows the update observes it",
    );
}

#[test]
fn a_graph_updated_in_place_is_refused_a_later_backward() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
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
    let matrix = graph.parameter(Shape::matrix(8, 4), Init::Zero, Element::Single);
    let transposed = graph.permute(matrix, [0, 1, 3, 2]);
    assert_eq!(transposed.shape(), Shape::matrix(4, 8));
    let plan = plan(&graph);
    assert_eq!(plan.span(matrix, PLACEMENT).elements, 32);
    assert!(
        refuses(|| {
            let _ = plan.span(transposed, PLACEMENT);
        }),
        "a transposed view was handed its own storage",
    );
    assert!(!plan.readable(transposed));
}

#[test]
fn a_view_that_its_storage_addresses_row_by_row_reads_back_as_that_storage() {
    let graph = Graph::new();
    let matrix = graph.input(Shape::matrix(8, 4), Element::Single);
    let flattened = graph.reshape(matrix, Shape::vector(32));
    let weight = graph.parameter(Shape::matrix(4, 2), Init::Zero, Element::Single);
    let squared = graph.reshape(weight, Shape::matrix(2, 4));
    let plan = plan(&graph);
    for (view, storage) in [(flattened, matrix), (squared, weight)] {
        assert_eq!(plan.span(view, PLACEMENT), plan.span(storage, PLACEMENT));
        assert!(plan.readable(view));
    }
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
        Element::Single,
    );
    let probabilities = graph.softmax(logits);
    assert_eq!(probabilities.shape(), Shape::matrix(16, 8));
    let grads = graph.backward(graph.sum(probabilities));
    assert_eq!(grads.of(logits).shape(), Shape::matrix(16, 8));
}

#[test]
fn a_graph_is_differentiated_once() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
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
    let data = graph.input(Shape::vector(4), Element::Single);
    let loss = graph.sum(graph.relu(data));
    assert!(
        refuses(|| {
            let _ = graph.backward(loss);
        }),
        "a loss without parameters was differentiated",
    );
}

#[test]
fn a_narrow_product_streams_the_operands_a_wide_one_stages() {
    let narrow_graph = Graph::new();
    let narrow_weight =
        narrow_graph.parameter(Shape::matrix(512, 512), Init::Zero, Element::Single);
    let narrow_data = narrow_graph.parameter(Shape::matrix(1, 512), Init::Zero, Element::Single);
    narrow_graph.retain(narrow_graph.matmul(narrow_data, narrow_weight));
    let narrow = plan_with(&narrow_graph, wide());
    assert!(!narrow.matmul_geometries().is_empty());
    assert!(
        narrow
            .matmul_geometries()
            .iter()
            .all(|(tile, _)| tile.strategy() == MatmulStrategy::Streamed),
        "a product of one row stages its operands through a workgroup instead of streaming them: {:?}",
        narrow.matmul_geometries(),
    );
    assert!(
        narrow
            .matmul_geometries()
            .iter()
            .all(|(tile, _)| tile.shared_bytes() == 0),
        "a streamed product stages an operand beside the registers it accumulates in",
    );

    let wide_graph = Graph::new();
    let wide_weight = wide_graph.parameter(Shape::matrix(1024, 1024), Init::Zero, Element::Single);
    let wide_data = wide_graph.parameter(Shape::matrix(64, 1024), Init::Zero, Element::Single);
    wide_graph.retain(wide_graph.matmul(wide_data, wide_weight));
    let wide = plan_with(&wide_graph, wide());
    assert!(!wide.matmul_geometries().is_empty());
    assert!(
        wide.matmul_geometries()
            .iter()
            .all(|(tile, _)| tile.strategy() == MatmulStrategy::Staged),
        "a product of many rows reuses its operands through a workgroup of shared memory: {:?}",
        wide.matmul_geometries(),
    );
    assert!(
        wide.matmul_geometries()
            .iter()
            .all(|(tile, _)| tile.shared_bytes() > 0),
        "a staged product stages nothing into the shared memory it asks the device for",
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
        Element::Single,
    );
    let input = graph.input(Shape::matrix(2, 4), Element::Single);
    let out = graph.softmax(graph.matmul(input, weight));
    graph.backward(graph.sum(out));
    let plan = plan(&graph);
    assert!(plan.value_count() as usize >= graph.value_count());
    assert!(plan.arena_bytes() > 0);
    assert!(plan.work() > 0);
    let layout = Layout::of(&graph, ALIGNMENT);
    let seed = layout
        .seeds()
        .iter()
        .find(|seed| {
            layout.weight_bytes(PLACEMENT, seed.address()) == plan.span(weight, PLACEMENT).offset
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
    let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
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
        let weight = graph.parameter(Shape::matrix(8, 4), Init::Zero, Element::Single);
        let data = graph.input(Shape::matrix(2, 8), Element::Single);
        let out = graph.softmax(graph.add(
            graph.matmul(data, weight),
            graph.fill(Shape::vector(4), 1.0),
        ));
        graph.retain(out);
        let grads = graph.backward(graph.sum(out));
        graph.retain(grads.of(weight));
        let plan = plan_with(&graph, profile);
        let layout = Layout::of(&graph, ALIGNMENT);
        for value in [weight, data, out, grads.of(weight)] {
            let span = plan.span(value, PLACEMENT);
            let bytes = u64::from(span.elements) * WORD_BYTES;
            let (base, limit) = match span.store {
                Store::Tensors => (PLACEMENT.tensors() * WORD_BYTES, plan.tensor_bytes()),
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
    let left = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let bias = graph.input(Shape::vector(4), Element::Single);
    let difference = graph.sub(bias, graph.mul(left, right));
    let quotient = graph.div(graph.mul(right, left), bias);
    graph.retain(difference);
    graph.retain(quotient);
    let plan = plan(&graph);
    assert_eq!(plan.task_count(), 2);
    let tasks = tasks(&plan);
    let steps = steps(&plan);
    assert_eq!(tasks[0].op, op::MUL);
    let folded_into_a_subtraction = steps[tasks[0].chain as usize];
    assert_eq!(folded_into_a_subtraction.op, op::SUB);
    assert_eq!(folded_into_a_subtraction.operand, bias.id());
    assert_eq!(
        folded_into_a_subtraction.swapped, 1,
        "a product folded into a subtraction is its subtrahend",
    );
    assert_eq!(tasks[1].op, op::MUL);
    let folded_into_a_quotient = steps[tasks[1].chain as usize];
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
    let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let scale = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let activated = graph.tanh(weight);
    let scaled = graph.mul(activated, scale);
    let loss = graph.sum(scaled);
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(weight));
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    let tangent = tasks
        .iter()
        .find(|task| task.op == op::TANH)
        .expect("the tangent keeps a task of its own while the product differentiates it");
    assert_eq!(tangent.out, activated.id());
    let partial = tasks
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
    let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let other = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let magnitude = graph.abs(weight);
    let scaled = graph.mul(magnitude, other);
    let loss = graph.sum(scaled);
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(weight));
    graph.retain(gradients.of(other));
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    let absolute = tasks
        .iter()
        .find(|task| task.op == op::ABS && Kind::of(task.kind) == Kind::Partial)
        .expect("a magnitude sign reaches the operand it was taken from");
    assert_eq!(
        absolute.a,
        weight.id(),
        "a magnitude partial has to read the operand it differentiates",
    );
    assert_eq!(absolute.b, neura_abi::NO_VALUE);
    let products = tasks
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
    let state = graph.resident(Shape::vector(4), Element::Single);
    let deep = {
        let mut value = graph.relu(graph.input(Shape::vector(4), Element::Single));
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
    let plan = plan(&graph);
    let updates = writers(&plan, state.id());
    assert_eq!(updates.len(), 2, "two tasks update the resident tensor");
    let reader = writers(&plan, out.id())[0];
    assert!(
        follows(&plan, updates[0], updates[1]),
        "the copy reaches the tensor the sum already updated",
    );
    assert!(
        updates.iter().all(|update| follows(&plan, *update, reader)),
        "the reader reaches the tensor after every update to it",
    );
}

#[test]
fn an_update_in_place_spreads_the_spans_it_names_over_workgroups() {
    let graph = Graph::new();
    let state = graph.state(Shape::vector(200_000), Init::Zero, Element::Single);
    graph.mul_into(state, graph.fill(Shape::scalar(), 0.5));
    let plan = plan(&graph);
    let updates = writers(&plan, state.id());
    assert!(
        updates.len() > 1,
        "one update in place of one tensor spans the work its decomposition names",
    );
    assert_eq!(
        plan.wave_count(),
        2,
        "the constant the update reads is a wave of its own",
    );
    let wave = wave_of(&plan, updates[0]);
    assert!(
        updates.iter().all(|task| wave_of(&plan, *task) == wave),
        "the spans of one update in place meet in one wave",
    );
    let mut workgroups = updates
        .iter()
        .map(|task| segment_of(&plan, *task))
        .collect::<Vec<_>>();
    workgroups.sort_unstable();
    workgroups.dedup();
    assert_eq!(
        workgroups.len(),
        updates.len(),
        "every span of an update in place rides a workgroup of its own",
    );
}

#[test]
fn a_chain_of_single_task_levels_rides_one_segment() {
    let graph = Graph::new();
    let mut value = graph.input(Shape::vector(64), Element::Single);
    for _ in 1..8 {
        value = graph.relu(graph.mul(value, value));
    }
    let out = graph.relu(value);
    graph.retain(out);
    let plan = plan(&graph);
    assert_eq!(plan.task_count(), 7);
    assert_eq!(
        plan.wave_count(),
        1,
        "a chain of single task levels leaves one wave",
    );
    assert_eq!(
        plan.segments().len(),
        1,
        "one workgroup carries the whole chain",
    );
    for task in 0..plan.task_count() as usize {
        assert_eq!(segment_of(&plan, task), 0);
        assert_eq!(wave_of(&plan, task), 0);
    }
}

#[test]
fn a_fold_reads_a_leaf_no_later_than_the_task_it_lands_behind() {
    let graph = Graph::new();
    let state = graph.resident(Shape::vector(4), Element::Single);
    let bias = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let read = graph.mul(state, bias);
    let patch = graph.fill(Shape::vector(4), 7.0);
    graph.copy_into(state, patch);
    let out = graph.relu(read);
    graph.retain(out);
    let plan = plan(&graph);
    let writer = writers(&plan, out.id());
    assert_eq!(writer.len(), 1);
    assert_eq!(
        kinds(&plan)[writer[0]],
        Kind::Unary,
        "the rectifier kept a task of its own behind the write of the leaf it reads",
    );
    assert!(readers(&plan, read.id()).contains(&writer[0]));
}

#[test]
fn a_fold_reaches_past_a_task_the_chain_does_not_read() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let product = graph.mul(left, right);
    let filler = graph.fill(Shape::vector(8), 1.0);
    let out = graph.add(product, filler);
    graph.retain(out);
    let plan = plan(&graph);
    assert_eq!(
        plan.task_count(),
        2,
        "the product folded into the sum it feeds and left the filler on the plan",
    );
    assert_eq!(kinds(&plan), vec![Kind::Fill, Kind::Binary]);
    let tasks = tasks(&plan);
    assert_eq!(tasks[1].out, out.id());
    assert_eq!(tasks[1].steps, 1);
}

#[test]
fn a_reduction_opens_the_chain_it_would_have_materialized() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(1, 8), Element::Single);
    let squared = graph.mul(data, data);
    let rows = graph.sum_rows(squared);
    graph.retain(rows);
    let plan = plan(&graph);
    assert_eq!(kinds(&plan), vec![Kind::SumAxis]);
    let tasks = tasks(&plan);
    assert_eq!(
        tasks[0].prelude_steps, 1,
        "the fold of the reduction opens the product it would have read",
    );
    assert_eq!(tasks[0].a, data.id(), "the reduction reads the operand");
    assert_eq!(tasks[0].out, rows.id(), "the reduction writes its own row");
    assert_eq!(writers(&plan, squared.id()), Vec::<usize>::new());
    let opened = steps(&plan)[tasks[0].prelude as usize];
    assert_eq!(opened.op, op::MUL);
    assert_eq!(opened.operand, data.id());
    assert_eq!(opened.swapped, 0);
}

#[test]
fn a_pinned_product_keeps_the_task_that_writes_it() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(1, 8), Element::Single);
    let squared = graph.mul(data, data);
    graph.retain(squared);
    graph.retain(graph.sum_rows(squared));
    let plan = plan(&graph);
    assert_eq!(kinds(&plan), vec![Kind::Binary, Kind::SumAxis]);
    assert!(
        tasks(&plan).iter().all(|task| task.prelude_steps == 0),
        "a reduction opens no tensor another task still reads",
    );
    assert_eq!(writers(&plan, squared.id()).len(), 1);
}

#[test]
fn a_task_that_reads_its_source_more_than_once_keeps_it_materialized() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(1, 8), Element::Single);
    let logits = graph.exp(data);
    graph.retain(graph.softmax(logits));
    let plan = plan(&graph);
    assert_eq!(kinds(&plan), vec![Kind::Unary, Kind::Softmax]);
    assert!(
        tasks(&plan).iter().all(|task| task.prelude_steps == 0),
        "a row a task walks three times holds the task that wrote it",
    );
    assert_eq!(writers(&plan, logits.id()).len(), 1);
}

#[test]
fn a_reduction_that_opens_a_chain_also_carries_its_epilogue() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(1, 8), Element::Single);
    let squared = graph.mul(data, data);
    graph.retain(graph.relu(graph.sum_rows(squared)));
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    assert_eq!(plan.task_count(), 1);
    assert_eq!(tasks[0].prelude_steps, 1);
    assert_eq!(tasks[0].steps, 1);
    let steps = steps(&plan);
    assert_eq!(steps[tasks[0].prelude as usize].op, op::MUL);
    assert_eq!(steps[tasks[0].chain as usize].op, op::RELU);
    assert_ne!(
        tasks[0].prelude, tasks[0].chain,
        "the chain a task opens stands beside the chain it closes",
    );
}

#[test]
fn a_product_two_reductions_read_keeps_the_task_that_writes_it() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(8), Element::Single);
    let squared = graph.mul(data, data);
    graph.retain(graph.sum(squared));
    graph.retain(graph.sum_rows(squared));
    let plan = plan(&graph);
    assert_eq!(
        kinds(&plan),
        vec![Kind::Binary, Kind::SumChunk, Kind::SumAxis],
    );
    assert!(
        tasks(&plan).iter().all(|task| task.prelude_steps == 0),
        "a reduction opens only a tensor no other task reads",
    );
}

#[test]
fn an_opened_reduction_hands_its_arena_back_to_the_product() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(4, 256), Element::Single);
    graph.retain(graph.sum_rows(graph.mul(data, data)));
    let opened = plan(&graph);
    let pinned = Graph::new();
    let data = pinned.input(Shape::matrix(4, 256), Element::Single);
    let squared = pinned.mul(data, data);
    pinned.retain(squared);
    pinned.retain(pinned.sum_rows(squared));
    assert_eq!(opened.task_count(), 4);
    assert!(
        opened.arena_bytes() < plan(&pinned).arena_bytes(),
        "the product a reduction opens holds no tensor of its own",
    );
}

#[test]
fn a_fold_keeps_the_storage_a_view_reads_written() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    let product = graph.mul(left, right);
    let doubled = graph.add(product, graph.fill(Shape::matrix(2, 3), 1.0));
    let rows = graph.sum_rows(graph.permute(product, [0, 1, 3, 2]));
    graph.retain(doubled);
    graph.retain(rows);
    let plan = plan(&graph);
    assert_eq!(
        writers(&plan, product.id()).len(),
        1,
        "the product a view reads kept the task that writes it",
    );
    assert_eq!(
        tasks(&plan)
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::SumAxis)
            .map(|task| task.count)
            .sum::<u32>(),
        rows.shape().elements(),
        "a fold over a view covers every row it walks",
    );
}

#[test]
fn an_update_in_place_reads_the_tensor_it_writes_through_its_own_layout() {
    let graph = Graph::new();
    let table = graph.resident(Shape::matrix(4, 4), Element::Single);
    let patch = graph.parameter(Shape::matrix(4, 4), Init::Zero, Element::Single);
    graph.add_into(table, patch);
    assert!(
        refuses(|| {
            graph.add_into(table, graph.permute(table, [0, 1, 3, 2]));
        }),
        "an update in place accepted a transposed view of the tensor it writes",
    );
    graph.add_into(
        table,
        graph.permute(graph.permute(table, [0, 1, 3, 2]), [0, 1, 3, 2]),
    );
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

#[test]
fn a_narrow_parameter_update_packs_the_image_the_chain_folded() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(300), Init::Zero, Element::Half);
    graph.mul_into(weight, graph.fill(Shape::vector(300), 2.0));
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    let values = records::<ValueRecord>(plan.values(), size_of::<ValueRecord>());
    assert_eq!(values[weight.id() as usize].element, Element::Half.code());
    assert_eq!(values[weight.id() as usize].store, Store::Weights.code());

    let convert = tasks
        .iter()
        .find(|task| Kind::of(task.kind) == Kind::Convert)
        .expect("a half precision parameter packs the image its update computed");
    assert_eq!(convert.out, weight.id());
    assert_eq!(convert.first, 0);
    assert_eq!(convert.count, 150);
    assert_eq!(convert.prelude_steps, 0);
    let image = values[convert.a as usize];
    assert_eq!(image.element, Element::Single.code());
    assert_eq!(image.store, Store::Tensors.code());
    assert_eq!(image.dims, Shape::vector(300).dims());

    let writer = tasks
        .iter()
        .find(|task| task.out == convert.a)
        .expect("the image holds the product the update computed");
    assert_eq!(Kind::of(writer.kind), Kind::Fill);
    assert_eq!(writer.steps, 1);
    let step = steps(&plan)[writer.chain as usize];
    assert_eq!(step.op, op::MUL);
    assert_eq!(step.operand, weight.id());
    assert_eq!(step.swapped, 1);
    assert_eq!(
        tasks.len(),
        2,
        "the fill and the convert that packs its product are the whole plan",
    );
}

#[test]
fn a_wide_narrow_update_in_place_spreads_the_words_it_packs() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(200_000), Init::Zero, Element::Half);
    let factor = graph.input(Shape::vector(200_000), Element::Single);
    graph.mul_into(weight, factor);
    let plan = plan(&graph);
    let packs = writers(&plan, weight.id());
    assert!(
        packs.len() > 1,
        "a wide narrow write packs the words its decomposition names",
    );
    let wave = wave_of(&plan, packs[0]);
    assert!(
        packs.iter().all(|task| wave_of(&plan, *task) == wave),
        "the words of one narrow write meet in one wave",
    );
    let mut workgroups = packs
        .iter()
        .map(|task| segment_of(&plan, *task))
        .collect::<Vec<_>>();
    workgroups.sort_unstable();
    workgroups.dedup();
    assert_eq!(
        workgroups.len(),
        packs.len(),
        "every word of a narrow write rides a workgroup of its own",
    );
}

#[test]
fn a_narrow_update_in_place_packs_the_word_it_reads() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(300), Init::Zero, Element::Half);
    let factor = graph.input(Shape::vector(300), Element::Single);
    graph.mul_into(weight, factor);
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    assert_eq!(
        tasks.len(),
        1,
        "an update in place of a narrow tensor reads and writes the very word its convert packs",
    );
    let convert = &tasks[0];
    assert_eq!(Kind::of(convert.kind), Kind::Convert);
    assert_eq!(convert.a, weight.id());
    assert_eq!(convert.out, weight.id());
    assert_eq!(convert.first, 0);
    assert_eq!(convert.count, 150);
    assert_eq!(convert.steps, 1);
    let step = steps(&plan)[convert.chain as usize];
    assert_eq!(step.op, op::MUL);
    assert_eq!(step.operand, factor.id());
    assert_eq!(step.swapped, 0);
}

#[test]
fn a_narrow_product_packs_the_image_it_computed() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(4, 8), Init::Zero, Element::Half);
    let right = graph.parameter(Shape::matrix(8, 16), Init::Zero, Element::Half);
    let product = graph.matmul(left, right);
    assert_eq!(graph.element(product), Element::Half);
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    let values = records::<ValueRecord>(plan.values(), size_of::<ValueRecord>());
    assert_eq!(
        values[product.id() as usize].element,
        Element::Half.code(),
        "a product of two narrow tensors keeps the numbers its operands carry",
    );
    let convert = tasks
        .iter()
        .find(|task| Kind::of(task.kind) == Kind::Convert)
        .expect("a narrow product packs the image its tiles wrote");
    assert_eq!(convert.out, product.id());
    assert_eq!(convert.first, 0);
    assert_eq!(convert.count, 32);
    assert_eq!(convert.prelude_steps, 0);
    let image = values[convert.a as usize];
    assert_eq!(image.element, Element::Single.code());
    assert_eq!(image.dims, Shape::matrix(4, 16).dims());
    assert_eq!(
        Shape::of(values[product.id() as usize].dims).elements(),
        64,
        "the image holds one number per element of the product",
    );
    tasks
        .iter()
        .find(|task| task.out == convert.a && Kind::of(task.kind) == Kind::Matmul)
        .expect("the image holds the product the tiles computed");
}

#[test]
fn a_quantized_parameter_packs_four_numbers_a_word_beside_its_quantum() {
    let graph = Graph::new();
    let weight = graph.quantized_parameter(Shape::vector(300), Init::Zero, 0.25);
    assert_eq!(graph.element(weight), Element::Int8);
    assert_eq!(graph.scale(weight), 0.25);
    let instead = Graph::new();
    instead.parameter(Shape::vector(300), Init::Zero, Element::Single);
    let single_precision = plan(&instead);
    let plan = plan(&graph);
    let values = records::<ValueRecord>(plan.values(), size_of::<ValueRecord>());
    let record = values[weight.id() as usize];
    assert_eq!(record.element, Element::Int8.code());
    assert_eq!(record.store, Store::Weights.code());
    assert_eq!(
        record.table,
        Element::Int8.payload_words(300) as u32,
        "the quantum a quantized tensor reconstructs by stands at the end of the words its numbers pack into",
    );
    assert_eq!(plan.weights().words(), Element::Int8.storage_words(300));
    assert_eq!(single_precision.weights().words(), 300);
}

#[test]
fn a_quantized_image_packs_four_numbers_a_word() {
    let graph = Graph::new();
    let source = graph.parameter(Shape::vector(300), Init::Zero, Element::Single);
    let quantized = graph.quantize(source, 0.125);
    graph.retain(quantized);
    let instead = Graph::new();
    let source = instead.parameter(Shape::vector(300), Init::Zero, Element::Single);
    instead.retain(instead.cast(source, Element::Half));
    let half_precision = plan(&instead);
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    let values = records::<ValueRecord>(plan.values(), size_of::<ValueRecord>());
    let record = values[quantized.id() as usize];
    assert_eq!(record.element, Element::Int8.code());
    assert_eq!(record.table, Element::Int8.payload_words(300) as u32);
    assert_eq!(record.store, Store::Tensors.code());
    let convert = tasks
        .iter()
        .find(|task| Kind::of(task.kind) == Kind::Convert)
        .expect("a quantized tensor packs the numbers its image computed");
    assert_eq!(convert.out, quantized.id());
    assert_eq!(convert.a, source.id());
    assert_eq!(convert.first, 0);
    assert_eq!(convert.count, 75);
    assert_eq!(steps(&plan)[convert.chain as usize].op, op::IDENTITY);
    assert_eq!(
        plan.arena_bytes(),
        Element::Int8.storage_words(300) * WORD_BYTES,
        "the arena holds a word of every four numbers a quantized tensor carries beside the quantum they share",
    );
    assert_eq!(
        half_precision.arena_bytes(),
        Element::Half.storage_words(300) * WORD_BYTES,
    );
    let span = plan.span(quantized, PLACEMENT);
    assert_eq!(
        span.payload_bytes(),
        Element::Int8.payload_words(300) * WORD_BYTES,
        "a quantized image packs a word of every four numbers it holds",
    );
    assert_eq!(
        span.table_offset(),
        span.payload_bytes(),
        "a tensor of the shape its bound declares stands quantum beside payload",
    );
    assert_eq!(span.table_bytes(), WORD_BYTES);
    assert_eq!(
        span.image_bytes(),
        Element::Int8.storage_words(300) * WORD_BYTES,
    );
}

#[test]
fn a_cast_declares_the_numbers_a_tensor_carries() {
    let graph = Graph::new();
    let wide = graph.input(Shape::vector(6), Element::Single);
    let narrow = graph.cast(wide, Element::Half);
    let folded = graph.sum(narrow);
    assert_eq!(graph.element(narrow), Element::Half);
    assert_eq!(
        graph.element(graph.cast(narrow, Element::Half)),
        Element::Half
    );
    assert_eq!(graph.element(graph.add(narrow, narrow)), Element::Half);
    assert_eq!(graph.element(graph.add(narrow, wide)), Element::Single);
    assert_eq!(graph.element(graph.relu(narrow)), Element::Half);
    assert_eq!(graph.element(folded), Element::Single);
    assert_eq!(
        graph.cast(narrow, Element::Half),
        narrow,
        "a cast that asks for the numbers a tensor already carries adds no task",
    );
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    let convert = tasks
        .iter()
        .find(|task| Kind::of(task.kind) == Kind::Convert)
        .expect("a cast to half packs the numbers it narrows");
    assert_eq!(convert.out, narrow.id());
    assert_eq!(convert.a, wide.id());
    assert_eq!(convert.count, 3);
    assert_eq!(convert.steps, 1);
    assert_eq!(steps(&plan)[convert.chain as usize].op, op::IDENTITY);
    let widened = tasks
        .iter()
        .find(|task| task.out == folded.id())
        .expect("a sum reads the narrow tensor it folds");
    assert_eq!(Kind::of(widened.kind), Kind::SumChunk);
}

fn narrow_rows_through_an_image(kind: Kind) {
    let graph = Graph::new();
    let table = graph.parameter(Shape::matrix(4, 2), Init::Zero, Element::Half);
    let indices = graph.input(Shape::matrix(3, 1), Element::Single);
    let updates = graph.input(Shape::matrix(3, 2), Element::Single);
    match kind {
        Kind::Scatter => graph.scatter_into(table, indices, updates),
        Kind::ScatterWrite => graph.write_into(table, indices, updates),
        other => panic!("a {} task reaches no row of a table", other.name()),
    }
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    let values = records::<ValueRecord>(plan.values(), size_of::<ValueRecord>());
    let convert = tasks
        .iter()
        .find(|task| Kind::of(task.kind) == Kind::Convert)
        .expect("a half precision table packs the rows its task reached");
    assert_eq!(convert.out, table.id());
    assert_eq!(convert.count, 4);
    assert_eq!(values[convert.a as usize].store, Store::Tensors.code());

    let rows = tasks
        .iter()
        .find(|task| Kind::of(task.kind) == kind)
        .expect("the task reaches the rows of the image");
    assert_eq!(rows.out, convert.a);
    let copy = tasks
        .iter()
        .find(|task| task.out == convert.a && Kind::of(task.kind) == Kind::Unary)
        .expect("the image holds the table before the task reaches it");
    assert_eq!(copy.a, table.id());
    assert_eq!(copy.param, 0.0);
    assert_eq!(copy.op, op::IDENTITY);
}

#[test]
fn a_narrow_parameter_reaches_the_rows_it_names_through_an_image() {
    narrow_rows_through_an_image(Kind::Scatter);
    narrow_rows_through_an_image(Kind::ScatterWrite);
}

#[test]
fn a_cursor_stays_in_the_plan_the_block_it_starts_from_reads_it() {
    let graph = Graph::new();
    let queries = graph.parameter(Shape::of([1, 1, 4, 4]), Init::Zero, Element::Single);
    let keys = graph.parameter(Shape::of([1, 1, 4, 4]), Init::Zero, Element::Single);
    let cursor = graph.fill(Shape::scalar(), 2.0);
    let out = graph.attention(
        queries,
        keys,
        keys,
        neura_graph::AttentionOptions {
            scale: 0.5,
            causal: true,
            origin: Some(cursor),
            segments: None,
            reach: None,
        },
    );
    graph.retain(out);
    let plan = plan(&graph);
    let written = writers(&plan, cursor.id());
    assert_eq!(
        written.len(),
        1,
        "a cursor is a tensor the task that walks a block reads, not a chain it folds away",
    );
    let attended = readers(&plan, cursor.id());
    let attention = attended
        .iter()
        .copied()
        .find(|task| Kind::of(tasks(&plan)[*task].kind) == Kind::Attention)
        .expect("the attention reads the cursor it starts from");
    assert!(
        follows(&plan, written[0], attention),
        "the task that writes the cursor runs before the block that starts from it",
    );
    assert_eq!(tasks(&plan)[attention].origin, cursor.id());
}

#[test]
fn a_cursor_holds_one_position_per_plane() {
    let graph = Graph::new();
    let queries = graph.parameter(Shape::of([2, 1, 4, 4]), Init::Zero, Element::Single);
    let keys = graph.parameter(Shape::of([2, 1, 4, 4]), Init::Zero, Element::Single);
    let positions = |dims: [u32; 4]| graph.fill(Shape::of(dims), 1.0);
    let options = |origin| neura_graph::AttentionOptions {
        scale: 0.5,
        causal: true,
        origin,
        segments: None,
        reach: None,
    };
    let _ = graph.attention(queries, keys, keys, options(None));
    let _ = graph.attention(queries, keys, keys, options(Some(positions([1, 1, 1, 1]))));
    let _ = graph.attention(queries, keys, keys, options(Some(positions([2, 1, 1, 1]))));
    assert!(refuses(|| {
        let _ = graph.attention(queries, keys, keys, options(Some(positions([3, 1, 1, 1]))));
    }));
    assert!(refuses(|| {
        let _ = graph.attention(queries, keys, keys, options(Some(positions([1, 1, 4, 1]))));
    }));
    assert!(refuses(|| {
        let brief = graph.parameter(Shape::of([2, 1, 2, 4]), Init::Zero, Element::Single);
        let _ = graph.attention(
            queries,
            brief,
            brief,
            options(Some(positions([1, 1, 1, 1]))),
        );
    }));
}

#[test]
fn a_block_quantized_weight_finds_its_quantum_beside_the_words_it_packs() {
    let graph = Graph::new();
    let weight = graph.block_quantized_parameter(Shape::vector(300), Init::Zero, Element::Int4);
    let plan = plan(&graph);
    let values = records::<ValueRecord>(plan.values(), size_of::<ValueRecord>());
    let record = values[weight.id() as usize];
    assert_eq!(record.element, Element::Int4.code());
    assert_eq!(record.store, Store::Weights.code());
    assert_eq!(
        record.table,
        Element::Int4.payload_words(300) as u32,
        "the quantum of every block stands beside the words its numbers pack into",
    );
    assert_eq!(Element::Int4.quanta(300), 3);
    assert_eq!(plan.weights().words(), Element::Int4.storage_words(300));
}

#[test]
fn a_plan_never_writes_a_block_quantized_tensor() {
    let graph = Graph::new();
    let weight = graph.block_quantized_parameter(Shape::vector(4), Init::Zero, Element::Int4);
    let data = graph.input(Shape::vector(4), Element::Single);
    graph.add_into(weight, data);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| plan(&graph))).is_err(),
        "a block quantized weight packs the quantum of every block out of the numbers it holds, and a plan carries none of them",
    );
}

#[test]
fn a_wide_attention_head_trades_its_key_span_for_the_row_it_carries() {
    let graph = Graph::new();
    let queries = graph.parameter(Shape::of([2, 1, 8, 64]), Init::Zero, Element::Single);
    let keys = graph.parameter(Shape::of([2, 1, 8, 64]), Init::Zero, Element::Single);
    let out = graph.attention(
        queries,
        keys,
        keys,
        neura_graph::AttentionOptions {
            scale: 0.125,
            causal: true,
            origin: None,
            segments: None,
            reach: None,
        },
    );
    graph.retain(out);
    let plan = plan(&graph);
    let tile = plan.attention()[0];
    assert!(
        tile.registers() <= AttentionTile::REGISTER_CEILING,
        "a row of {} numbers and its gradient outrun the {} registers a device thread carries",
        tile.width(),
        AttentionTile::REGISTER_CEILING,
    );
    assert_eq!(
        tile.keys(),
        8,
        "a row of {} numbers leaves room for the widest key span both budgets carry",
        tile.width(),
    );
    assert_eq!(
        tile.registers(),
        AttentionTile::REGISTER_CEILING,
        "a wide row spends the whole register budget its device thread carries",
    );
}

#[test]
fn a_head_too_wide_for_one_thread_is_refused() {
    let graph = Graph::new();
    let queries = graph.parameter(Shape::of([2, 1, 8, 80]), Init::Zero, Element::Single);
    let keys = graph.parameter(Shape::of([2, 1, 8, 80]), Init::Zero, Element::Single);
    let _ = graph.attention(
        queries,
        keys,
        keys,
        neura_graph::AttentionOptions {
            scale: 0.125,
            causal: true,
            origin: None,
            segments: None,
            reach: None,
        },
    );
    assert!(
        refuses(|| {
            let _ = plan(&graph);
        }),
        "a query row of 80 numbers and its gradient outrun the registers of one device thread",
    );
}

#[test]
fn a_reader_of_one_span_stays_beside_every_write_of_that_span() {
    let graph = Graph::new();
    let state = graph.state(Shape::vector(100_000), Init::Zero, Element::Single);
    let factor = graph.fill(Shape::vector(100_000), 0.5);
    let addend = graph.fill(Shape::vector(100_000), 0.25);
    graph.mul_into(state, factor);
    let halfway = graph.relu(state);
    graph.add_into(state, addend);
    graph.retain(halfway);
    let plan = plan(&graph);
    let tasks = tasks(&plan);
    let mut checked = 0;
    for update in writers(&plan, state.id()) {
        for reader in readers(&plan, state.id()) {
            if update == reader {
                continue;
            }
            let written = &tasks[update];
            let read = &tasks[reader];
            let overlap = written.first < read.first + read.count
                && read.first < written.first + written.count;
            if !overlap {
                continue;
            }
            checked += 1;
            assert!(
                follows(&plan, reader, update) || follows(&plan, update, reader),
                "task {update} writes the span task {reader} reads, and neither rides a wave or a slot before the other",
            );
        }
    }
    assert!(
        checked > 0,
        "a span of one tensor is read beside the writes of that span",
    );
}

#[test]
fn a_product_shortlists_the_tile_of_every_strategy_its_profile_offers() {
    let staged = MatmulTile::new(MatmulStrategy::Staged, 64, 64, 8, 8, 8);
    let cooperative = MatmulTile::cooperative(CooperativeTile::new(
        CooperativeMatrix::new(32, 16, 16, 16),
        (1, 2),
        (1, 1),
        16,
    ));
    let profile = Profile::of(&[staged, cooperative]);
    let product = Product::of(1, 128, 128, 8);
    assert_eq!(product.planned(profile), staged);
    assert_eq!(product.gathered(profile), Some(cooperative));
    assert_eq!(product.shortlist(profile), vec![staged, cooperative]);
    let gathered_only = Profile::of(&[cooperative]);
    assert_eq!(gathered_only.tiles(), &[cooperative]);
    assert_eq!(product.planned(gathered_only), cooperative);
    assert_eq!(product.shortlist(gathered_only), vec![cooperative]);
}

#[test]
fn a_plan_walks_the_tile_a_measured_choice_names() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(64, 32), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(32, 64), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    graph.retain(out);
    let profile = wide();
    let product = Product::of(1, 64, 64, 32);
    let planned = product.planned(profile);
    let chosen = *profile
        .tiles()
        .iter()
        .find(|tile| **tile != planned)
        .expect("a profile offers a tile beside the one it plans");
    let chosen_plan = Plan::chosen(&graph, ALIGNMENT, profile, &[(product, chosen)]);
    assert_eq!(chosen_plan.products(), &[product]);
    let chosen_tasks = tasks(&chosen_plan);
    assert!(!chosen_tasks.is_empty(), "a measured plan holds no task");
    for task in &chosen_tasks {
        assert_eq!(
            chosen_plan.tiles()[task.geometry as usize],
            chosen,
            "a measured plan walks another tile than the one it was handed",
        );
    }
    let planned_plan = plan_with(&graph, profile);
    for task in tasks(&planned_plan) {
        assert_eq!(planned_plan.tiles()[task.geometry as usize], planned);
    }
}

#[test]
fn a_plan_refuses_a_measured_tile_its_profile_does_not_offer() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(64, 32), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(32, 64), Init::Zero, Element::Single);
    graph.retain(graph.matmul(left, right));
    let profile = narrow();
    let foreign = *wide()
        .tiles()
        .iter()
        .find(|tile| !profile.tiles().contains(tile))
        .expect("a wider profile offers a tile the narrow one does not");
    assert!(
        refuses(|| {
            let _ = Plan::chosen(
                &graph,
                ALIGNMENT,
                profile,
                &[(Product::of(1, 64, 64, 32), foreign)],
            );
        }),
        "a plan walked a measured tile its profile does not offer",
    );
}

#[test]
fn a_plan_walks_the_extent_a_device_count_authors() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 8, 1]), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, 8, 4]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let live = graph.trim(tokens, 2, count);
    let out = graph.matmul(
        live,
        graph.parameter(Shape::matrix(4, 4), Init::Zero, Element::Single),
    );
    graph.retain(out);
    let plan = plan(&graph);
    assert!(
        plan.carries_authored(),
        "the plan walks a device counted extent"
    );
    assert_eq!(
        plan.host_slots(),
        Vec::new(),
        "a device counted extent is bound by no host",
    );
    assert_eq!(plan.patches().len(), 1, "one task authors the extent");
    assert_eq!(plan.patches()[0].count, count.id());
    let author = tasks(&plan)
        .iter()
        .position(|task| task.patch == 0)
        .expect("the task that authors the extent");
    assert_eq!(
        tasks(&plan)[author].out,
        count.id(),
        "the task that patches the extent is the task that counts it",
    );
    let patch = plan.patches()[0];
    let slots = patched_slots(&plan, patch);
    assert!(
        plan.authored_values(live.id())
            .iter()
            .any(|slot| slots.contains(slot)),
        "the trimmed tensor walks the extent its count authors",
    );
    let list = plan.patch_list();
    let patched = &list[patch.values as usize..(patch.values + patch.values_count) as usize];
    assert!(
        patched.contains(&live.id()) && patched.contains(&out.id()),
        "the tensors whose rows the count rules are patched",
    );
    assert!(
        patch.tasks_count > 0,
        "the count rules the range of some task"
    );
    let patched_tasks = &list[patch.tasks as usize..(patch.tasks + patch.tasks_count) as usize];
    for at in patched_tasks {
        let task = tasks(&plan)[*at as usize];
        assert_ne!(
            task.split,
            neura_abi::split::RANGE,
            "a task the device count rules computes its own range on the device",
        );
        assert!(
            (task.measure as usize) < plan.measures().len(),
            "a task the device count rules walks a measure of the plan",
        );
    }
}

#[test]
fn a_trimmed_tensor_lands_its_gradient_in_the_walk_of_the_storage_that_owns_it() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 8, 1]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let gate = graph.gradient_input(Shape::of([1, 1, 1, 4]), Element::Single);
    let tokens = graph.gradient_input(Shape::of([1, 1, 8, 4]), Element::Single);
    let rows = graph.trim(tokens, 2, count);
    let loss = graph.sum(graph.mul(rows, gate));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(tokens);
    graph.retain(gradient);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let extend = records
        .iter()
        .position(|task| Kind::of(task.kind) == Kind::Extend)
        .expect("a trimmed tensor lands its gradient through an extend task");
    assert_eq!(
        records[extend].out,
        gradient.id(),
        "the extend task writes the gradient of the tensor that owns the storage",
    );
    assert_eq!(
        records[extend].b,
        rows.id(),
        "the extend task walks the extent the trimmed tensor names",
    );
    assert_eq!(
        records[extend].split,
        neura_abi::split::RANGE,
        "the extend task walks the bounds the storage of the tensor it lands in declares",
    );
    let patch = plan.patches()[0];
    let list = plan.patch_list();
    let values = &list[patch.values as usize..(patch.values + patch.values_count) as usize];
    assert!(
        values.contains(&rows.id()),
        "the patch refreshes the length of the tensor the trim walks",
    );
    assert!(
        values.contains(&records[extend].a),
        "the patch refreshes the length of the gradient the extend reads",
    );
    let author = records
        .iter()
        .position(|task| task.patch != neura_abi::NO_VALUE)
        .expect("the task that authors the count");
    assert!(
        follows(&plan, author, extend),
        "the extend task lands its gradient after the task that authors the count",
    );
}

#[test]
fn a_count_rules_every_extent_the_graph_hands_it() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 16, 1]), Element::Single);
    let count = graph.sum(probe);
    let longer = graph.input(Shape::of([1, 1, 16, 4]), Element::Single);
    let shorter = graph.input(Shape::of([1, 1, 12, 4]), Element::Single);
    let first = graph.trim(longer, 2, count);
    let second = graph.trim(shorter, 2, count);
    let out = graph.add(graph.sum(first), graph.sum(second));
    graph.retain(out);
    let plan = plan(&graph);
    let records = tasks(&plan);
    assert_eq!(
        plan.patches().len(),
        1,
        "one task counts both extents, and one patch rules every extent a count walks",
    );
    let patch = plan.patches()[0];
    assert_eq!(patch.count, count.id());
    assert_eq!(
        patched_slots(&plan, patch),
        [
            first
                .shape()
                .free(2)
                .expect("a trimmed tensor walks the count on axis 2"),
            second
                .shape()
                .free(2)
                .expect("a trimmed tensor walks the count on axis 2"),
        ],
        "the patch rules the extent of every tensor the count was handed",
    );
    assert_eq!(
        records
            .iter()
            .filter(|task| task.patch != neura_abi::NO_VALUE)
            .count(),
        1,
        "the task that counts the extents carries the one patch",
    );
    let list = plan.patch_list();
    let values = &list[patch.values as usize..(patch.values + patch.values_count) as usize];
    assert!(
        values.contains(&first.id()) && values.contains(&second.id()),
        "the tensors of every extent the count walks are patched",
    );
    let patched = &list[patch.tasks as usize..(patch.tasks + patch.tasks_count) as usize];
    let walks = |value: u32| {
        patched.iter().any(|at| {
            let task = records[*at as usize];
            [task.a, task.b, task.c, task.d, task.e, task.f]
                .into_iter()
                .chain([task.out, task.extra])
                .any(|touched| touched == value)
        })
    };
    assert!(
        walks(first.id()) && walks(second.id()),
        "the ranges of the tasks that walk either extent are patched",
    );
}

#[test]
fn a_count_a_scalar_step_reads_keeps_the_task_that_writes_it() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 8, 1]), Element::Single);
    let count = graph.relu(graph.sum(probe));
    let free = graph.counted(8, count);
    let tokens = graph.input(Shape::of([1, 1, 8, 4]).freed(&[(2, free)]), Element::Single);
    let factor = graph.input(Shape::scalar(), Element::Single);
    let scaled = graph.mul(count, factor);
    let out = graph.mul(graph.sum(tokens), scaled);
    graph.retain(out);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let writer = records
        .iter()
        .position(|task| task.out == count.id())
        .expect("the count has a writer");
    assert_eq!(
        records.iter().filter(|task| task.out == count.id()).count(),
        1,
        "a scalar step that reads the count folds no other task into the writer that counts it",
    );
    assert_eq!(Kind::of(records[writer].kind), Kind::Unary);
    assert_eq!(records[writer].op, op::RELU);
    assert_eq!(plan.patches().len(), 1, "one task authors the extent");
    assert_eq!(plan.patches()[0].count, count.id());
    assert_eq!(
        records[writer].patch, 0,
        "the patch lands on the task that counts the extent, and that task survives every fold",
    );
}

#[test]
fn every_task_that_walks_a_device_count_stands_after_the_task_that_authors_it() {
    let graph = Graph::new();
    let bound = 64u32;
    let depth = 1024u32;
    let columns = 64u32;
    let probe = graph.input(Shape::of([1, 1, bound, 1]), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, bound, depth]), Element::Single);
    let weight = graph.parameter(Shape::matrix(depth, columns), Init::Zero, Element::Single);
    let count = graph.sum_axis(probe, 2);
    let live = graph.trim(tokens, 2, count);
    let product = graph.matmul(live, weight);
    let rotated = graph.rope(product, None, 10_000.0);
    let attended = graph.attention(
        rotated,
        rotated,
        rotated,
        AttentionOptions {
            scale: 0.125,
            causal: true,
            origin: None,
            segments: None,
            reach: None,
        },
    );
    let probabilities = graph.softmax(attended);
    let narrow = graph.cast(probabilities, Element::Half);
    graph.retain(product);
    graph.retain(narrow);
    let plan = plan(&graph);
    assert_eq!(plan.patches().len(), 1, "one task authors the extent");
    assert!(
        kinds(&plan).contains(&Kind::MatmulFold),
        "the product splits its depth across tasks, and the fold reads the partials only the split laid out",
    );
    let records = tasks(&plan);
    let author = records
        .iter()
        .position(|task| task.patch != neura_abi::NO_VALUE)
        .expect("the task that authors the extent");
    let patch = plan.patches()[0];
    let list = plan.patch_list();
    let values = &list[patch.values as usize..(patch.values + patch.values_count) as usize];
    let steps = steps(&plan);
    for (index, task) in records.iter().enumerate() {
        let reads = [
            task.a,
            task.b,
            task.c,
            task.d,
            task.e,
            task.f,
            task.origin,
            task.segment,
        ]
        .into_iter()
        .chain(
            (task.prelude..task.prelude + task.prelude_steps)
                .map(|step| steps[step as usize].operand),
        )
        .chain((task.chain..task.chain + task.steps).map(|step| steps[step as usize].operand))
        .filter(|value| *value != neura_abi::NO_VALUE)
        .collect::<Vec<u32>>();
        if reads.iter().any(|value| values.contains(value)) {
            assert!(
                follows(&plan, author, index),
                "task {index} walks the extent the task {author} authors, and the device only patches that extent after the authoring workgroup runs it",
            );
        }
    }
    let patched = &list[patch.tasks as usize..(patch.tasks + patch.tasks_count) as usize];
    assert!(
        !patched.is_empty(),
        "the count rules the range of some task",
    );
    for at in patched {
        assert!(
            follows(&plan, author, *at as usize),
            "task {at} walks a range the task {author} authors, and the device only patches that range after the authoring workgroup runs it",
        );
    }
}

#[test]
fn a_count_that_walks_the_extent_it_authors_is_refused() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 8, 1]), Element::Single);
    let counted = graph.free(8);
    let masked = graph.input(
        Shape::of([1, 1, 8, 1]).freed(&[(2, counted)]),
        Element::Single,
    );
    let count = graph.sum(masked);
    graph.author(counted, count);
    let tokens = graph.input(Shape::of([1, 1, 8, 4]), Element::Single);
    graph.retain(graph.mul(tokens, probe));
    assert!(
        refuses(|| {
            plan(&graph);
        }),
        "a count that walks the extent it authors was planned",
    );
}

#[test]
fn a_device_count_of_every_plane_a_batch_walks_is_refused() {
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, 8, 1]), Element::Single);
    let counted = graph.free(8);
    let planes = graph.input(
        Shape::of([2, 1, 8, 4]).freed(&[(2, counted)]),
        Element::Single,
    );
    let count = graph.sum(probe);
    graph.author(counted, count);
    graph.retain(graph.mul(planes, planes));
    assert!(
        refuses(|| {
            plan(&graph);
        }),
        "a device count whose extent cuts every plane a batch walks was planned",
    );
}

#[test]
fn a_compaction_walks_the_rows_a_mask_names() {
    let graph = Graph::new();
    let rows = graph.free(8);
    let mask = graph.input(Shape::matrix(8, 1).freed(&[(2, rows)]), Element::Single);
    let compacted = graph.compact(mask);
    let table = graph.input(Shape::matrix(8, 2), Element::Single);
    let selected = graph.gather(table, compacted.indices);
    let total = graph.sum(selected);
    graph.retain(compacted.indices);
    graph.retain(selected);
    graph.retain(total);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let names = kinds(&plan);
    assert!(
        names.contains(&Kind::PrefixChunk)
            && names.contains(&Kind::PrefixScan)
            && names.contains(&Kind::PrefixClose)
            && names.contains(&Kind::Compact),
        "a compaction walks a two-level prefix its count closes: {names:?}",
    );
    assert!(plan.carries_authored());
    assert_eq!(plan.host_slots(), [(rows.slot(), 8)]);
    let count = graph
        .shape(compacted.indices)
        .free(2)
        .expect("an index list walks the rows a device count holds");
    assert_eq!(
        plan.authored_slots()[rows.slot() as usize],
        neura_abi::NO_VALUE,
        "the rows of the mask are bound by the host",
    );
    assert_eq!(plan.authored_slots()[count as usize], count);
    let patches = plan.patches();
    assert_eq!(patches.len(), 1, "one slot is authored");
    assert_eq!(patched_slots(&plan, patches[0]), [count]);
    assert_eq!(
        patches[0].segment,
        neura_abi::NO_VALUE,
        "a prefix sum closes no offsets of a ragged axis",
    );
    let author = records
        .iter()
        .position(|task| task.patch != neura_abi::NO_VALUE)
        .expect("the closing task authors the count");
    assert_eq!(Kind::of(records[author].kind), Kind::PrefixClose);
    let compact = records
        .iter()
        .position(|task| Kind::of(task.kind) == Kind::Compact)
        .expect("a compaction walks the rows its mask names");
    assert_eq!(records[compact].a, mask.id());
    assert_ne!(
        records[compact].b,
        neura_abi::NO_VALUE,
        "a compaction walks the offsets its mask sums",
    );
    assert!(
        follows(&plan, author, compact),
        "a compaction walks the count the closing task authors",
    );
    for reader in records
        .iter()
        .enumerate()
        .filter(|(_, task)| task.out != compacted.indices.id() && task.b == compacted.indices.id())
        .map(|(index, _)| index)
    {
        assert!(
            follows(&plan, compact, reader),
            "task {reader} walks the index list the compaction fills",
        );
    }
}

#[test]
fn a_ragged_axis_walks_the_offsets_a_device_prefix_closes() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let ragged = graph.ragged(16, lengths);
    let cache = graph.resident(
        Shape::of([1, 1, 16, 4]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(Shape::of([1, 4, 1, 4]), Element::Single);
    let cursor = graph.input(Shape::of([1, 4, 1, 1]), Element::Single);
    let out = graph.attention(
        query,
        cache,
        cache,
        AttentionOptions {
            scale: 0.5,
            causal: true,
            origin: Some(cursor),
            segments: Some(ragged.offsets),
            reach: None,
        },
    );
    graph.retain(out);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let names = kinds(&plan);
    assert!(
        names.contains(&Kind::PrefixChunk)
            && names.contains(&Kind::PrefixScan)
            && names.contains(&Kind::PrefixClose),
        "a ragged axis walks a two-level prefix closed by the extent it authors: {names:?}",
    );
    assert!(plan.carries_authored());
    assert_eq!(plan.authored_slots().len(), 1);
    assert_eq!(plan.host_slots(), Vec::new());
    let patches = plan.patches();
    assert_eq!(patches.len(), 1, "one slot is authored");
    let patch = patches[0];
    assert_eq!(patched_slots(&plan, patch), [ragged.extent.slot()]);
    assert_eq!(
        patch.segment,
        ragged.offsets.id(),
        "the patch that authors a ragged extent closes the offsets of that axis",
    );
    let author = records
        .iter()
        .position(|task| task.patch != neura_abi::NO_VALUE)
        .expect("the closing task authors the extent");
    assert_eq!(Kind::of(records[author].kind), Kind::PrefixClose);
    let list = plan.patch_list();
    let values = &list[patch.values as usize..(patch.values + patch.values_count) as usize];
    assert_eq!(
        values,
        [cache.id()],
        "the packed tensor takes the live extent"
    );
    let patched = &list[patch.tasks as usize..(patch.tasks + patch.tasks_count) as usize];
    assert_eq!(patched.len(), 4, "every plane of the cache is patched");
    let mut planes = Vec::new();
    for at in patched {
        let task = records[*at as usize];
        assert_eq!(Kind::of(task.kind), Kind::Attention);
        assert_eq!(task.segment, ragged.offsets.id());
        assert!(
            follows(&plan, author, *at as usize),
            "task {at} walks the offsets the closing task authors",
        );
        planes.push(task.plane);
    }
    planes.sort_unstable();
    assert_eq!(planes, [0, 1, 2, 3]);
}

#[test]
fn a_row_map_walks_the_planes_a_ragged_axis_closes() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let ragged = graph.ragged(16, lengths);
    let rows = graph.rows(ragged);
    graph.retain(rows.plane);
    graph.retain(rows.position);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let at = records
        .iter()
        .enumerate()
        .filter(|(_, task)| Kind::of(task.kind) == Kind::Rows)
        .map(|(index, _)| index)
        .collect::<Vec<usize>>();
    assert_eq!(
        at.len(),
        4,
        "a row map walks the planes a ragged axis closes",
    );
    let patches = plan.patches();
    assert_eq!(patches.len(), 1, "one slot is authored");
    let patch = patches[0];
    assert_eq!(patched_slots(&plan, patch), [ragged.extent.slot()]);
    assert_eq!(patch.segment, ragged.offsets.id());
    let author = records
        .iter()
        .position(|task| task.patch != neura_abi::NO_VALUE)
        .expect("the closing task authors the extent");
    assert_eq!(Kind::of(records[author].kind), Kind::PrefixClose);
    let list = plan.patch_list();
    let values = &list[patch.values as usize..(patch.values + patch.values_count) as usize];
    assert!(
        values.contains(&rows.plane.id()) && values.contains(&rows.position.id()),
        "the row map takes the live rows of the axis: {values:?}",
    );
    let patched = &list[patch.tasks as usize..(patch.tasks + patch.tasks_count) as usize];
    let mut planes = Vec::new();
    for task in at {
        assert_eq!(records[task].split, neura_abi::split::RAGGED);
        assert_eq!(
            (records[task].first, records[task].count),
            (0, 0),
            "the device hands a row map its rows",
        );
        assert_eq!(records[task].a, ragged.offsets.id());
        assert_eq!(records[task].segment, ragged.offsets.id());
        assert!(patched.contains(&(task as u32)));
        assert!(
            follows(&plan, author, task),
            "a row map walks the offsets the closing task authors",
        );
        planes.push(records[task].plane);
    }
    planes.sort_unstable();
    assert_eq!(planes, [0, 1, 2, 3]);
}

#[test]
fn a_ragged_axis_a_device_count_narrows_closes_its_offsets_once() {
    let graph = Graph::new();
    let flags = graph.input(Shape::of([1, 1, 1, 4]), Element::Single);
    let count = graph.sum(flags);
    let live = graph.counted(4, count);
    let lengths = graph.input(Shape::of([1, 1, 1, 4]).freed(&[(3, live)]), Element::Single);
    let ragged = graph.ragged(16, lengths);
    let cache = graph.resident(
        Shape::of([1, 1, 16, 4]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let query = graph.input(Shape::of([1, 4, 1, 4]).freed(&[(1, live)]), Element::Single);
    let cursor = graph.input(Shape::of([1, 4, 1, 1]), Element::Single);
    let out = graph.attention(
        query,
        cache,
        cache,
        AttentionOptions {
            scale: 0.5,
            causal: true,
            origin: Some(cursor),
            segments: Some(ragged.offsets),
            reach: None,
        },
    );
    graph.retain(out);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let patches = plan.patches();
    assert_eq!(
        patches.len(),
        2,
        "a device count and the closing total author two extents",
    );
    let closing = patches
        .iter()
        .find(|patch| patch.segment != neura_abi::NO_VALUE)
        .expect("the closing patch closes the offsets of its ragged axis");
    assert_eq!(patched_slots(&plan, *closing), [ragged.extent.slot()]);
    assert_eq!(closing.segment, ragged.offsets.id());
    let planes = patches
        .iter()
        .find(|patch| patched_slots(&plan, **patch) == [live.slot()])
        .expect("the device count authors the plane extent");
    assert_eq!(
        planes.segment,
        neura_abi::NO_VALUE,
        "a patch that closes no ragged axis walks no offsets",
    );
    let list = plan.patch_list();
    let closed = &list[closing.tasks as usize..(closing.tasks + closing.tasks_count) as usize];
    assert_eq!(closed.len(), 4, "every plane of the cache closes its keys");
    for at in closed {
        let task = records[*at as usize];
        assert_eq!(Kind::of(task.kind), Kind::Attention);
        assert_eq!(task.segment, ragged.offsets.id());
    }
    let walked = &list[planes.tasks as usize..(planes.tasks + planes.tasks_count) as usize];
    let attention = walked
        .iter()
        .filter(|at| Kind::of(records[**at as usize].kind) == Kind::Attention)
        .collect::<Vec<_>>();
    assert_eq!(
        attention.len(),
        4,
        "every plane of the cache walks its own tokens",
    );
    for at in attention {
        assert_eq!(records[*at as usize].segment, ragged.offsets.id());
    }
}

#[test]
fn a_ragged_axis_a_binding_narrows_walks_a_measure() {
    let fixed = Graph::new();
    let fixed_lengths = fixed.input(Shape::vector(4), Element::Single);
    let fixed_ragged = fixed.ragged(16, fixed_lengths);
    fixed.retain(fixed_ragged.offsets);
    for task in tasks(&plan(&fixed)) {
        if matches!(Kind::of(task.kind), Kind::PrefixChunk | Kind::PrefixScan) {
            assert_eq!(
                task.split,
                neura_abi::split::RANGE,
                "a prefix of one shape walks the range its plan carries",
            );
        }
    }

    let graph = Graph::new();
    let live = graph.free(4);
    let lengths = graph.input(Shape::of([1, 4]).freed(&[(3, live)]), Element::Single);
    let ragged = graph.ragged(16, lengths);
    graph.retain(ragged.offsets);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let scanning = records
        .iter()
        .filter(|task| matches!(Kind::of(task.kind), Kind::PrefixChunk | Kind::PrefixScan))
        .collect::<Vec<_>>();
    assert_eq!(scanning.len(), 2, "one chunk and the scan that walks it");
    for task in scanning {
        assert_eq!(
            task.split,
            neura_abi::split::UNIFORM,
            "a prefix of a length a binding narrows asks the device for the range it walks",
        );
        let measure = &plan.measures()[task.measure as usize];
        assert_eq!(measure.kind, neura_abi::measure::ELEMENTS);
        assert_eq!(measure.value, lengths.id());
    }
}

#[test]
fn a_row_that_reads_a_wider_tensor_waits_for_the_task_that_writes_it() {
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, 8, 4]), Element::Single);
    let row = graph.input(Shape::of([1, 1, 1, 64]), Element::Single);
    let filter = graph.parameter(Shape::of([1, 1, 64, 4]), Init::Zero, Element::Single);
    let bias = graph.matmul(row, filter);
    let activated = graph.add(graph.softmax(data), bias);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let writing = writers(&plan, bias.id());
    assert!(
        !writing.is_empty(),
        "the product writes the row the softmax adds to every row it weighs",
    );
    for (index, task) in records.iter().enumerate() {
        if Kind::of(task.kind) != Kind::Softmax {
            continue;
        }
        for writer in &writing {
            assert!(
                follows(&plan, *writer, index),
                "a step that reads the four numbers of a row walks the whole row of {activated:?} rather than the range of the task, so the task waits for the wave that writes it",
            );
        }
    }
}

#[test]
fn a_convert_of_a_walk_a_binding_narrows_waits_for_every_task_that_writes_it() {
    let graph = Graph::new();
    let tokens = graph.free(8192);
    let shape = Shape::of([1, 1, 1, 8192]).freed(&[(3, tokens)]);
    let left = graph.input(shape, Element::Single);
    let right = graph.input(shape, Element::Single);
    let product = graph.mul(left, right);
    let half = graph.cast(product, Element::Half);
    graph.retain(product);
    graph.retain(half);
    let plan = plan(&graph);
    let writing = writers(&plan, product.id());
    let converting = kinds(&plan)
        .iter()
        .enumerate()
        .filter(|(_, kind)| **kind == Kind::Convert)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    assert!(
        writing.len() > 1 && converting.len() > 1,
        "a walk of 8192 numbers a binding rules is written in {} pieces and narrowed in {} of them",
        writing.len(),
        converting.len(),
    );
    for task in &converting {
        for writer in &writing {
            assert!(
                follows(&plan, *writer, *task),
                "the convert of a walk a binding narrows reads the numbers task {writer} writes, and the plan stands it before them",
            );
        }
    }
}

#[test]
fn a_grouped_product_walks_the_tiles_every_segment_holds() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(3), Element::Single);
    let ragged = graph.ragged(24, lengths);
    let left = graph.input(
        Shape::of([1, 1, 24, 4]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([3, 1, 4, 2]),
        Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    graph.freeze(&[weights]);
    let out = graph.grouped_matmul(left, weights, ragged.offsets);
    graph.retain(out);
    let plan = plan(&graph);
    let records = tasks(&plan);
    let names = kinds(&plan);
    assert!(
        names.contains(&Kind::PrefixChunk)
            && names.contains(&Kind::PrefixScan)
            && names.contains(&Kind::PrefixClose),
        "a grouped product walks the offsets a two-level prefix closes: {names:?}",
    );
    let patches = plan.patches();
    assert_eq!(patches.len(), 1, "one slot is authored");
    let patch = patches[0];
    assert_eq!(patched_slots(&plan, patch), [ragged.extent.slot()]);
    assert_eq!(
        patch.segment,
        ragged.offsets.id(),
        "the patch that authors a ragged extent closes the offsets of that axis",
    );
    let author = records
        .iter()
        .position(|task| task.patch != neura_abi::NO_VALUE)
        .expect("the closing task authors the extent");
    assert_eq!(Kind::of(records[author].kind), Kind::PrefixClose);
    let list = plan.patch_list();
    let values = &list[patch.values as usize..(patch.values + patch.values_count) as usize];
    assert_eq!(
        values,
        [left.id(), out.id()],
        "the packed rows and the rows a grouped product weighs take the live extent",
    );
    let measure = plan
        .measures()
        .iter()
        .find(|measure| measure.kind == neura_abi::measure::TILES)
        .expect("a grouped product walks the tiles of the rows it weighs");
    assert_eq!(measure.value, out.id());
    assert!(measure.rows > 0 && measure.columns > 0);
    let patched = &list[patch.tasks as usize..(patch.tasks + patch.tasks_count) as usize];
    let mut segments = Vec::new();
    let mut tiles = Vec::new();
    for at in patched {
        let task = records[*at as usize];
        assert_eq!(Kind::of(task.kind), Kind::Matmul);
        assert_eq!(task.split, neura_abi::split::SEGMENT);
        assert_eq!(task.segment, ragged.offsets.id());
        assert_eq!(
            task.measure,
            plan.measures()
                .iter()
                .position(|kept| std::ptr::eq(kept, measure))
                .expect("the measure a grouped product walks")
                .try_into()
                .expect("a measure fits a device word"),
        );
        assert!(
            follows(&plan, author, *at as usize),
            "task {at} walks the offsets the closing task authors",
        );
        segments.push(task.plane);
        tiles.push((task.group, task.index));
    }
    segments.sort_unstable();
    segments.dedup();
    assert_eq!(
        segments,
        [0, 1, 2],
        "every segment a ragged axis closes holds the tiles of its own rows",
    );
    assert_eq!(
        tiles.len(),
        segments.len() * 3,
        "each segment holds the tiles of the bound its packed rows walk",
    );
}

#[test]
fn a_packed_attention_hands_its_gradients_the_segments_its_offsets_close() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let bound = 256;
    let ragged = graph.ragged(bound, lengths);
    let packed = Shape::of([1, 1, bound, 4]).freed(&[(2, ragged.extent)]);
    let query = graph.gradient_input(Shape::of([1, 4, 1, 4]), Element::Single);
    let key = graph.gradient_input(packed, Element::Single);
    let value = graph.gradient_input(packed, Element::Single);
    let cursor = graph.input(Shape::of([1, 4, 1, 1]), Element::Single);
    let out = graph.attention(
        query,
        key,
        value,
        AttentionOptions {
            scale: 0.5,
            causal: true,
            origin: Some(cursor),
            segments: Some(ragged.offsets),
            reach: None,
        },
    );
    let loss = graph.sum(out);
    let collected = graph.backward(loss);
    graph.retain(collected.of(query));
    graph.retain(collected.of(key));
    graph.retain(collected.of(value));
    let plan = plan(&graph);
    let records = tasks(&plan);
    let patch = plan.patches()[0];
    assert_eq!(patch.segment, ragged.offsets.id());
    let author = records
        .iter()
        .position(|task| task.patch != neura_abi::NO_VALUE)
        .expect("the closing task authors the extent");
    let patched = plan.patch_list()
        [patch.tasks as usize..(patch.tasks + patch.tasks_count) as usize]
        .to_vec();
    for kind in [
        Kind::Attention,
        Kind::AttentionQueryGrad,
        Kind::AttentionKeyGrad,
        Kind::AttentionValueGrad,
    ] {
        let walked = records
            .iter()
            .enumerate()
            .filter(|(_, task)| Kind::of(task.kind) == kind)
            .collect::<Vec<(usize, &TaskRecord)>>();
        assert!(!walked.is_empty(), "the graph walks a {kind:?} task");
        for (index, task) in walked {
            assert_eq!(
                task.segment,
                ragged.offsets.id(),
                "every {kind:?} task takes its key span from the offsets its ragged axis closes",
            );
            assert!(
                patched.contains(&(index as u32)),
                "the patch that closes the offsets hands task {index} of {kind:?} its span",
            );
            assert!(
                follows(&plan, author, index),
                "task {index} of {kind:?} walks the offsets the closing task authors",
            );
        }
    }
    let mut planes = Vec::new();
    let mut chunks = Vec::new();
    let group = bound.div_ceil(narrow().workgroup());
    assert!(
        group > 1,
        "a bound of {bound} rows spans one chunk per workgroup"
    );
    for (index, task) in records.iter().enumerate() {
        let kind = Kind::of(task.kind);
        if kind == Kind::AttentionKeyGrad || kind == Kind::AttentionValueGrad {
            assert_eq!(
                task.split,
                neura_abi::split::RAGGED,
                "a gradient walks the keys of one segment, and task {index} separates its rows by the count the ragged axis closes",
            );
            assert_eq!(
                (task.first, task.count),
                (0, 0),
                "task {index} takes the rows its segment holds at run time",
            );
            assert_eq!(
                task.group, group,
                "task {index} walks one of the {group} chunks a plane of {bound} rows holds",
            );
            assert!(
                task.index < group,
                "task {index} walks chunk {} of {group} chunks",
                task.index,
            );
            planes.push(task.plane);
            chunks.push((task.plane, task.index));
        }
    }
    planes.sort_unstable();
    planes.dedup();
    assert_eq!(planes, [0, 1, 2, 3]);
    chunks.sort_unstable();
    chunks.dedup();
    assert_eq!(
        chunks.len(),
        4 * group as usize,
        "every plane walks every chunk of the rows its bound holds, and one task covers one chunk",
    );
}

#[test]
fn a_row_map_parts_a_plane_beside_the_chunks_the_gradients_of_its_attention_walk() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let bound = 256;
    let ragged = graph.ragged(bound, lengths);
    let packed = Shape::of([1, 1, bound, 4]).freed(&[(2, ragged.extent)]);
    let query = graph.gradient_input(Shape::of([1, 4, 1, 4]), Element::Single);
    let key = graph.gradient_input(packed, Element::Single);
    let value = graph.gradient_input(packed, Element::Single);
    let cursor = graph.input(Shape::of([1, 4, 1, 1]), Element::Single);
    let out = graph.attention(
        query,
        key,
        value,
        AttentionOptions {
            scale: 0.5,
            causal: true,
            origin: Some(cursor),
            segments: Some(ragged.offsets),
            reach: None,
        },
    );
    let rows = graph.rows(ragged);
    let loss = graph.sum(out);
    let collected = graph.backward(loss);
    for value in [
        collected.of(query),
        collected.of(key),
        collected.of(value),
        rows.plane,
        rows.position,
    ] {
        graph.retain(value);
    }
    let plan = plan(&graph);
    let records = tasks(&plan);
    let group = bound.div_ceil(narrow().workgroup());
    assert!(
        group > 1,
        "a bound of {bound} rows outruns the {} rows one workgroup walks",
        narrow().workgroup(),
    );
    let mut grids = std::collections::BTreeMap::<(u32, u32), Vec<u32>>::new();
    let mut row_map = Vec::new();
    for task in records
        .iter()
        .filter(|task| task.split == neura_abi::split::RAGGED)
    {
        assert_eq!(
            task.segment,
            ragged.offsets.id(),
            "a ragged task walks the offsets the axis it belongs to closes",
        );
        if Kind::of(task.kind) == Kind::Rows {
            row_map.push(task.plane);
            assert_eq!(
                (task.group, task.index),
                (1, 0),
                "a row map walks the rows of one plane in one task",
            );
            continue;
        }
        assert_eq!(
            task.group, group,
            "a gradient cuts a plane into the {group} chunks one workgroup walks",
        );
        grids
            .entry((task.out, task.plane))
            .or_default()
            .push(task.index);
    }
    row_map.sort_unstable();
    assert_eq!(
        row_map,
        [0, 1, 2, 3],
        "a row map walks every plane the ragged axis closes",
    );
    assert_eq!(
        grids.len(),
        8,
        "the two gradient families of a packed attention part every plane",
    );
    for ((out, plane), mut chunks) in grids {
        chunks.sort_unstable();
        assert_eq!(
            chunks,
            (0..group).collect::<Vec<u32>>(),
            "plane {plane} of value {out} walks the chunks {chunks:?}",
        );
    }
}

#[test]
fn two_ragged_axes_of_the_plane_count_they_close_part_their_planes_apart() {
    let graph = Graph::new();
    let mut axes = Vec::new();
    let mut loss = None::<Value>;
    for bound in [256u32, 64] {
        let lengths = graph.input(Shape::vector(4), Element::Single);
        let ragged = graph.ragged(bound, lengths);
        let packed = Shape::of([1, 1, bound, 4]).freed(&[(2, ragged.extent)]);
        let query = graph.gradient_input(Shape::of([1, 4, 1, 4]), Element::Single);
        let key = graph.gradient_input(packed, Element::Single);
        let cursor = graph.input(Shape::of([1, 4, 1, 1]), Element::Single);
        let out = graph.attention(
            query,
            key,
            key,
            AttentionOptions {
                scale: 0.5,
                causal: true,
                origin: Some(cursor),
                segments: Some(ragged.offsets),
                reach: None,
            },
        );
        let summed = graph.sum(out);
        loss = Some(match loss {
            None => summed,
            Some(walked) => graph.add(walked, summed),
        });
        axes.push((bound, ragged.offsets, key));
    }
    let collected = graph.backward(loss.expect("two attentions weigh a loss"));
    let axes = axes
        .into_iter()
        .map(|(bound, offsets, key)| {
            let gradient = collected.of(key);
            graph.retain(gradient);
            (bound, offsets, gradient)
        })
        .collect::<Vec<_>>();
    let plan = plan(&graph);
    let mut grids = Vec::<(u32, u32, u32, u32)>::new();
    for task in tasks(&plan)
        .iter()
        .filter(|task| task.split == neura_abi::split::RAGGED)
    {
        grids.push((task.segment, task.out, task.plane, task.index));
    }
    for (bound, offsets, key) in axes {
        let group = bound.div_ceil(narrow().workgroup());
        for plane in 0..4 {
            let mut chunks = grids
                .iter()
                .filter(|(segment, out, walked, _)| {
                    *segment == offsets.id() && *out == key.id() && *walked == plane
                })
                .map(|(_, _, _, index)| *index)
                .collect::<Vec<u32>>();
            chunks.sort_unstable();
            assert_eq!(
                chunks,
                (0..group).collect::<Vec<u32>>(),
                "plane {plane} of the ragged axis of {bound} rows walks the chunks {chunks:?}",
            );
        }
    }
}

#[test]
fn a_segmented_product_weighs_the_rows_of_every_segment_into_its_weight_gradient() {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(3), Element::Single);
    let bound = 24;
    let ragged = graph.ragged(bound, lengths);
    let packed = Shape::of([1, 1, bound, 4]).freed(&[(2, ragged.extent)]);
    let left = graph.gradient_input(packed, Element::Single);
    let weights = graph.gradient_input(Shape::of([3, 1, 4, 2]), Element::Single);
    let out = graph.grouped_matmul(left, weights, ragged.offsets);
    let weight = graph.gradient_input(
        Shape::of([1, 1, bound, 2]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let loss = graph.sum(graph.mul(out, weight));
    let collected = graph.backward(loss);
    graph.retain(collected.of(left));
    graph.retain(collected.of(weights));
    let plan = plan(&graph);
    let records = tasks(&plan);
    let patch = plan.patches()[0];
    assert_eq!(patch.segment, ragged.offsets.id());
    let author = records
        .iter()
        .position(|task| task.patch != neura_abi::NO_VALUE)
        .expect("the closing task authors the extent");
    let patched = plan.patch_list()
        [patch.tasks as usize..(patch.tasks + patch.tasks_count) as usize]
        .to_vec();
    let mut walked = Vec::new();
    for (index, task) in records.iter().enumerate() {
        if Kind::of(task.kind) != Kind::MatmulWeightGrad {
            continue;
        }
        assert_eq!(
            task.split,
            neura_abi::split::RANGE,
            "a weight gradient walks the tiles of the weights it weighs, and no offsets cut them",
        );
        assert_eq!(task.segment, ragged.offsets.id());
        assert_eq!(
            task.keys, 0,
            "the patch that closes the axis writes the rows of the segment the task weighs",
        );
        assert!(
            patched.contains(&(index as u32)),
            "the patch that closes the axis hands the weight gradient its rows",
        );
        assert!(
            follows(&plan, author, index),
            "task {index} weighs the rows the closing task authors",
        );
        walked.push((task.plane, task.index));
    }
    assert!(!walked.is_empty(), "the graph weighs a weight gradient");
    let tiles = walked.len() / 3;
    assert_eq!(
        walked.len(),
        3 * tiles,
        "every segment holds the same tiles of the weights it weighs",
    );
    for plane in 0..3 {
        let mut indices = walked
            .iter()
            .filter(|(walked, _)| *walked == plane)
            .map(|(_, index)| *index)
            .collect::<Vec<u32>>();
        indices.sort_unstable();
        assert_eq!(
            indices,
            (0..tiles as u32).collect::<Vec<u32>>(),
            "segment {plane} weighs the tiles 0..{tiles} of its own weights",
        );
    }
}
