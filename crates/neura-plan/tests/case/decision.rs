use neura_abi::{Element, Kind, NO_VALUE, Placement, TaskRecord, ValueRecord, strategy};
use neura_graph::{Graph, Init, Shape, Value, Window};
use neura_profile::{AttentionTile, Budget, Profile};

fn narrow() -> Profile {
    Profile::derive(Budget::BASELINE, None)[0]
}

fn wide() -> Profile {
    *Profile::derive(Budget::BASELINE, None)
        .last()
        .expect("a profile")
}
use neura_plan::{DEFAULT_ENCODING_BYTES, Plan};
use std::mem::size_of;

const ALIGNMENT: u64 = 256;
const PLACEMENT: Placement = Placement::new(1 << 16, 1 << 18);

fn plan_with(graph: &Graph, profile: Profile) -> Plan {
    Plan::of(graph, ALIGNMENT, profile, DEFAULT_ENCODING_BYTES)
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn tasks(plan: &Plan) -> Vec<TaskRecord> {
    plan.tasks()
        .as_chunks::<{ std::mem::size_of::<TaskRecord>() }>()
        .0
        .iter()
        .map(|task| bytemuck::pod_read_unaligned(task))
        .collect()
}

fn tasks_of(plan: &Plan, value: Value) -> Vec<TaskRecord> {
    tasks(plan)
        .into_iter()
        .filter(|task| task.out == value.id())
        .collect()
}

fn values_of(plan: &Plan) -> Vec<ValueRecord> {
    plan.values()
        .as_chunks::<{ size_of::<ValueRecord>() }>()
        .0
        .iter()
        .map(|value| bytemuck::pod_read_unaligned(value))
        .collect()
}

#[test]
fn a_row_walks_the_strategy_its_rounds_ask_for() {
    let rows = 1024u32;
    let graph = Graph::new();
    let short = graph.argmax(graph.input(Shape::matrix(4, 8), Element::Single));
    let middle = graph.argmax(graph.input(Shape::matrix(rows, 64), Element::Single));
    let long = graph.argmax(graph.input(Shape::matrix(2, 4096), Element::Single));
    graph.retain(short);
    graph.retain(middle);
    graph.retain(long);
    for (profile, middle_geometry) in [
        (narrow(), strategy::WORKGROUP_ROW),
        (wide(), strategy::THREAD_ROW),
    ] {
        let plan = plan_with(&graph, profile);
        assert_eq!(tasks_of(&plan, short).len(), 4);
        let folds = tasks_of(&plan, middle);
        let rows_per_task = rows.div_ceil(profile.workgroups());
        assert_eq!(
            folds.len(),
            rows as usize / rows_per_task as usize,
            "a row fold hands the device the tasks its device width asks for",
        );
        assert_eq!(tasks_of(&plan, long).len(), 2);
        for (index, task) in tasks_of(&plan, short).into_iter().enumerate() {
            assert_eq!(task.geometry, strategy::THREAD_ROW);
            assert_eq!(task.first, index as u32);
            assert_eq!(task.count, 1);
            assert_eq!(Kind::of(task.kind), Kind::Argmax);
        }
        let mut covered = 0;
        for task in tasks_of(&plan, middle) {
            assert_eq!(
                task.geometry, middle_geometry,
                "a block of {rows} rows and 64 columns folds through {middle_geometry} on {profile:?}",
            );
            assert!(task.count > 0 && task.count <= rows);
            covered += task.count;
        }
        assert_eq!(covered, rows);
        for (index, task) in tasks_of(&plan, long).into_iter().enumerate() {
            assert_eq!(task.geometry, strategy::WORKGROUP_ROW);
            assert_eq!(task.first, index as u32);
            assert_eq!(task.count, 1);
        }
    }
}

#[test]
fn a_softmax_walks_the_body_its_rounds_ask_for() {
    let graph = Graph::new();
    let narrow_rows = graph.softmax(graph.input(Shape::matrix(4096, 64), Element::Single));
    let wide_rows = graph.softmax(graph.input(Shape::matrix(4096, 1024), Element::Single));
    let starved_rows = graph.softmax(graph.input(Shape::matrix(128, 64), Element::Single));
    graph.retain(narrow_rows);
    graph.retain(wide_rows);
    graph.retain(starved_rows);
    for (profile, narrow_geometry) in [
        (narrow(), strategy::WORKGROUP_ROW),
        (wide(), strategy::THREAD_ROW),
    ] {
        let plan = plan_with(&graph, profile);
        for (value, rows, geometry, label) in [
            (
                narrow_rows,
                4096u32,
                narrow_geometry,
                "4096 rows of 64 columns",
            ),
            (
                wide_rows,
                4096,
                strategy::WORKGROUP_ROW,
                "4096 rows of 1024 columns",
            ),
            (
                starved_rows,
                128,
                strategy::WORKGROUP_ROW,
                "128 rows of 64 columns",
            ),
        ] {
            let tasks = tasks_of(&plan, value);
            let mut covered = 0;
            for task in &tasks {
                assert_eq!(
                    task.geometry, geometry,
                    "{label} fold through {geometry} on {profile:?}",
                );
                covered += task.count;
            }
            assert_eq!(covered, rows, "{label} cover every row once");
        }
    }
}

#[test]
fn a_starved_row_folds_out_of_the_partials_its_workgroups_fill() {
    let columns = 32000u32;
    let graph = Graph::new();
    let input = graph.input(Shape::matrix(1, columns), Element::Single);
    let summed = graph.sum_rows(input);
    graph.retain(summed);
    let plan = plan_with(&graph, wide());
    let records = values_of(&plan);
    let folds = tasks_of(&plan, summed);
    assert_eq!(
        folds.len() as u32,
        1,
        "one row folds its column sections once"
    );
    assert_eq!(folds.iter().map(|task| task.count).sum::<u32>(), 1);
    let partials = folds[0].a;
    let pieces = records[partials as usize].dims[3];
    assert!(
        pieces < columns,
        "one row of {columns} numbers folds out of {pieces} partials of its own",
    );
    let chunks = tasks(&plan)
        .into_iter()
        .filter(|task| task.out == partials)
        .collect::<Vec<_>>();
    assert_eq!(
        chunks.len() as u32,
        pieces,
        "the row parts its columns between the workgroups the device leaves idle",
    );
    assert!(
        chunks
            .iter()
            .all(|task| task.geometry == strategy::WORKGROUP_ROW),
        "a column section folds the lane its workgroup walks",
    );
    assert_eq!(chunks.iter().map(|task| task.count).sum::<u32>(), pieces);
}

#[test]
fn a_fold_walks_a_long_axis_in_stages_instead_of_one_lane() {
    let graph = Graph::new();
    let short_input = graph.input(Shape::matrix(512, 8), Element::Single);
    let short = graph.sum_rows(short_input);
    let long_input = graph.input(Shape::matrix(2, 40000), Element::Single);
    let long = graph.sum_rows(long_input);
    let deep_input = graph.input(Shape::matrix(4096, 4096), Element::Single);
    let deep = graph.sum_rows(deep_input);
    let turned = graph.sum_axis(graph.input(Shape::of([4, 64, 1024, 4]), Element::Single), 1);
    graph.retain(short);
    graph.retain(long);
    graph.retain(deep);
    graph.retain(turned);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        let records = values_of(&plan);
        let short = tasks_of(&plan, short);
        assert!(
            short
                .iter()
                .all(|task| task.geometry == strategy::THREAD_ELEMENT),
            "a lane folds one element of the target, never a row of the workgroup",
        );
        assert_eq!(
            short.iter().map(|task| task.count).sum::<u32>(),
            512,
            "a fold hands every element of its target a task",
        );
        assert!(
            short.iter().all(|task| task.a == short_input.id()),
            "an axis of eight elements folds out of the tensor itself",
        );
        let long = tasks_of(&plan, long);
        assert_eq!(long.iter().map(|task| task.count).sum::<u32>(), 2);
        assert!(
            long.iter()
                .all(|task| task.slot == 3 && records[task.a as usize].dims[3] < 40000),
            "an axis of 40000 elements folds out of partials of its own",
        );
        let deep = tasks_of(&plan, deep);
        assert_eq!(deep.iter().map(|task| task.count).sum::<u32>(), 4096);
        assert!(
            deep.iter()
                .all(|task| task.slot == 3 && records[task.a as usize].dims[3] < 4096),
            "a lane folds a bounded span of every axis it walks",
        );
        let turned = tasks_of(&plan, turned);
        assert_eq!(
            turned.iter().map(|task| task.count).sum::<u32>(),
            4 * 1024 * 4
        );
        assert!(
            turned
                .iter()
                .all(|task| task.slot == 1 && records[task.a as usize].dims[1] < 8),
            "the axis a fold names decides the stage, not the place it stands in",
        );
    }
}

#[test]
fn a_product_hands_the_device_a_tile_for_every_plane() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([2, 3, 64, 16]), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::of([2, 3, 16, 32]), Init::Zero, Element::Single);
    let batched = graph.matmul(left, right);
    let shared = graph.matmul(
        left,
        graph.parameter(Shape::matrix(16, 32), Init::Zero, Element::Single),
    );
    let spread = graph.matmul(
        graph.parameter(Shape::of([2, 1, 64, 16]), Init::Zero, Element::Single),
        right,
    );
    graph.retain(batched);
    graph.retain(shared);
    graph.retain(spread);
    assert_eq!(batched.shape(), Shape::of([2, 3, 64, 32]));
    assert_eq!(shared.shape(), Shape::of([2, 3, 64, 32]));
    assert_eq!(spread.shape(), Shape::of([2, 3, 64, 32]));
    let plan = plan_with(&graph, wide());
    let geometries = plan.matmul_geometries();
    assert_eq!(
        geometries.len(),
        1,
        "three products of one shape take one tile of it",
    );
    let (tile, count) = geometries[0];
    let per_plane = 64u32.div_ceil(tile.rows()) * 32u32.div_ceil(tile.columns());
    assert_eq!(
        count,
        3 * 6 * per_plane,
        "every plane of every product carries its own tiles",
    );
    for product in [batched, shared, spread] {
        let tasks = tasks_of(&plan, product);
        assert_eq!(tasks.len() as u32, 6 * per_plane);
        for (tile_index, task) in tasks.into_iter().enumerate() {
            assert_eq!(Kind::of(task.kind), Kind::Matmul);
            assert_eq!(task.first, tile_index as u32);
            assert_eq!(task.count, 1);
            assert_eq!(
                task.splits, 1,
                "a product of one depth block splits nothing"
            );
        }
    }
    assert_eq!(
        plan.work(),
        u64::from(3 * 6 * per_plane) * tile.tile_work(),
        "a plan accounts the tiles of every plane",
    );
}

#[test]
fn a_product_refuses_batches_that_do_not_meet() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([2, 1, 4, 3]), Init::Zero, Element::Single);
    assert!(refuses(|| {
        let _ = graph.matmul(
            left,
            graph.parameter(Shape::of([3, 1, 3, 5]), Init::Zero, Element::Single),
        );
    }));
    assert!(refuses(|| {
        let _ = graph.matmul(
            left,
            graph.parameter(Shape::of([2, 1, 4, 5]), Init::Zero, Element::Single),
        );
    }));
}

#[test]
fn a_choice_covers_every_row_once() {
    let rows = 1000u32;
    let graph = Graph::new();
    let seed = graph.input(Shape::scalar(), Element::Single);
    let action = graph.categorical(graph.input(Shape::matrix(rows, 16), Element::Single), seed);
    graph.retain(action);
    let plan = plan_with(&graph, wide());
    let tasks = tasks_of(&plan, action);
    let per_task = tasks
        .first()
        .expect("a choice of a thousand rows holds a task")
        .count;
    assert!(
        per_task > 1,
        "a choice hands every row a task of its own instead of a block of rows",
    );
    let mut cursor = 0;
    for task in tasks {
        assert_eq!(
            task.first, cursor,
            "a block of rows begins where no block before it ended",
        );
        assert_eq!(
            task.count, per_task,
            "a choice hands its rows to the device in blocks of one size",
        );
        assert_eq!(Kind::of(task.kind), Kind::Categorical);
        assert_eq!(task.b, seed.id());
        cursor += task.count;
    }
    assert_eq!(cursor, rows);
    assert_eq!(plan.span(action, PLACEMENT).elements, rows);
}

#[test]
fn an_index_list_carries_one_index_per_row() {
    let graph = Graph::new();
    let indices = graph.input(Shape::matrix(6, 1), Element::Single);
    let mask = graph.one_hot(indices, 4);
    graph.retain(mask);
    let plan = plan_with(&graph, wide());
    let tasks = tasks_of(&plan, mask);
    assert_eq!(tasks.len(), 1);
    assert_eq!(Kind::of(tasks[0].kind), Kind::OneHot);
    assert_eq!(tasks[0].a, indices.id());
    assert_eq!(plan.span(mask, PLACEMENT).elements, 24);
    assert_eq!(graph.shape(mask), Shape::matrix(6, 4));
}

#[test]
fn a_gather_copies_the_rows_of_the_table_it_names() {
    let graph = Graph::new();
    let table = graph.input(Shape::matrix(5, 3), Element::Single);
    let indices = graph.input(Shape::matrix(7, 1), Element::Single);
    let picked = graph.gather(table, indices);
    graph.retain(picked);
    let plan = plan_with(&graph, wide());
    let tasks = tasks_of(&plan, picked);
    assert_eq!(tasks.len(), 1);
    assert_eq!(Kind::of(tasks[0].kind), Kind::Gather);
    assert_eq!(tasks[0].a, table.id());
    assert_eq!(tasks[0].b, indices.id());
    assert_eq!(plan.span(picked, PLACEMENT).elements, 21);
    assert_eq!(graph.shape(picked), Shape::matrix(7, 3));
}

#[test]
fn a_convolution_hands_the_device_the_window_it_walks() {
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 64, 64]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1], [1, 1]);
    let convolved = graph.conv2d(input, filter, window);
    assert_eq!(convolved.shape(), Shape::of([2, 4, 32, 32]));
    graph.retain(convolved);
    let plan = plan_with(&graph, wide());
    let tasks = tasks_of(&plan, convolved);
    assert_eq!(tasks.len(), 4);
    let mut cursor = 0;
    for task in tasks {
        assert_eq!(Kind::of(task.kind), Kind::Conv2d);
        assert_eq!(task.a, input.id());
        assert_eq!(task.b, filter.id());
        assert_eq!(task.first, cursor);
        assert_eq!(task.count, 2048);
        assert_eq!(task.stride_rows, 2);
        assert_eq!(task.stride_columns, 2);
        assert_eq!(task.pad_rows, 1);
        assert_eq!(task.pad_columns, 1);
        assert_eq!(task.dilation_rows, 1);
        assert_eq!(task.dilation_columns, 1);
        cursor += task.count;
    }
    assert_eq!(cursor, 8192);
    assert_eq!(plan.span(convolved, PLACEMENT).elements, 8192);
    assert_eq!(
        plan.work(),
        8192 * 27,
        "a convolution accounts every channel of every tap it reads",
    );
}

#[test]
fn a_dilated_convolution_hands_the_device_the_span_it_walks() {
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 64, 64]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1], [4, 3]);
    let convolved = graph.conv2d(input, filter, window);
    assert_eq!(convolved.shape(), Shape::of([2, 4, 29, 30]));
    graph.retain(convolved);
    let plan = plan_with(&graph, wide());
    let tasks = tasks_of(&plan, convolved);
    assert_eq!(tasks.len(), 4);
    let mut cursor = 0;
    for task in tasks {
        assert_eq!(Kind::of(task.kind), Kind::Conv2d);
        assert_eq!(task.dilation_rows, 4);
        assert_eq!(task.dilation_columns, 3);
        assert_eq!(task.first, cursor);
        cursor += task.count;
    }
    assert_eq!(cursor, 6960);
    assert_eq!(
        plan.work(),
        6960 * 27,
        "a dilation changes the numbers a tap reads, not the taps a convolution accounts",
    );
}

#[test]
fn a_convolution_weight_gradient_chunks_its_positions_and_folds_them() {
    let graph = Graph::new();
    let input = graph.parameter(Shape::of([1, 16, 8, 8]), Init::Zero, Element::Single);
    let filter = graph.parameter(Shape::of([16, 16, 3, 3]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [1, 1], [1, 1], [1, 1]);
    let convolved = graph.conv2d(input, filter, window);
    let gradients = graph.backward(graph.sum(convolved));
    let output_grad = gradients.of(convolved);
    let weight_grad = gradients.of(filter);
    graph.retain(weight_grad);
    let plan = plan_with(&graph, wide());
    let gradient_tasks = tasks(&plan)
        .into_iter()
        .filter(|task| Kind::of(task.kind) == Kind::Conv2dWeightGrad)
        .collect::<Vec<_>>();
    let partials = gradient_tasks
        .iter()
        .find(|task| task.out != weight_grad.id())
        .map(|task| task.out)
        .expect("a weight gradient splits its positions before it folds them");
    assert_eq!(
        values_of(&plan)[partials as usize].dims,
        [1, 1, 64, 2304],
        "the chunks of a weight gradient are summed down one row each",
    );
    let mut chunks = 0u32;
    let mut folded = 0u32;
    for task in gradient_tasks {
        if task.out == weight_grad.id() {
            assert_eq!(task.geometry, strategy::WEIGHT_FOLD);
            assert_eq!(task.a, partials);
            folded += task.count;
        } else {
            assert_eq!(task.out, partials);
            assert_eq!(task.geometry, strategy::WEIGHT_CHUNK);
            assert_eq!(task.a, input.id());
            assert_eq!(task.b, output_grad.id());
            assert_eq!(task.c, filter.id());
            chunks += 1;
        }
    }
    assert_eq!(chunks, 128);
    assert_eq!(folded, 2304);
    assert_eq!(plan.span(weight_grad, PLACEMENT).elements, 2304);
}

#[test]
fn a_convolution_stops_the_graph_it_cannot_walk() {
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 6, 6]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    assert!(refuses(|| {
        let _ = graph.conv2d(input, filter, Window::sliding([5, 5]));
    }));
    assert!(refuses(|| {
        let _ = graph.conv2d(
            input,
            graph.parameter(Shape::of([4, 2, 3, 3]), Init::Zero, Element::Single),
            Window::sliding([3, 3]),
        );
    }));
    assert!(refuses(|| {
        let _ = graph.conv2d(
            graph.input(Shape::of([2, 3, 2, 2]), Element::Single),
            graph.parameter(Shape::of([4, 3, 5, 5]), Init::Zero, Element::Single),
            Window::new([5, 5], [1, 1], [1, 1], [1, 1]),
        );
    }));
}

#[test]
fn a_choice_stops_the_graph_it_cannot_fold() {
    let graph = Graph::new();
    let table = graph.parameter(Shape::matrix(4, 4), Init::Zero, Element::Single);
    let indices = graph.input(Shape::matrix(2, 1), Element::Single);
    assert!(refuses(|| {
        let _ = graph.argmax(graph.permute(table, [0, 1, 3, 2]));
    }));
    assert!(refuses(|| {
        let _ = graph.gather(
            graph.input(Shape::matrix(4, 4), Element::Single),
            graph.input(Shape::matrix(2, 2), Element::Single),
        );
    }));
    assert!(refuses(|| {
        let _ = graph.one_hot(indices, 0);
    }));
    assert!(refuses(|| {
        let _ = graph.one_hot(graph.permute(table, [0, 1, 3, 2]), 2);
    }));
    assert!(refuses(|| {
        let _ = graph.categorical(
            graph.input(Shape::vector(4), Element::Single),
            graph.input(Shape::vector(2), Element::Single),
        );
    }));
}

#[test]
fn an_index_carries_no_gradient_of_its_own() {
    let index_graph = Graph::new();
    let indices = index_graph.input(Shape::matrix(2, 1), Element::Single);
    let drawn = index_graph.argmax(indices);
    assert!(refuses(|| {
        let _ = index_graph.backward(drawn);
    }));

    let table_graph = Graph::new();
    let table = table_graph.parameter(Shape::matrix(4, 3), Init::Zero, Element::Single);
    let indices = table_graph.input(Shape::matrix(2, 1), Element::Single);
    let picked = table_graph.matmul(table_graph.one_hot(indices, 4), table);
    let gradients = table_graph.backward(table_graph.sum(picked));
    assert_eq!(gradients.of(table).shape(), Shape::matrix(4, 3));
}

#[test]
fn a_gather_walks_its_gradient_back_into_the_table_it_reads() {
    let graph = Graph::new();
    let table = graph.parameter(Shape::matrix(4, 3), Init::Zero, Element::Single);
    let indices = graph.input(Shape::matrix(2, 1), Element::Single);
    let picked = graph.gather(table, indices);
    let gradients = graph.backward(graph.sum(picked));
    assert_eq!(gradients.of(table).shape(), Shape::matrix(4, 3));
}

#[test]
fn a_scatter_adds_updates_into_the_leaf_it_names() {
    let graph = Graph::new();
    let table = graph.resident(Shape::matrix(4, 3), Element::Single);
    let indices = graph.input(Shape::matrix(2, 1), Element::Single);
    let updates = graph.input(Shape::matrix(2, 3), Element::Single);
    graph.scatter_into(table, indices, updates);
    let plan = plan_with(&graph, narrow());
    assert_eq!(plan.task_count(), 1);
    assert!(!plan.updates_weights());
}

#[test]
fn a_scatter_stops_at_every_tensor_it_cannot_update() {
    let graph = Graph::new();
    let table = graph.resident(Shape::matrix(4, 3), Element::Single);
    let derived = graph.mul(table, table);
    let indices = graph.input(Shape::matrix(2, 1), Element::Single);
    let updates = graph.input(Shape::matrix(2, 3), Element::Single);
    let wide = graph.input(Shape::matrix(2, 4), Element::Single);
    assert!(refuses(|| {
        graph.scatter_into(derived, indices, updates);
    }));
    assert!(refuses(|| {
        graph.scatter_into(table, indices, wide);
    }));
    assert!(refuses(|| {
        graph.scatter_into(table, graph.permute(indices, [0, 1, 3, 2]), updates);
    }));
}

#[test]
fn an_attention_hands_the_device_a_row_block_for_every_plane() {
    let heads = 2u32;
    let batch = 2u32;
    let tokens = 9u32;
    let width = 4u32;
    let graph = Graph::new();
    let tensor = |tokens: u32| {
        graph.parameter(
            Shape::of([heads, batch, tokens, width]),
            Init::Zero,
            Element::Single,
        )
    };
    let out = graph.attention(
        tensor(tokens),
        tensor(tokens),
        tensor(tokens),
        neura_graph::AttentionOptions {
            bias: None,
            scale: Some(graph.knob(0.5)),
            causal: true,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        let tiles = plan.attention();
        assert_eq!(
            tiles.len(),
            1,
            "one width of one graph takes one attention tile",
        );
        assert_eq!(tiles[0].width(), width);
        assert!(
            tiles[0].shared_bytes(profile.workgroup()) <= profile.shared_bytes(),
            "an attention stages more than the pool its profile offers",
        );
        let per_task = profile.workgroup() / (tiles[0].groups() * tiles[0].slices());
        let tasks = tasks_of(&plan, out);
        let blocks = tokens.div_ceil(per_task);
        assert_eq!(
            tasks.len() as u32,
            heads * batch * blocks,
            "an attention hands the device one task per row block of every plane",
        );
        assert!(
            tiles[0].groups() > 1,
            "{} rows over {} planes leave a {}-workgroup device idle, and an attention of width {width} walks no key group beside the {} its tile names",
            tokens,
            heads * batch,
            profile.workgroup(),
            tiles[0].groups(),
        );
        let mut plane = 0u32;
        let mut block = 0u32;
        for task in tasks {
            assert_eq!(Kind::of(task.kind), Kind::Attention);
            assert_eq!(task.geometry, 0);
            assert_ne!(task.extra, neura_abi::NO_VALUE);
            assert!(task.count > 0 && task.count <= per_task);
            assert_eq!(task.first, plane * tokens + block * per_task);
            assert!(task.knob != neura_abi::NO_VALUE && task.slot == 1);
            block += 1;
            if block == blocks {
                block = 0;
                plane += 1;
            }
        }
    }
}

#[test]
fn a_key_chain_longer_than_the_device_width_walks_half_a_workgroup_of_rows() {
    let heads = 16u32;
    let tokens = 512u32;
    let width = 64u32;
    let graph = Graph::new();
    let tensor = |tokens: u32| {
        graph.parameter(
            Shape::of([heads, 1, tokens, width]),
            Init::Zero,
            Element::Single,
        )
    };
    let queries = tensor(tokens);
    let keys = tensor(tokens);
    let out = graph.attention(
        queries,
        keys,
        keys,
        neura_graph::AttentionOptions {
            bias: None,
            scale: Some(graph.knob(0.5)),
            causal: false,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    let profile = narrow();
    let plan = plan_with(&graph, profile);
    let tile = plan.attention()[0];
    let blocks = tokens.div_ceil(tile.keys());
    assert!(
        blocks >= 64,
        "{heads} planes of {tokens} rows walk {blocks} key blocks of {} keys, a chain short enough to keep whole rows",
        tile.keys(),
    );
    assert_eq!(
        tile.groups(),
        2,
        "a key chain of {blocks} blocks parts the rows of a {}-workgroup in half so the keys of one block stay together",
        profile.workgroup(),
    );
    let per_task = profile.workgroup() / (tile.groups() * tile.slices());
    assert_eq!(per_task * 2, profile.workgroup());
    assert_eq!(
        tasks_of(&plan, out).len() as u32,
        heads * tokens.div_ceil(per_task),
        "an attention of {heads} planes hands the device one task per half-workgroup row block",
    );
}

#[test]
fn a_key_chain_shorter_than_the_device_width_walks_the_rows_the_plan_asks_for() {
    let heads = 4u32;
    let tokens = 256u32;
    let width = 32u32;
    let graph = Graph::new();
    let tensor = |tokens: u32| {
        graph.parameter(
            Shape::of([heads, 1, tokens, width]),
            Init::Zero,
            Element::Single,
        )
    };
    let out = graph.attention(
        tensor(tokens),
        tensor(tokens),
        tensor(tokens),
        neura_graph::AttentionOptions {
            bias: None,
            scale: Some(graph.knob(0.5)),
            causal: false,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    let profile = wide();
    let plan = plan_with(&graph, profile);
    let tile = plan.attention()[0];
    let blocks = tokens.div_ceil(tile.keys());
    assert!(
        blocks < 64,
        "{heads} planes of {tokens} rows walk {blocks} key blocks of {} keys, a chain long enough to part their keys",
        tile.keys(),
    );
    assert!(
        tile.groups() > 1,
        "{} rows over {heads} planes leave a {}-workgroup device idle, and the attention walks no key group beside the {} its tile names",
        tokens,
        profile.workgroup(),
        tile.groups(),
    );
    assert!(
        !tasks(&plan)
            .iter()
            .any(|task| Kind::of(task.kind) == Kind::AttentionFold),
        "a key chain short enough to walk whole rows parts no keys it must fold",
    );
    let per_task = profile.workgroup() / (tile.groups() * tile.slices());
    assert_eq!(
        tasks_of(&plan, out).len() as u32,
        heads * tokens.div_ceil(per_task),
        "a short key chain hands the device one task per row block of every plane",
    );
}

#[test]
fn an_attention_parts_its_keys_beside_the_rows_the_device_already_walks() {
    let heads = 8u32;
    let tokens = 512u32;
    let width = 64u32;
    let graph = Graph::new();
    let tensor = |tokens: u32| {
        graph.parameter(
            Shape::of([heads, 1, tokens, width]),
            Init::Zero,
            Element::Single,
        )
    };
    let out = graph.attention(
        tensor(tokens),
        tensor(tokens),
        tensor(tokens),
        neura_graph::AttentionOptions {
            bias: None,
            scale: Some(graph.knob(0.5)),
            causal: false,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    let profile = wide();
    let plan = plan_with(&graph, profile);
    let tile = plan.attention()[0];
    let per_task = profile.workgroup() / (tile.groups() * tile.slices());
    let row_tasks = heads * tokens.div_ceil(per_task);
    let sections = tasks(&plan)
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::Attention)
        .map(|task| task.splits)
        .max()
        .unwrap_or(1);
    assert_eq!(
        sections, 2,
        "{heads} planes of {tokens} rows walk {row_tasks} row blocks of {per_task} rows, and the two key sections their chain balances fit beside them",
    );
    assert_eq!(
        tasks(&plan)
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::Attention)
            .count() as u32,
        row_tasks * sections,
        "every row block of every plane walks every key section once",
    );
}

#[test]
fn an_attention_keys_against_the_whole_pool_a_profile_offers() {
    let tokens = 64u32;
    let width = 64u32;
    let graph = Graph::new();
    let data = graph.parameter(
        Shape::of([1, 1, tokens, width]),
        Init::Zero,
        Element::Single,
    );
    let left = graph.parameter(Shape::matrix(64, 64), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(64, 64), Init::Zero, Element::Single);
    graph.retain(graph.matmul(left, right));
    let out = graph.attention(
        data,
        data,
        data,
        neura_graph::AttentionOptions {
            bias: None,
            scale: Some(graph.knob(0.5)),
            causal: false,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    let profile = wide();
    let plan = plan_with(&graph, profile);
    let rows = (tokens as u64).div_ceil(profile.workgroups() as u64).max(1) as u32;
    assert_eq!(
        plan.attention(),
        [AttentionTile::fit(
            profile.scratch_bytes(),
            profile.workgroup(),
            profile.registers(),
            width,
            rows,
            AttentionTile::PANELS,
        )],
        "an attention keys as deep as the pool its profile offers, while the products beside it stage from that same pool",
    );
}

#[test]
fn a_key_gradient_stages_the_forward_output_it_recomputes_with() {
    let tokens = 8u32;
    let width = 32u32;
    fn options(graph: &Graph) -> neura_graph::AttentionOptions {
        neura_graph::AttentionOptions {
            bias: None,
            scale: Some(graph.knob(0.5)),
            causal: true,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        }
    }
    let graph = Graph::new();
    let tensor = || {
        graph.parameter(
            Shape::of([1, 1, tokens, width]),
            Init::Zero,
            Element::Single,
        )
    };
    let queries = tensor();
    let keys = tensor();
    let values = tensor();
    let out = graph.attention(queries, keys, values, options(&graph));
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let query_grad = gradients.of(queries);
    let key_grad = gradients.of(keys);
    let value_grad = gradients.of(values);
    graph.retain(query_grad);
    graph.retain(key_grad);
    graph.retain(value_grad);
    let profile = narrow();
    let plan = plan_with(&graph, profile);
    let tile_of = |kind: Kind| {
        let records = tasks(&plan);
        let task = records
            .iter()
            .find(|task| Kind::of(task.kind) == kind)
            .unwrap_or_else(|| panic!("the plan walks no {kind:?} task"));
        plan.attention()[task.geometry as usize]
    };
    assert_eq!(
        tile_of(Kind::Attention).panels(),
        AttentionTile::PANELS,
        "a forward stages the keys and the values it walks, and no more",
    );
    assert_eq!(
        tile_of(Kind::AttentionQueryGrad).panels(),
        AttentionTile::PANELS,
        "a query gradient stages the keys and the values a row weighs",
    );
    assert_eq!(
        tile_of(Kind::AttentionValueGrad).panels(),
        AttentionTile::PANELS,
        "a value gradient stages the queries and the gradient a key reads",
    );
    assert_eq!(
        tile_of(Kind::AttentionKeyGrad).panels(),
        AttentionTile::RECOMPUTED_PANELS,
        "a key gradient stages the forward output it recomputes with beside the query and its gradient",
    );
    let forward = Graph::new();
    let data = forward.parameter(
        Shape::of([1, 1, tokens, width]),
        Init::Zero,
        Element::Single,
    );
    forward.retain(forward.attention(data, data, data, options(&forward)));
    let plan = plan_with(&forward, profile);
    assert!(
        plan.attention()
            .iter()
            .all(|tile| tile.panels() == AttentionTile::PANELS),
        "a plan without a key gradient stages no more than the operands its bodies walk",
    );
}

#[test]
fn a_starved_attention_parts_every_row_block_across_its_threads_and_key_sections() {
    let graph = Graph::new();
    let tokens = 100u32;
    let width = 96u32;
    let tensor = graph.parameter(
        Shape::of([1, 1, tokens, width]),
        Init::Zero,
        Element::Single,
    );
    let out = graph.attention(
        tensor,
        tensor,
        tensor,
        neura_graph::AttentionOptions {
            bias: None,
            scale: Some(graph.knob(0.5)),
            causal: false,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    let profile = narrow();
    let plan = plan_with(&graph, profile);
    let tile = plan.attention()[0];
    assert_eq!(
        tile.slices(),
        2,
        "a row of {width} numbers spreads over two threads",
    );
    let per_task = profile.workgroup() / (tile.groups() * tile.slices());
    let records = tasks(&plan);
    let partials = records
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::Attention)
        .collect::<Vec<_>>();
    let sections = partials
        .iter()
        .map(|task| task.section)
        .max()
        .expect("a starved attention walks a key section")
        + 1;
    assert!(
        sections > 1,
        "an attention whose rows starve the device parts the keys {tokens} rows walk",
    );
    let blocks = tokens.div_ceil(per_task);
    let partials_value = partials[0].out;
    assert_eq!(
        partials.len() as u32,
        blocks * sections,
        "an attention hands the device one task per {per_task} row block of each of {sections} key sections",
    );
    for section in 0..sections {
        let mut firsts = partials
            .iter()
            .filter(|task| task.section == section)
            .map(|task| task.first)
            .collect::<Vec<_>>();
        firsts.sort_unstable();
        firsts.dedup();
        assert_eq!(
            firsts.len() as u32,
            blocks,
            "key section {section} walks every row block once",
        );
    }
    for task in &partials {
        assert_eq!(task.splits, sections);
        assert_eq!(task.out, partials_value);
        assert!(task.count > 0 && task.count <= per_task);
    }
    let folds = records
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::AttentionFold)
        .collect::<Vec<_>>();
    assert!(
        !folds.is_empty(),
        "the key sections of a row block come back together in a fold",
    );
    for fold in folds {
        assert_eq!(fold.out, out.id());
        assert_eq!(fold.a, partials_value);
        assert_eq!(fold.splits, sections);
    }
}

#[test]
fn a_pointwise_task_walks_the_plan_index_over_the_tensors_it_shares_a_frame_with() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(64, 64), Element::Single);
    let scaled = graph.mul(data, graph.fill(Shape::scalar(), 2.0));
    let shifted = graph.add(graph.relu(scaled), data);
    graph.retain(shifted);
    let plan = plan_with(&graph, wide());
    let tasks = tasks(&plan)
        .into_iter()
        .filter(|task| matches!(Kind::of(task.kind), Kind::Unary | Kind::Binary))
        .collect::<Vec<_>>();
    assert!(
        tasks.len() >= 2,
        "a chain of a product and a sum walks two tasks"
    );
    for task in tasks {
        assert_eq!(
            task.geometry,
            strategy::INDEX,
            "a {} task over tensors of one dense frame walks the index the plan hands it",
            Kind::of(task.kind).name(),
        );
    }
}

#[test]
fn a_pointwise_task_walks_the_frame_of_a_row_that_shares_no_frame() {
    let graph = Graph::new();
    let rows = graph.input(Shape::of([1, 1, 4, 3]), Element::Single);
    let row = graph.input(Shape::of([1, 1, 1, 3]), Element::Single);
    let summed = graph.add(rows, row);
    graph.retain(summed);
    let plan = plan_with(&graph, wide());
    let tasks = tasks_of(&plan, summed);
    assert_eq!(tasks.len(), 1);
    assert_eq!(
        tasks[0].geometry,
        strategy::FRAME,
        "a row that spans one column of its frame walks the frame",
    );
}

#[test]
fn a_pointwise_task_walks_the_frame_of_a_view_that_shares_no_frame() {
    let graph = Graph::new();
    let left = graph.input(Shape::of([1, 1, 3, 3]), Element::Single);
    let square = graph.input(Shape::of([1, 1, 3, 3]), Element::Single);
    let turned = graph.permute(square, [0, 1, 3, 2]);
    let walked = graph.add(left, turned);
    graph.retain(walked);
    let plan = plan_with(&graph, wide());
    let tasks = tasks_of(&plan, walked);
    assert_eq!(tasks.len(), 1);
    assert_eq!(
        tasks[0].geometry,
        strategy::FRAME,
        "a tensor a permutation turns walks the frame it names",
    );
}

#[test]
fn a_pointwise_task_walks_the_frame_of_a_last_axis_a_binding_shrinks() {
    let graph = Graph::new();
    let columns = graph.free(64);
    let shape = Shape::of([1, 1, 4, 64]).freed(&[(3, columns)]);
    let rows = graph.input(shape, Element::Single);
    let other = graph.input(shape, Element::Single);
    let summed = graph.add(rows, other);
    graph.retain(summed);
    let plan = plan_with(&graph, wide());
    let tasks = tasks_of(&plan, summed);
    assert_eq!(tasks.len(), 1);
    assert_eq!(
        tasks[0].geometry,
        strategy::FRAME,
        "a walk the plan index names reads the number of a step by the stride of the last axis, and a binding of one number hands that axis no step: a last axis a free extent walks reads its own frame",
    );
}

#[test]
fn a_convert_walks_the_frame_of_a_last_axis_a_binding_shrinks() {
    let graph = Graph::new();
    let columns = graph.free(64);
    let shape = Shape::of([1, 1, 4, 64]).freed(&[(3, columns)]);
    let rows = graph.input(shape, Element::Single);
    let narrowed = graph.cast(rows, Element::Half);
    graph.retain(narrowed);
    let plan = plan_with(&graph, wide());
    let converts = tasks_of(&plan, narrowed);
    assert!(
        !converts.is_empty(),
        "a narrow tensor is written by a convert"
    );
    for task in converts {
        assert_eq!(
            task.geometry,
            strategy::FRAME,
            "a convert of a last axis a free extent walks reads the frame of every number it narrows",
        );
    }
}

#[test]
fn a_convert_walks_the_plan_index_over_the_tensor_it_narrows() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(16, 16), Element::Single);
    let narrowed = graph.cast(graph.relu(data), Element::Half);
    graph.retain(narrowed);
    let plan = plan_with(&graph, wide());
    let converts = tasks_of(&plan, narrowed);
    assert!(
        !converts.is_empty(),
        "a narrow tensor is written by a convert"
    );
    for task in converts {
        assert_eq!(task.geometry, strategy::INDEX);
    }
}

#[test]
fn a_sample_walks_the_body_its_rounds_ask_for() {
    let graph = Graph::new();
    let narrow_rows = graph.sample(
        graph.input(Shape::matrix(4096, 16), Element::Single),
        graph.input(Shape::scalar(), Element::Single),
        graph.input(Shape::scalar(), Element::Single),
        graph.input(Shape::scalar(), Element::Single),
    );
    let wide_rows = graph.sample(
        graph.input(Shape::matrix(4, 4096), Element::Single),
        graph.input(Shape::scalar(), Element::Single),
        graph.input(Shape::scalar(), Element::Single),
        graph.input(Shape::scalar(), Element::Single),
    );
    graph.retain(narrow_rows);
    graph.retain(wide_rows);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        for (value, rows, geometry, label) in [
            (
                narrow_rows,
                4096u32,
                strategy::THREAD_ROW,
                "4096 rows of 16 columns",
            ),
            (
                wide_rows,
                4,
                strategy::WORKGROUP_ROW,
                "4 rows of 4096 columns",
            ),
        ] {
            let tasks = tasks_of(&plan, value);
            let mut cursor = 0;
            for task in &tasks {
                assert_eq!(Kind::of(task.kind), Kind::Sample);
                assert_eq!(
                    task.geometry, geometry,
                    "{label} draw through {geometry} on {profile:?}",
                );
                assert_eq!(
                    task.first, cursor,
                    "a block of rows begins where no block before it ended",
                );
                cursor += task.count;
            }
            assert_eq!(cursor, rows, "{label} cover every row once");
        }
    }
}

#[test]
fn a_top_k_walks_the_body_its_rounds_ask_for() {
    let graph = Graph::new();
    let value = graph.input(Shape::matrix(4096, 16), Element::Single);
    let (narrow_values, narrow_indices) = graph.top_k(value, 4);
    let wide_rows = graph.input(Shape::matrix(4, 4096), Element::Single);
    let (wide_values, wide_indices) = graph.top_k(wide_rows, 4);
    graph.retain(narrow_values);
    graph.retain(narrow_indices);
    graph.retain(wide_values);
    graph.retain(wide_indices);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        for (kept, indices, rows, geometry, label) in [
            (
                narrow_values,
                narrow_indices,
                4096u32,
                strategy::THREAD_ROW,
                "4096 rows of 16 columns",
            ),
            (
                wide_values,
                wide_indices,
                4,
                strategy::WORKGROUP_ROW,
                "4 rows of 4096 columns",
            ),
        ] {
            let tasks = tasks_of(&plan, kept);
            let mut cursor = 0;
            for task in &tasks {
                assert_eq!(Kind::of(task.kind), Kind::TopK);
                assert_eq!(
                    task.geometry, geometry,
                    "{label} rank through {geometry} on {profile:?}",
                );
                assert_eq!(task.out, kept.id());
                assert_eq!(task.extra, indices.id());
                assert_eq!(task.keep, 4);
                assert_eq!(
                    task.first, cursor,
                    "a block of rows begins where no block before it ended",
                );
                cursor += task.count;
            }
            assert_eq!(cursor, rows, "{label} cover every row once");
        }
        assert_eq!(plan.span(narrow_values, PLACEMENT).elements, 4096 * 4);
        assert_eq!(plan.span(narrow_indices, PLACEMENT).elements, 4096 * 4);
    }
}

#[test]
fn a_top_k_of_a_thousand_rows_holds_a_task() {
    let rows = 1000u32;
    let graph = Graph::new();
    let value = graph.input(Shape::matrix(rows, 16), Element::Single);
    let (kept_values, kept_indices) = graph.top_k(value, 4);
    graph.retain(kept_values);
    graph.retain(kept_indices);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        let tasks = tasks_of(&plan, kept_values);
        assert!(!tasks.is_empty(), "a top k of a thousand rows holds a task");
        let mut cursor = 0;
        for task in &tasks {
            assert_eq!(Kind::of(task.kind), Kind::TopK);
            assert_eq!(task.a, value.id());
            assert_eq!(task.out, kept_values.id());
            assert_eq!(task.extra, kept_indices.id());
            assert_eq!(task.keep, 4);
            assert_eq!(
                task.first, cursor,
                "a block of rows begins where no block before it ended",
            );
            cursor += task.count;
        }
        assert_eq!(cursor, rows);
        assert_eq!(plan.span(kept_values, PLACEMENT).elements, rows * 4);
        assert_eq!(plan.span(kept_indices, PLACEMENT).elements, rows * 4);
    }
}

#[test]
fn a_starved_top_k_parts_its_columns_between_the_tasks_of_its_row() {
    let graph = Graph::new();
    let value = graph.input(Shape::matrix(1, 8192), Element::Single);
    let (kept_values, kept_indices) = graph.top_k(value, 40);
    graph.retain(kept_values);
    graph.retain(kept_indices);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        let all = tasks(&plan);
        let chunks = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::TopK)
            .collect::<Vec<_>>();
        let folds = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::TopKFold)
            .collect::<Vec<_>>();
        assert!(
            chunks.len() > 1,
            "one row of 8192 numbers starves the column sections of one task",
        );
        assert_eq!(folds.len(), 1, "one row folds its candidates once");
        assert_eq!(folds[0].out, kept_values.id());
        assert_eq!(folds[0].extra, kept_indices.id());
        assert_eq!(folds[0].splits as usize, chunks.len());
        assert_eq!(folds[0].first, 0);
        assert_eq!(folds[0].count, 1);
        let mut sections = chunks.iter().map(|task| task.section).collect::<Vec<_>>();
        sections.sort_unstable();
        sections.dedup();
        assert_eq!(
            sections.len(),
            chunks.len(),
            "every column section walks a share of its own",
        );
        for task in &chunks {
            assert_eq!(task.geometry, strategy::WORKGROUP_ROW);
            assert_eq!(task.a, value.id());
            assert_eq!(task.first, 0);
            assert_eq!(task.count, 1);
            assert_eq!(task.keep, 40);
            assert_eq!(task.splits as usize, chunks.len());
        }
    }
}

#[test]
fn a_starved_sample_parts_its_classes_between_the_tasks_of_its_row() {
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(1, 8192), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let keep = graph.parameter(Shape::scalar(), Init::Constant(4.0), Element::Single);
    let cumulative = graph.parameter(Shape::scalar(), Init::Constant(0.9), Element::Single);
    let draw = graph.sample(logits, seed, keep, cumulative);
    graph.retain(draw);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        let all = tasks(&plan);
        let chunks = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::Sample)
            .collect::<Vec<_>>();
        let folds = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::SampleFold)
            .collect::<Vec<_>>();
        assert!(
            chunks.len() > 1,
            "one row of 8192 classes starves the column sections of one task",
        );
        assert_eq!(folds.len(), 1, "one row folds its candidates once");
        assert_eq!(folds[0].out, draw.id());
        assert_eq!(folds[0].splits as usize, chunks.len());
        for task in &chunks {
            assert_eq!(task.geometry, strategy::WORKGROUP_ROW);
            assert_eq!(task.a, logits.id());
            assert_eq!(task.b, seed.id());
            assert_eq!(task.c, keep.id());
            assert_eq!(task.d, cumulative.id());
            assert_eq!(task.first, 0);
            assert_eq!(task.count, 1);
            assert_eq!(task.splits as usize, chunks.len());
        }
    }
}

#[test]
fn a_starved_argmax_parts_its_columns_between_the_tasks_of_its_row() {
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(1, 8192), Element::Single);
    let action = graph.argmax(logits);
    graph.retain(action);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        let all = tasks(&plan);
        let chunks = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::Argmax)
            .collect::<Vec<_>>();
        let folds = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::ChoiceFold)
            .collect::<Vec<_>>();
        assert!(
            chunks.len() > 1,
            "one row of 8192 numbers starves the column sections of one task",
        );
        assert_eq!(folds.len(), 1, "one row folds its candidates once");
        assert_eq!(folds[0].out, action.id());
        assert_eq!(folds[0].a, chunks[0].out);
        assert_eq!(folds[0].splits as usize, chunks.len());
        assert_eq!(folds[0].first, 0);
        assert_eq!(folds[0].count, 1);
        let mut sections = chunks.iter().map(|task| task.section).collect::<Vec<_>>();
        sections.sort_unstable();
        sections.dedup();
        assert_eq!(
            sections.len(),
            chunks.len(),
            "every column section walks a share of its own",
        );
        for task in &chunks {
            assert_eq!(task.geometry, strategy::WORKGROUP_ROW);
            assert_eq!(task.a, logits.id());
            assert_eq!(task.out, chunks[0].out);
            assert_eq!(task.first, 0);
            assert_eq!(task.count, 1);
            assert_eq!(task.splits as usize, chunks.len());
            assert_eq!(
                task.steps, 0,
                "a column section hands its chain to the fold"
            );
        }
    }
}

#[test]
fn a_starved_categorical_parts_its_classes_between_the_tasks_of_its_row() {
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(1, 8192), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let draw = graph.categorical(logits, seed);
    graph.retain(draw);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        let all = tasks(&plan);
        let chunks = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::Categorical)
            .collect::<Vec<_>>();
        let folds = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::ChoiceFold)
            .collect::<Vec<_>>();
        assert!(
            chunks.len() > 1,
            "one row of 8192 classes starves the column sections of one task",
        );
        assert_eq!(folds.len(), 1, "one row folds its candidates once");
        assert_eq!(folds[0].out, draw.id());
        assert_eq!(folds[0].a, chunks[0].out);
        assert_eq!(folds[0].splits as usize, chunks.len());
        for task in &chunks {
            assert_eq!(task.geometry, strategy::WORKGROUP_ROW);
            assert_eq!(task.a, logits.id());
            assert_eq!(task.b, seed.id());
            assert_eq!(task.first, 0);
            assert_eq!(task.count, 1);
            assert_eq!(task.splits as usize, chunks.len());
        }
    }
}

#[test]
fn a_wide_row_the_rows_leave_workgroups_idle_parts_its_columns() {
    let rows = 16u32;
    let columns = 32000u32;
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(rows, columns), Element::Single);
    let action = graph.argmax(logits);
    graph.retain(action);
    let plan = plan_with(&graph, wide());
    let all = tasks(&plan);
    let chunks = all
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::Argmax)
        .collect::<Vec<_>>();
    let folds = all
        .iter()
        .filter(|task| Kind::of(task.kind) == Kind::ChoiceFold)
        .collect::<Vec<_>>();
    assert!(
        chunks.len() > rows as usize,
        "sixteen rows of {columns} numbers leave workgroups of the device idle",
    );
    assert_eq!(folds.len() as u32, rows);
    let sections = chunks.len() as u32 / rows;
    assert!(sections > 1);
    assert_eq!(folds[0].splits, sections);
    for task in &chunks {
        assert_eq!(task.splits, sections);
        assert_eq!(task.count, 1);
    }
}

#[test]
fn a_choice_that_already_fills_the_device_walks_whole_rows() {
    let graph = Graph::new();
    let rows = 512u32;
    let value = graph.input(Shape::matrix(rows, 8192), Element::Single);
    let (kept_values, _) = graph.top_k(value, 40);
    graph.retain(kept_values);
    let logits = graph.input(Shape::matrix(rows, 8192), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let keep = graph.parameter(Shape::scalar(), Init::Constant(40.0), Element::Single);
    let cumulative = graph.parameter(Shape::scalar(), Init::Constant(0.9), Element::Single);
    let draw = graph.sample(logits, seed, keep, cumulative);
    graph.retain(draw);
    let action = graph.argmax(logits);
    graph.retain(action);
    let drawn = graph.categorical(logits, seed);
    graph.retain(drawn);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        assert!(
            !plan.kinds().contains(&Kind::TopKFold),
            "{rows} rows already fill the device",
        );
        assert!(!plan.kinds().contains(&Kind::SampleFold));
        assert!(!plan.kinds().contains(&Kind::ChoiceFold));
    }
}

#[test]
fn a_starved_softmax_parts_a_row_between_its_column_sections() {
    let columns = 8192u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(1, columns), Init::Zero, Element::Single);
    let probabilities = graph.softmax(logits);
    graph.retain(probabilities);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        let all = tasks(&plan);
        let chunks = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::Softmax && task.c == NO_VALUE)
            .collect::<Vec<_>>();
        let folds = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::SoftmaxFold)
            .collect::<Vec<_>>();
        let applies = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::Softmax && task.c != NO_VALUE)
            .collect::<Vec<_>>();
        let sections = columns / 1024;
        assert_eq!(
            chunks.len() as u32,
            sections,
            "one row of {columns} numbers parts its columns between {sections} tasks",
        );
        assert_eq!(folds.len(), 1, "one row folds its column sections once");
        assert_eq!(applies.len() as u32, sections);
        assert_eq!(folds[0].out, applies[0].c);
        assert_eq!(folds[0].a, chunks[0].out);
        assert_eq!(folds[0].splits, sections);
        for task in chunks.iter().chain(&applies) {
            assert_eq!(task.splits, sections);
            assert_eq!(task.a, logits.id());
            assert_eq!(task.first, 0);
            assert_eq!(task.count, 1);
        }
        for (section, task) in chunks.iter().enumerate() {
            assert_eq!(task.section as usize, section);
            assert_ne!(task.out, probabilities.id());
        }
        for (section, task) in applies.iter().enumerate() {
            assert_eq!(task.section as usize, section);
            assert_eq!(task.out, probabilities.id());
        }
    }
}

#[test]
fn a_softmax_that_already_fills_the_device_walks_whole_rows() {
    let rows = 512u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, 4096), Init::Zero, Element::Single);
    let probabilities = graph.softmax(logits);
    graph.retain(probabilities);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        assert!(
            !plan.kinds().contains(&Kind::SoftmaxFold),
            "{rows} rows fill the {profile:?} device without parting their columns",
        );
        for task in tasks(&plan)
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::Softmax)
        {
            assert_eq!(
                task.splits, 1,
                "a row the device already fills walks its whole row",
            );
        }
    }
}

#[test]
fn a_starved_class_log_softmax_parts_a_row_between_its_column_sections() {
    let columns = 8192u32;
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(1, columns), Init::Zero, Element::Single);
    let classes = graph.input(Shape::matrix(1, 1), Element::Single);
    let chosen = graph.class_log_softmax(logits, classes);
    let loss = graph.sum(chosen);
    let gradients = graph.backward(loss);
    let gradient = gradients.of(logits);
    graph.retain(loss);
    graph.retain(gradient);
    for profile in [narrow(), wide()] {
        let plan = plan_with(&graph, profile);
        assert!(plan.kinds().contains(&Kind::ClassLogSoftmaxFold));
        assert!(plan.kinds().contains(&Kind::ClassLogSoftmaxGrad));
        let all = tasks(&plan);
        let chunks = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::ClassLogSoftmax && task.out != chosen.id())
            .collect::<Vec<_>>();
        let folds = all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::ClassLogSoftmaxFold)
            .collect::<Vec<_>>();
        let sections = columns / 1024;
        assert_eq!(
            chunks.len() as u32,
            sections,
            "one row of {columns} numbers parts its columns between {sections} tasks",
        );
        assert_eq!(folds.len(), 1, "one row folds its column sections once");
        assert_eq!(folds[0].a, chunks[0].out);
        assert_eq!(folds[0].b, logits.id());
        assert_eq!(folds[0].c, classes.id());
        assert_eq!(folds[0].out, chosen.id());
        assert_ne!(
            folds[0].extra, NO_VALUE,
            "a fold publishes the normalizer every gradient reads",
        );
        assert_eq!(folds[0].splits, sections);
        for task in &chunks {
            assert_eq!(task.splits, sections);
            assert_eq!(task.a, logits.id());
            assert_eq!(task.extra, NO_VALUE);
            assert_eq!(task.first, 0);
            assert_eq!(task.count, 1);
        }
        for (section, task) in chunks.iter().enumerate() {
            assert_eq!(task.section as usize, section);
            assert_ne!(task.out, chosen.id());
        }
        for task in all
            .iter()
            .filter(|task| Kind::of(task.kind) == Kind::ClassLogSoftmaxGrad)
        {
            assert_eq!(task.a, logits.id());
            assert_eq!(task.b, classes.id());
            assert_eq!(task.d, folds[0].extra);
            assert_eq!(task.out, gradient.id());
        }
    }
}
