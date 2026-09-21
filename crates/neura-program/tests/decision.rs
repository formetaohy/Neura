use neura_abi::{Kind, NARROW, Precision, Profile, TaskRecord, ValueRecord, WIDE, strategy};
use neura_program::{Encoding, Graph, Init, Placement, Shape, Value, Window};
use std::mem::size_of;

const ALIGNMENT: u64 = 256;
const PLACEMENT: Placement = Placement::new(1 << 16, 1 << 18);

fn encoding_with(graph: &Graph, profile: Profile) -> Encoding {
    graph.encode(ALIGNMENT, profile, Precision::Single)
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn tape(encoding: &Encoding) -> Vec<TaskRecord> {
    encoding
        .tasks()
        .as_chunks::<{ std::mem::size_of::<TaskRecord>() }>()
        .0
        .iter()
        .map(|task| bytemuck::pod_read_unaligned(task))
        .collect()
}

fn tasks_of(encoding: &Encoding, value: Value) -> Vec<TaskRecord> {
    tape(encoding)
        .into_iter()
        .filter(|task| task.out == value.id())
        .collect()
}

fn values_of(encoding: &Encoding) -> Vec<ValueRecord> {
    encoding
        .values()
        .as_chunks::<{ size_of::<ValueRecord>() }>()
        .0
        .iter()
        .map(|value| bytemuck::pod_read_unaligned(value))
        .collect()
}

#[test]
fn a_row_folds_in_the_strategy_its_length_asks_for() {
    let graph = Graph::new();
    let short = graph.argmax(graph.input(Shape::matrix(4, 8)));
    let middle = graph.argmax(graph.input(Shape::matrix(333, 200)));
    let long = graph.argmax(graph.input(Shape::matrix(2, 4096)));
    graph.retain(short);
    graph.retain(middle);
    graph.retain(long);
    for (profile, middle_geometry) in [
        (NARROW, strategy::WORKGROUP_ROW),
        (WIDE, strategy::THREAD_ROW),
    ] {
        let encoding = encoding_with(&graph, profile);
        assert_eq!(tasks_of(&encoding, short).len(), 4);
        assert_eq!(tasks_of(&encoding, middle).len(), 167);
        assert_eq!(tasks_of(&encoding, long).len(), 2);
        for (index, task) in tasks_of(&encoding, short).into_iter().enumerate() {
            assert_eq!(task.geometry, strategy::THREAD_ROW);
            assert_eq!(task.first, index as u32);
            assert_eq!(task.count, 1);
            assert_eq!(Kind::of(task.kind), Kind::Argmax);
        }
        let mut covered = 0;
        for task in tasks_of(&encoding, middle) {
            assert_eq!(
                task.geometry, middle_geometry,
                "a row of 200 elements folds through {middle_geometry} on {profile:?}",
            );
            assert!(task.count > 0 && task.count <= 2);
            covered += task.count;
        }
        assert_eq!(covered, 333);
        for (index, task) in tasks_of(&encoding, long).into_iter().enumerate() {
            assert_eq!(task.geometry, strategy::WORKGROUP_ROW);
            assert_eq!(task.first, index as u32);
            assert_eq!(task.count, 1);
        }
    }
}

#[test]
fn a_row_fold_hands_the_device_the_geometry_its_axis_asks_for() {
    let graph = Graph::new();
    let short = graph.sum_rows(graph.input(Shape::matrix(64, 8)));
    let long = graph.sum_rows(graph.input(Shape::matrix(2, 200)));
    let view = graph.sum_rows(graph.transpose(graph.input(Shape::matrix(200, 2))));
    graph.retain(short);
    graph.retain(long);
    graph.retain(view);
    for (profile, long_geometry) in [
        (NARROW, strategy::WORKGROUP_ROW),
        (WIDE, strategy::THREAD_ROW),
    ] {
        let encoding = encoding_with(&graph, profile);
        let short = tasks_of(&encoding, short);
        assert_eq!(short.len(), 64);
        for (index, task) in short.into_iter().enumerate() {
            assert_eq!(Kind::of(task.kind), Kind::SumAxis);
            assert_eq!(task.slot, 3);
            assert_eq!(
                task.geometry,
                strategy::THREAD_ROW,
                "a row of eight elements folds through one thread",
            );
            assert_eq!(task.first, index as u32);
            assert_eq!(task.count, 1);
        }
        let long = tasks_of(&encoding, long);
        assert_eq!(long.len(), 2);
        for (index, task) in long.into_iter().enumerate() {
            assert_eq!(
                task.geometry, long_geometry,
                "a row of 200 elements folds through {long_geometry} on {profile:?}",
            );
            assert_eq!(task.slot, 3);
            assert_eq!(task.first, index as u32);
            assert_eq!(task.count, 1);
        }
        let view = tasks_of(&encoding, view);
        assert_eq!(view.len(), 1);
        assert_eq!(
            view[0].geometry,
            strategy::THREAD_ELEMENT,
            "a row a view holds apart folds element by element",
        );
        assert_eq!(view[0].slot, 3);
        assert_eq!(view[0].count, 2);
    }
}

#[test]
fn a_product_hands_the_device_a_tile_for_every_plane() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([2, 3, 64, 16]), Init::Zero);
    let right = graph.parameter(Shape::of([2, 3, 16, 32]), Init::Zero);
    let batched = graph.matmul(left, right);
    let shared = graph.matmul(left, graph.parameter(Shape::matrix(16, 32), Init::Zero));
    let spread = graph.matmul(
        graph.parameter(Shape::of([2, 1, 64, 16]), Init::Zero),
        right,
    );
    graph.retain(batched);
    graph.retain(shared);
    graph.retain(spread);
    assert_eq!(batched.shape(), Shape::of([2, 3, 64, 32]));
    assert_eq!(shared.shape(), Shape::of([2, 3, 64, 32]));
    assert_eq!(spread.shape(), Shape::of([2, 3, 64, 32]));
    let encoding = encoding_with(&graph, WIDE);
    let geometries = encoding.matmul_geometries();
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
        let tasks = tasks_of(&encoding, product);
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
        encoding.work(),
        u64::from(3 * 6 * per_plane) * tile.tile_work(),
        "a plan accounts the tiles of every plane",
    );
}

#[test]
fn a_product_refuses_batches_that_do_not_meet() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([2, 1, 4, 3]), Init::Zero);
    assert!(refuses(|| {
        let _ = graph.matmul(left, graph.parameter(Shape::of([3, 1, 3, 5]), Init::Zero));
    }));
    assert!(refuses(|| {
        let _ = graph.matmul(left, graph.parameter(Shape::of([2, 1, 4, 5]), Init::Zero));
    }));
    assert!(refuses(|| {
        let _ = graph.sum_rows(graph.parameter(Shape::matrix(4, 1), Init::Zero));
    }));
}

#[test]
fn a_choice_covers_every_row_once() {
    let rows = 1000u32;
    let graph = Graph::new();
    let seed = graph.input(Shape::scalar());
    let action = graph.categorical(graph.input(Shape::matrix(rows, 16)), seed);
    graph.retain(action);
    let encoding = encoding_with(&graph, WIDE);
    let tasks = tasks_of(&encoding, action);
    assert_eq!(tasks.len(), 250);
    let mut cursor = 0;
    for task in tasks {
        assert_eq!(task.first, cursor);
        assert_eq!(task.count, 4);
        assert_eq!(Kind::of(task.kind), Kind::Categorical);
        assert_eq!(task.b, seed.id());
        cursor += task.count;
    }
    assert_eq!(cursor, rows);
    assert_eq!(encoding.span(action, PLACEMENT).elements, rows);
}

#[test]
fn an_index_list_carries_one_index_per_row() {
    let graph = Graph::new();
    let indices = graph.input(Shape::matrix(6, 1));
    let mask = graph.one_hot(indices, 4);
    graph.retain(mask);
    let encoding = encoding_with(&graph, WIDE);
    let tasks = tasks_of(&encoding, mask);
    assert_eq!(tasks.len(), 1);
    assert_eq!(Kind::of(tasks[0].kind), Kind::OneHot);
    assert_eq!(tasks[0].a, indices.id());
    assert_eq!(encoding.span(mask, PLACEMENT).elements, 24);
    assert_eq!(graph.shape(mask), Shape::matrix(6, 4));
}

#[test]
fn a_gather_copies_the_rows_of_the_table_it_names() {
    let graph = Graph::new();
    let table = graph.input(Shape::matrix(5, 3));
    let indices = graph.input(Shape::matrix(7, 1));
    let picked = graph.gather(table, indices);
    graph.retain(picked);
    let encoding = encoding_with(&graph, WIDE);
    let tasks = tasks_of(&encoding, picked);
    assert_eq!(tasks.len(), 1);
    assert_eq!(Kind::of(tasks[0].kind), Kind::Gather);
    assert_eq!(tasks[0].a, table.id());
    assert_eq!(tasks[0].b, indices.id());
    assert_eq!(encoding.span(picked, PLACEMENT).elements, 21);
    assert_eq!(graph.shape(picked), Shape::matrix(7, 3));
}

#[test]
fn a_convolution_hands_the_device_the_window_it_walks() {
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 64, 64]));
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero);
    let window = Window::new([3, 3], [2, 2], [1, 1]);
    let convolved = graph.conv2d(input, filter, window);
    assert_eq!(convolved.shape(), Shape::of([2, 4, 32, 32]));
    graph.retain(convolved);
    let encoding = encoding_with(&graph, WIDE);
    let tasks = tasks_of(&encoding, convolved);
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
        cursor += task.count;
    }
    assert_eq!(cursor, 8192);
    assert_eq!(encoding.span(convolved, PLACEMENT).elements, 8192);
    assert_eq!(
        encoding.work(),
        8192 * 27,
        "a convolution accounts every channel of every tap it reads",
    );
}

#[test]
fn a_convolution_weight_gradient_chunks_its_positions_and_folds_them() {
    let graph = Graph::new();
    let input = graph.parameter(Shape::of([1, 16, 8, 8]), Init::Zero);
    let filter = graph.parameter(Shape::of([16, 16, 3, 3]), Init::Zero);
    let window = Window::new([3, 3], [1, 1], [1, 1]);
    let convolved = graph.conv2d(input, filter, window);
    let gradients = graph.backward(graph.sum(convolved));
    let output_grad = gradients.of(convolved);
    let weight_grad = gradients.of(filter);
    graph.retain(weight_grad);
    let encoding = encoding_with(&graph, WIDE);
    let gradient_tasks = tape(&encoding)
        .into_iter()
        .filter(|task| Kind::of(task.kind) == Kind::Conv2dWeightGrad)
        .collect::<Vec<_>>();
    let partials = gradient_tasks
        .iter()
        .find(|task| task.out != weight_grad.id())
        .map(|task| task.out)
        .expect("a weight gradient splits its positions before it folds them");
    assert_eq!(
        values_of(&encoding)[partials as usize].dims,
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
    assert_eq!(encoding.span(weight_grad, PLACEMENT).elements, 2304);
}

#[test]
fn a_convolution_stops_the_graph_it_cannot_walk() {
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 6, 6]));
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero);
    assert!(refuses(|| {
        let _ = graph.conv2d(input, filter, Window::sliding([5, 5]));
    }));
    assert!(refuses(|| {
        let _ = graph.conv2d(
            input,
            graph.parameter(Shape::of([4, 2, 3, 3]), Init::Zero),
            Window::sliding([3, 3]),
        );
    }));
    assert!(refuses(|| {
        let _ = graph.conv2d(
            graph.input(Shape::of([2, 3, 2, 2])),
            graph.parameter(Shape::of([4, 3, 5, 5]), Init::Zero),
            Window::new([5, 5], [1, 1], [1, 1]),
        );
    }));
}

#[test]
fn a_choice_stops_the_graph_it_cannot_fold() {
    let graph = Graph::new();
    let table = graph.parameter(Shape::matrix(4, 4), Init::Zero);
    let indices = graph.input(Shape::matrix(2, 1));
    assert!(refuses(|| {
        let _ = graph.argmax(graph.transpose(table));
    }));
    assert!(refuses(|| {
        let _ = graph.gather(
            graph.input(Shape::matrix(4, 4)),
            graph.input(Shape::matrix(2, 2)),
        );
    }));
    assert!(refuses(|| {
        let _ = graph.one_hot(indices, 0);
    }));
    assert!(refuses(|| {
        let _ = graph.one_hot(graph.transpose(table), 2);
    }));
    assert!(refuses(|| {
        let _ = graph.categorical(graph.input(Shape::vector(4)), graph.input(Shape::vector(2)));
    }));
}

#[test]
fn an_index_carries_no_gradient_of_its_own() {
    let index_graph = Graph::new();
    let indices = index_graph.input(Shape::matrix(2, 1));
    let drawn = index_graph.argmax(indices);
    assert!(refuses(|| {
        let _ = index_graph.backward(drawn);
    }));

    let table_graph = Graph::new();
    let table = table_graph.parameter(Shape::matrix(4, 3), Init::Zero);
    let indices = table_graph.input(Shape::matrix(2, 1));
    let picked = table_graph.matmul(table_graph.one_hot(indices, 4), table);
    let gradients = table_graph.backward(table_graph.sum(picked));
    assert_eq!(gradients.of(table).shape(), Shape::matrix(4, 3));
}

#[test]
fn a_gather_walks_its_gradient_back_into_the_table_it_reads() {
    let graph = Graph::new();
    let table = graph.parameter(Shape::matrix(4, 3), Init::Zero);
    let indices = graph.input(Shape::matrix(2, 1));
    let picked = graph.gather(table, indices);
    let gradients = graph.backward(graph.sum(picked));
    assert_eq!(gradients.of(table).shape(), Shape::matrix(4, 3));
}

#[test]
fn a_scatter_adds_updates_into_the_leaf_it_names() {
    let graph = Graph::new();
    let table = graph.resident(Shape::matrix(4, 3));
    let indices = graph.input(Shape::matrix(2, 1));
    let updates = graph.input(Shape::matrix(2, 3));
    graph.scatter_into(table, indices, updates);
    let encoding = encoding_with(&graph, NARROW);
    assert_eq!(encoding.task_count(), 1);
    assert!(!encoding.updates_weights());
}

#[test]
fn a_scatter_stops_at_every_tensor_it_cannot_update() {
    let graph = Graph::new();
    let table = graph.resident(Shape::matrix(4, 3));
    let derived = graph.mul(table, table);
    let indices = graph.input(Shape::matrix(2, 1));
    let updates = graph.input(Shape::matrix(2, 3));
    let wide = graph.input(Shape::matrix(2, 4));
    assert!(refuses(|| {
        graph.scatter_into(derived, indices, updates);
    }));
    assert!(refuses(|| {
        graph.scatter_into(table, indices, wide);
    }));
    assert!(refuses(|| {
        graph.scatter_into(table, graph.transpose(indices), updates);
    }));
}
