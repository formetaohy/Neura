use neura_abi::{Kind, NARROW, Precision, Profile, TaskRecord, WIDE, strategy};
use neura_program::{Encoding, Graph, Init, Placement, Shape, Value};

const ALIGNMENT: u64 = 256;
const PLACEMENT: Placement = Placement::new(1 << 20, 1 << 18, 1 << 16);

fn encoding_with(graph: &Graph, profile: Profile) -> Encoding {
    graph.encode(ALIGNMENT, profile, Precision::Single, PLACEMENT)
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
    assert_eq!(encoding.span(action).elements, rows);
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
    assert_eq!(encoding.span(mask).elements, 24);
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
    assert_eq!(encoding.span(picked).elements, 21);
    assert_eq!(graph.shape(picked), Shape::matrix(7, 3));
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
        let _ = graph.gather(table, indices);
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
