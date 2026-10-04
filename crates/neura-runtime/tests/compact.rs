use neura_abi::Element;
use neura_graph::{Graph, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const BOUND: u32 = 32;
const WIDTH: u32 = 4;

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn data(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..count)
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn flags(live: u32, selected: &[u32]) -> Vec<f32> {
    (0..live)
        .map(|row| if selected.contains(&row) { 1.0 } else { 0.0 })
        .collect()
}

fn gathered(table: &[f32], rows: &[u32], width: u32) -> Vec<f32> {
    let mut values = Vec::with_capacity(rows.len() * width as usize);
    for row in rows {
        let start = (row * width) as usize;
        values.extend_from_slice(&table[start..start + width as usize]);
    }
    values
}

fn total(values: &[f32]) -> f32 {
    values.iter().sum()
}

struct Selection {
    runtime: Runtime,
    graph: Graph<'static>,
    mask: Value<'static>,
    table: Value<'static>,
    indices: Value<'static>,
    count: Value<'static>,
    selected: Value<'static>,
    sum: Value<'static>,
}

impl Selection {
    fn of(bound: u32, width: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let rows = graph.free(bound);
        let mask = graph.input(Shape::matrix(bound, 1).freed(&[(2, rows)]), Element::Single);
        let compacted = graph.compact(mask);
        let table = graph.input(Shape::matrix(bound, width), Element::Single);
        let selected = graph.gather(table, compacted.indices);
        let sum = graph.sum(selected);
        graph.retain(compacted.indices);
        graph.retain(selected);
        graph.retain(sum);
        Self {
            runtime: open(),
            graph,
            mask,
            table,
            indices: compacted.indices,
            count: compacted.count,
            selected,
            sum,
        }
    }

    fn compile(&self) -> Program<'_> {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(&self, program: &Program<'_>, live: u32, selected: &[u32], seed: u32) -> SelectionOut {
        let table = data(BOUND * WIDTH, seed);
        self.runtime.bind(program, &[live]);
        self.runtime
            .write(program, self.mask, &flags(live, selected));
        self.runtime.write(program, self.table, &table);
        self.runtime.run(program);
        let indices = self
            .runtime
            .read(program, self.indices)
            .into_iter()
            .map(|row| row as u32)
            .collect::<Vec<u32>>();
        let count = self.runtime.read(program, self.count)[0];
        let gathered = self.runtime.read(program, self.selected);
        let sum = self.runtime.read(program, self.sum)[0];
        SelectionOut {
            table,
            indices,
            count,
            gathered,
            sum,
        }
    }
}

struct SelectionOut {
    table: Vec<f32>,
    indices: Vec<u32>,
    count: f32,
    gathered: Vec<f32>,
    sum: f32,
}

impl SelectionOut {
    fn check(&self, selected: &[u32]) {
        assert_eq!(
            self.indices, selected,
            "a compaction walks the rows a mask selects in order",
        );
        assert_eq!(
            self.count as usize,
            selected.len(),
            "a compaction counts the rows it selects",
        );
        assert_close(
            &self.gathered,
            &gathered(&self.table, selected, WIDTH),
            1e-6,
        );
        assert_close(
            &[self.sum],
            &[total(&gathered(&self.table, selected, WIDTH))],
            1e-5,
        );
    }
}

struct Prefix {
    runtime: Runtime,
    graph: Graph<'static>,
    value: Value<'static>,
    exclusive: Value<'static>,
    total: Value<'static>,
}

impl Prefix {
    fn of(bound: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let rows = graph.free(bound);
        let value = graph.input(Shape::matrix(bound, 1).freed(&[(2, rows)]), Element::Single);
        let prefix = graph.prefix_sum(value);
        graph.retain(prefix.exclusive);
        graph.retain(prefix.total);
        Self {
            runtime: open(),
            graph,
            value,
            exclusive: prefix.exclusive,
            total: prefix.total,
        }
    }

    fn compile(&self) -> Program<'_> {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(&self, program: &Program<'_>, numbers: &[f32]) -> (Vec<f32>, f32) {
        self.runtime.bind(program, &[numbers.len() as u32]);
        self.runtime.write(program, self.value, numbers);
        self.runtime.run(program);
        (
            self.runtime.read(program, self.exclusive),
            self.runtime.read(program, self.total)[0],
        )
    }
}

fn prefix_reference(numbers: &[f32]) -> (Vec<f32>, f32) {
    let mut running = 0.0f32;
    let mut exclusive = Vec::with_capacity(numbers.len());
    for number in numbers {
        exclusive.push(running);
        running += number;
    }
    (exclusive, running)
}

#[test]
fn a_prefix_sum_walks_the_numbers_a_binding_holds() {
    let prefix = Prefix::of(BOUND);
    let program = prefix.compile();
    let numbers = data(BOUND, 11);
    for live in [BOUND, 17, 1, 0] {
        let walked = numbers[..live as usize].to_vec();
        let (exclusive, total) = prefix.step(&program, &walked);
        let (expected, expected_total) = prefix_reference(&walked);
        assert_close(&exclusive, &expected, 1e-5);
        assert!(
            (total - expected_total).abs() <= 1e-5,
            "the walk of {} numbers totals {total} where {expected_total} was expected",
            live,
        );
    }
}

#[test]
fn a_compaction_selects_the_rows_a_mask_names() {
    let selection = Selection::of(BOUND, WIDTH);
    let program = selection.compile();
    let every = (0..BOUND).collect::<Vec<u32>>();
    for (live, selected) in [
        (
            BOUND,
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        ),
        (BOUND, vec![0, 2, 5, 17, 31]),
        (12, vec![3, 4, 11]),
        (12, vec![]),
        (1, vec![0]),
        (0, vec![]),
    ] {
        let out = selection.step(&program, live, &selected, 23);
        let mut expected = selected.clone();
        expected.sort_unstable();
        out.check(&expected);
    }
    let out = selection.step(&program, BOUND, &every, 23);
    out.check(&every);
}

#[test]
fn a_compaction_walks_the_rows_a_device_count_names() {
    let graph: Graph<'static> = Graph::new();
    let counts = graph.input(Shape::matrix(BOUND, 1), Element::Single);
    let count = graph.sum(counts);
    let mask = graph.trim(graph.mul(counts, counts), 2, count);
    let compacted = graph.compact(mask);
    let table = graph.input(Shape::matrix(BOUND, WIDTH), Element::Single);
    let selected = graph.gather(table, compacted.indices);
    graph.retain(compacted.indices);
    graph.retain(selected);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let table_values = data(BOUND * WIDTH, 29);
    runtime.write(&program, table, &table_values);
    for rows in [
        flags(BOUND, &(0..BOUND).collect::<Vec<u32>>()),
        flags(BOUND, &[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]),
        flags(BOUND, &[31]),
        flags(BOUND, &[]),
    ] {
        runtime.write(&program, counts, &rows);
        runtime.run(&program);
        let live = rows.iter().filter(|flag| **flag == 1.0).count();
        let walked = rows
            .iter()
            .enumerate()
            .take(live)
            .filter(|(_, flag)| **flag == 1.0)
            .map(|(row, _)| row as u32)
            .collect::<Vec<u32>>();
        let indices = runtime
            .read(&program, compacted.indices)
            .into_iter()
            .map(|row| row as u32)
            .collect::<Vec<u32>>();
        assert_eq!(
            indices, walked,
            "a compaction walks the rows a device count names",
        );
        assert_close(
            &runtime.read(&program, selected),
            &gathered(&table_values, &walked, WIDTH),
            1e-6,
        );
    }
}

#[test]
fn a_compaction_refuses_a_mask_whose_flag_is_neither() {
    let selection = Selection::of(BOUND, WIDTH);
    let program = selection.compile();
    let mut fractional = flags(4, &[1, 3]);
    fractional[2] = 0.5;
    selection.runtime.bind(&program, &[4]);
    selection
        .runtime
        .write(&program, selection.mask, &fractional);
    selection.runtime.run(&program);
    assert!(
        refuses(|| {
            selection.runtime.read(&program, selection.indices);
        }),
        "a mask whose flag is neither a 1 nor a 0 compacted",
    );
}

#[test]
fn a_compaction_of_a_bound_the_host_narrows_keeps_its_rows() {
    let selection = Selection::of(BOUND, WIDTH);
    let program = selection.compile();
    let wide = selection.step(&program, BOUND, &[1, 4, 9, 16, 25], 31);
    wide.check(&[1, 4, 9, 16, 25]);
    let narrow = selection.step(&program, 6, &[1, 4, 5], 31);
    narrow.check(&[1, 4, 5]);
    let again = selection.step(&program, BOUND, &[1, 4, 9, 16, 25], 31);
    again.check(&[1, 4, 9, 16, 25]);
}
