use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const DEPTH: u32 = 2048;
const ROWS: u32 = 8;
const COLUMNS: u32 = 4;

#[test]
fn a_step_that_reads_a_wider_row_waits_for_the_product_that_writes_it() {
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, ROWS, COLUMNS]), Element::Single);
    let row = graph.input(Shape::of([1, 1, 1, DEPTH]), Element::Single);
    let filter = graph.parameter(
        Shape::of([1, 1, DEPTH, COLUMNS]),
        Init::Zero,
        Element::Single,
    );
    let bias = graph.matmul(row, filter);
    let activated = graph.add(graph.softmax(data), bias);
    graph.retain(activated);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &vec![0.0; (ROWS * COLUMNS) as usize]);
    runtime.write(&program, row, &vec![1.0; DEPTH as usize]);
    runtime.write(
        &program,
        filter,
        &vec![1.0 / DEPTH as f32; (DEPTH * COLUMNS) as usize],
    );
    runtime.run(&program);
    let out = runtime.read(&program, activated);
    let expected = vec![1.0 / COLUMNS as f32 + 1.0; (ROWS * COLUMNS) as usize];
    assert_close(&out, &expected, 1e-6);
}
