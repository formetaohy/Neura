use crate::measure;
use neura::{Element, Graph, Init, Runtime, Shape};

const SEGMENTS: u32 = 16;
const ROWS: u32 = 256;
const BOUND: u32 = SEGMENTS * ROWS;
const DEPTH: u32 = 256;
const COLUMNS: u32 = 256;

pub(crate) fn section(runtime: &Runtime) {
    let rows = [grouped(runtime), segmented(runtime)];
    measure::report("ragged paths", &rows);
}

fn grouped(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(SEGMENTS), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let rows = graph.input(
        Shape::of([1, 1, BOUND, DEPTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let weights = graph.parameter(
        Shape::of([SEGMENTS, 1, DEPTH, COLUMNS]),
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    graph.retain(graph.grouped_matmul(rows, weights, ragged.offsets));
    let held_rows = vec![0.5; (BOUND * DEPTH) as usize];
    let held_lengths = vec![ROWS as f32; SEGMENTS as usize];
    let program = measure::program(
        runtime,
        &graph,
        &[rows, lengths],
        &[&held_rows, &held_lengths],
    );
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(
        timing.step,
        u64::from(BOUND * DEPTH + SEGMENTS * DEPTH * COLUMNS + BOUND * COLUMNS) * 4,
        "",
    );
    measure::row(
        format!("a grouped product over {SEGMENTS} segments of {ROWS} rows"),
        &program,
        timing,
        rate,
    )
}

fn segmented(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(SEGMENTS), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let values = graph.input(
        Shape::of([1, 1, BOUND, COLUMNS]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    graph.retain(graph.segment_sum(values, ragged));
    let held_values = vec![0.5; (BOUND * COLUMNS) as usize];
    let held_lengths = vec![ROWS as f32; SEGMENTS as usize];
    let program = measure::program(
        runtime,
        &graph,
        &[values, lengths],
        &[&held_values, &held_lengths],
    );
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(
        timing.step,
        u64::from(BOUND * COLUMNS + SEGMENTS * COLUMNS) * 4,
        "",
    );
    measure::row(
        format!("a segment sum over {SEGMENTS} segments of {ROWS} rows of {COLUMNS}"),
        &program,
        timing,
        rate,
    )
}
