use crate::measure;
use neura::{Element, Graph, Init, Runtime, Shape, Value, decoder};

const ROWS: u32 = 64;
const HIDDEN: u32 = 1024;
const VOCABULARY: u32 = 32768;
const HEADS: u32 = 8;
const WIDTH: u32 = 256;

pub(crate) fn section(runtime: &Runtime) {
    let rows = [gathered(runtime), joined(runtime), rotated(runtime)];
    measure::report("movement paths", &rows);
}

fn gathered(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let table = graph.parameter(
        Shape::of([1, 1, VOCABULARY, HIDDEN]),
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    let indices = graph.input(Shape::of([1, 1, ROWS, 1]), Element::Single);
    graph.retain(graph.gather(table, indices));
    let held = (0..ROWS).map(|at| (at * 511) as f32).collect::<Vec<f32>>();
    let program = measure::program(runtime, &graph, &[indices], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(ROWS * HIDDEN) * 8, "");
    measure::row(
        format!("{ROWS} rows gathered from a vocabulary of {VOCABULARY} of {HIDDEN}"),
        &program,
        timing,
        rate,
    )
}

fn joined(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let value = plane(&graph, Shape::of([HEADS, 1, ROWS, WIDTH]));
    graph.retain(decoder::joined(&graph, value, HEADS));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(HEADS * ROWS * WIDTH) * 8, "");
    measure::row(
        format!("a join of {HEADS} heads of {ROWS} rows of {WIDTH}"),
        &program,
        timing,
        rate,
    )
}

fn rotated(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let value = plane(&graph, Shape::of([HEADS, 1, ROWS, WIDTH]));
    graph.retain(graph.rope(value, None, None));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(HEADS * ROWS * WIDTH) * 8, "");
    measure::row(
        format!("a rotation over {HEADS} planes of {ROWS} rows of {WIDTH}"),
        &program,
        timing,
        rate,
    )
}

fn plane(graph: &Graph, shape: Shape) -> Value {
    graph.parameter(
        shape,
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    )
}
