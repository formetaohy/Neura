use crate::measure;
use neura::{Element, Graph, Init, Runtime, Shape, Value};

const ROWS: u32 = 64;
const WIDE: u32 = 32768;
const STARVED: u32 = 131072;
const CANDIDATES: u32 = 64;
const KEPT: u32 = 40;
const MASS: f32 = 0.95;
const LOGITS: u32 = 32000;
const VOCABULARY: u32 = 151936;

pub(crate) fn section(runtime: &Runtime) {
    let rows = [
        starved(runtime),
        wide(runtime),
        loss(runtime),
        candidates(runtime),
        sample(runtime),
        choice(runtime),
    ];
    measure::report("decision paths", &rows);
}

fn starved(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let logits = plane(&graph, Shape::of([1, 1, 1, STARVED]));
    graph.retain(graph.softmax(logits));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(STARVED) * 8, "");
    measure::row(
        format!("a softmax over one row of {STARVED}"),
        &program,
        timing,
        rate,
    )
}

fn wide(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let logits = plane(&graph, Shape::of([1, 1, ROWS, WIDE]));
    graph.retain(graph.softmax(logits));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(ROWS * WIDE) * 8, "");
    measure::row(
        format!("a softmax over {ROWS} rows of {WIDE}"),
        &program,
        timing,
        rate,
    )
}

fn loss(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let logits = plane(&graph, Shape::of([1, 1, ROWS, WIDE]));
    let classes = graph.input(Shape::of([1, 1, ROWS, 1]), Element::Single);
    graph.retain(graph.class_log_softmax(logits, classes));
    let held = (0..ROWS).map(|at| (at * 511) as f32).collect::<Vec<f32>>();
    let program = measure::program(runtime, &graph, &[classes], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(ROWS * WIDE) * 8, "");
    measure::row(
        format!("a class log softmax over {ROWS} rows of {WIDE}"),
        &program,
        timing,
        rate,
    )
}

fn candidates(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let logits = plane(&graph, Shape::of([1, 1, 1, LOGITS]));
    let (values, indices) = graph.top_k(logits, CANDIDATES);
    graph.retain(values);
    graph.retain(indices);
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(
        timing.step,
        u64::from(LOGITS) * 4 + u64::from(CANDIDATES) * 8,
        "",
    );
    measure::row(
        format!("the top {CANDIDATES} of one row of {LOGITS}"),
        &program,
        timing,
        rate,
    )
}

fn sample(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let logits = plane(&graph, Shape::of([1, 1, 1, LOGITS]));
    let seed = graph.knob(7.0);
    let keep = graph.knob(KEPT as f32);
    let cumulative = graph.knob(MASS);
    graph.retain(graph.sample(logits, seed, keep, cumulative));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(LOGITS) * 4 + 8, "");
    measure::row(
        format!("a sample of one token from {LOGITS}"),
        &program,
        timing,
        rate,
    )
}

fn choice(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let logits = plane(&graph, Shape::of([1, 1, 1, VOCABULARY]));
    graph.retain(graph.argmax(logits));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(VOCABULARY) * 4 + 4, "");
    measure::row(
        format!("the choice of one token from {VOCABULARY}"),
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
