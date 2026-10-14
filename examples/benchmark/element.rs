use crate::measure;
use neura::{Element, Graph, Runtime, Shape};

const SPREAD: u32 = 1 << 20;
const TOTAL: u32 = 1 << 24;
const TOKEN: u32 = 512;

pub(crate) fn section(runtime: &Runtime) {
    let rows = [
        rectifier(runtime),
        total(runtime),
        drawn(runtime),
        cast(runtime),
        packed(runtime),
    ];
    measure::report("element paths", &rows);
}

fn rectifier(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(SPREAD), Element::Single);
    graph.retain(graph.relu(graph.mul(data, data)));
    let held = vec![0.5; SPREAD as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(SPREAD) * 8, "");
    measure::row(
        format!("a rectifier over {SPREAD} numbers"),
        &program,
        timing,
        rate,
    )
}

fn total(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(TOTAL), Element::Single);
    graph.retain(graph.sum(data));
    let held = vec![0.5; TOTAL as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(TOTAL) * 4 + 4, "");
    measure::row(
        format!("a sum over {TOTAL} numbers"),
        &program,
        timing,
        rate,
    )
}

fn drawn(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let seed = graph.knob(7.0);
    graph.retain(graph.normal(Shape::vector(SPREAD), seed));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(SPREAD) * 4, "");
    measure::row(
        format!("a normal draw over {SPREAD} numbers"),
        &program,
        timing,
        rate,
    )
}

fn cast(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(SPREAD), Element::Single);
    graph.retain(graph.cast(data, Element::Half));
    let held = vec![0.5; SPREAD as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(SPREAD) * 6, "");
    measure::row(
        format!("a cast of {SPREAD} numbers to half"),
        &program,
        timing,
        rate,
    )
}

fn packed(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, 1, TOKEN]), Element::Single);
    graph.retain(graph.block_quantize(data, Element::Q8_0));
    let held = vec![0.5; TOKEN as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, 1, "token");
    measure::row(
        format!("one token of {TOKEN} keys packed to q8_0"),
        &program,
        timing,
        rate,
    )
}
