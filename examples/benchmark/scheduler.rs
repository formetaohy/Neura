use crate::measure;
use neura::{Element, Graph, Init, Runtime, Shape};

const TINY: u32 = 8;
const CHAIN: u32 = 512;
const PAIRS: u32 = 256;
const GEMVS: u32 = 32;
const DEPTH: u32 = 1024;

pub(crate) fn section(runtime: &Runtime) {
    let rows = [
        dispatch(runtime),
        readback(runtime),
        walk(runtime),
        gate(runtime),
        chained(runtime),
    ];
    measure::report("fixed costs", &rows);
}

fn dispatch(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(TINY), Element::Single);
    graph.retain(graph.mul(data, data));
    let held = vec![0.5; TINY as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    assert!(
        program.task_count() == 1,
        "a benchmark of one dispatch holds one task, and this graph walks {}",
        program.task_count(),
    );
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, 1, "dispatch");
    measure::row("one tiny task", &program, timing, rate)
}

fn readback(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(TINY), Element::Single);
    let out = graph.mul(data, data);
    graph.retain(out);
    let held = vec![0.5; TINY as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    let timing = measure::roundtripped(runtime, &program, out, measure::CHEAP);
    let rate = measure::each(timing.submit, 1, "round trip");
    measure::row("one tiny task read back", &program, timing, rate)
}

fn walk(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(TINY), Element::Single);
    let mut value = data;
    for _ in 0..CHAIN {
        value = graph.mul(value, value);
    }
    graph.retain(value);
    let held = vec![0.5; TINY as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    assert_eq!(
        program.wave_count(),
        1,
        "a benchmark of the task walk chains {CHAIN} single-use tasks through one segment, and this graph gates {} waves",
        program.wave_count(),
    );
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, CHAIN, "task");
    measure::row(
        format!("a chain of {CHAIN} single-use tasks"),
        &program,
        timing,
        rate,
    )
}

fn gate(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let left = graph.input(Shape::of([1, 1, 1, TINY]), Element::Single);
    let right = graph.input(Shape::of([1, 1, 1, TINY]), Element::Single);
    let weights = graph.parameter(
        Shape::of([1, 1, TINY, TINY]),
        Init::Uniform {
            low: -0.1,
            high: 0.1,
        },
        Element::Single,
    );
    let mut a = left;
    let mut b = right;
    for _ in 0..PAIRS {
        let joined = graph.add(graph.matmul(a, weights), graph.matmul(b, weights));
        a = graph.slice(joined, 3, 0, TINY);
        b = graph.slice(joined, 3, 0, TINY);
    }
    graph.retain(a);
    let held = vec![0.5; TINY as usize];
    let program = measure::program(runtime, &graph, &[left, right], &[&held, &held]);
    assert!(
        program.wave_count() >= PAIRS,
        "a benchmark of the wave gate chains {PAIRS} pairs of products, and this graph gates only {} waves",
        program.wave_count(),
    );
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, program.wave_count(), "wave");
    measure::row(
        format!("a chain of {PAIRS} gated pairs"),
        &program,
        timing,
        rate,
    )
}

fn chained(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, 1, DEPTH]), Element::Single);
    let stored = graph.block_quantized_parameter(
        Shape::of([1, 1, DEPTH, DEPTH]),
        Init::Uniform {
            low: -0.1,
            high: 0.1,
        },
        Element::Q8_0,
    );
    let weights = graph.permute(stored, [0, 1, 3, 2]);
    let mut value = data;
    for _ in 0..GEMVS {
        value = graph.matmul(value, weights);
    }
    graph.retain(value);
    let held = vec![0.5; DEPTH as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    assert!(
        program.wave_count() >= GEMVS,
        "a benchmark of gated products chains {GEMVS} narrow products, and this graph gates only {} waves",
        program.wave_count(),
    );
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, program.wave_count(), "wave");
    measure::row(
        format!("a chain of {GEMVS} narrow products of q8_0"),
        &program,
        timing,
        rate,
    )
}
