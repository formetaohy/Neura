use crate::measure;
use neura::{Element, Graph, Init, Runtime, Shape};

const DEPTH: u32 = 1024;
const COLUMNS: u32 = 4096;
const VOCABULARY: u32 = 151936;
const LADDER: u32 = 1024;
const SWEEP: [u32; 4] = [64, 1024, 16384, VOCABULARY];
const NARROW: [Element; 7] = [
    Element::Single,
    Element::Half,
    Element::Q8_0,
    Element::Q3K,
    Element::Q4K,
    Element::Q5K,
    Element::Q6K,
];

pub(crate) fn section(runtime: &Runtime) {
    let mut rows = Vec::new();
    for element in NARROW {
        rows.push(product(
            runtime,
            element,
            COLUMNS,
            true,
            measure::CHEAP,
            format!(
                "a 1x{DEPTH}x{COLUMNS} product of {} in gguf order",
                element.name()
            ),
        ));
    }
    rows.push(product(
        runtime,
        Element::Single,
        COLUMNS,
        false,
        measure::CHEAP,
        format!("a 1x{DEPTH}x{COLUMNS} product of single in linear order"),
    ));
    for columns in SWEEP {
        rows.push(product(
            runtime,
            Element::Q8_0,
            columns,
            true,
            if columns >= 16384 {
                measure::HEAVY
            } else {
                measure::CHEAP
            },
            format!("a 1x{DEPTH}x{columns} product of q8_0 in gguf order"),
        ));
    }
    rows.push(product(
        runtime,
        Element::Q4K,
        6144,
        true,
        measure::CHEAP,
        format!("a 1x{DEPTH}x6144 product of q4_k in gguf order"),
    ));
    rows.push(product(
        runtime,
        Element::Q6K,
        VOCABULARY,
        true,
        measure::HEAVY,
        format!("a vocabulary of {VOCABULARY} columns of q6_k"),
    ));
    measure::report("narrow products", &rows);
    ladder(runtime);
}

fn product(
    runtime: &Runtime,
    element: Element,
    columns: u32,
    gguf: bool,
    rounds: u32,
    label: String,
) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, 1, DEPTH]), Element::Single);
    let init = Init::Uniform {
        low: -0.2,
        high: 0.2,
    };
    let weights = if gguf {
        let stored = if element.block().is_some() {
            graph.block_quantized_parameter(Shape::of([1, 1, columns, DEPTH]), init, element)
        } else {
            graph.parameter(Shape::of([1, 1, columns, DEPTH]), init, element)
        };
        graph.permute(stored, [0, 1, 3, 2])
    } else {
        graph.parameter(Shape::of([1, 1, DEPTH, columns]), init, element)
    };
    graph.retain(graph.matmul(data, weights));
    let held = vec![0.5; DEPTH as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    let words = element.payload_words(u64::from(columns) * u64::from(DEPTH));
    let timing = measure::timed(runtime, &program, rounds);
    let rate = measure::traffic(timing.step, words * 4, "");
    measure::row(label, &program, timing, rate)
}

const WIDE_ROWS: [u32; 6] = [2, 4, 8, 32, 64, 512];

pub(crate) fn wide_section(runtime: &Runtime) {
    let mut rows = Vec::new();
    for element in NARROW {
        for width in WIDE_ROWS {
            rows.push(wide(
                runtime,
                element,
                width,
                COLUMNS,
                format!(
                    "a {width}x{DEPTH}x{COLUMNS} product of {} in gguf order",
                    element.name()
                ),
            ));
        }
    }
    rows.push(wide(
        runtime,
        Element::Q8_0,
        2,
        DEPTH,
        format!("a 2x{DEPTH}x{DEPTH} product of q8_0 in gguf order"),
    ));
    rows.push(wide(
        runtime,
        Element::Q8_0,
        8,
        VOCABULARY,
        format!("an 8x{DEPTH}x{VOCABULARY} product of q8_0 in gguf order"),
    ));
    measure::report("wide products", &rows);
}

fn wide(
    runtime: &Runtime,
    element: Element,
    rows: u32,
    columns: u32,
    label: String,
) -> measure::Row {
    let graph = Graph::new();
    let data = graph.input(Shape::of([1, 1, rows, DEPTH]), Element::Single);
    let init = Init::Uniform {
        low: -0.2,
        high: 0.2,
    };
    let stored = if element.block().is_some() {
        graph.block_quantized_parameter(Shape::of([1, 1, columns, DEPTH]), init, element)
    } else {
        graph.parameter(Shape::of([1, 1, columns, DEPTH]), init, element)
    };
    let weights = graph.permute(stored, [0, 1, 3, 2]);
    graph.retain(graph.matmul(data, weights));
    let held = vec![0.5; (rows * DEPTH) as usize];
    let program = measure::program(runtime, &graph, &[data], &[&held]);
    let words = element.payload_words(u64::from(columns) * u64::from(DEPTH));
    let rounds = if rows > 64 {
        measure::HEAVY
    } else {
        measure::CHEAP
    };
    let timing = measure::timed(runtime, &program, rounds);
    let rate = measure::traffic(timing.step, words * 4, "");
    measure::row(label, &program, timing, rate)
}

fn ladder(runtime: &Runtime) {
    println!("\nprofile search");
    println!(
        "{:<40} {:>10} {:>22} {:>9} {:>13}",
        "graph", "workgroup", "matmul tiles", "step us", "TFLOP/s",
    );
    for profile in runtime.profiles() {
        let graph = Graph::new();
        let left = graph.parameter(Shape::matrix(LADDER, LADDER), Init::Zero, Element::Single);
        let right = graph.parameter(Shape::matrix(LADDER, LADDER), Init::Zero, Element::Single);
        graph.retain(graph.matmul(left, right));
        let weights = runtime.weights(&graph);
        let program = runtime.compile_with(&graph, &weights, profile);
        runtime.run(&program);
        let timing = measure::timed(runtime, &program, measure::HEAVY);
        let work = 2.0 * f64::from(LADDER).powi(3);
        println!(
            "{:<40} {:>10} {:>22} {:>9.1} {:>13.1}",
            format!("a {LADDER}x{LADDER}x{LADDER} product"),
            format!("{} threads", profile.workgroup()),
            measure::tiles(&program),
            timing.step,
            work / (timing.step * 1e-6) / 1e12,
        );
    }
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(LADDER, LADDER), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(LADDER, LADDER), Init::Zero, Element::Single);
    graph.retain(graph.matmul(left, right));
    let weights = runtime.weights(&graph);
    let tuned = runtime.tune(&graph, &weights);
    println!(
        "the device measured every profile it offers and kept {} threads",
        tuned.profile().workgroup(),
    );
}
