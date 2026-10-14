use neura::{MatmulTile, Program, Runtime, Value};
use std::time::Instant;

pub(crate) const CHEAP: u32 = 64;
pub(crate) const HEAVY: u32 = 16;
const WARMUP: u32 = 8;

pub(crate) struct Timing {
    pub(crate) submit: f64,
    pub(crate) step: f64,
}

pub(crate) fn timed(runtime: &Runtime, program: &Program, rounds: u32) -> Timing {
    for _ in 0..WARMUP {
        runtime.run(program);
    }
    runtime.context().drain();
    let mut submit = 0.0;
    for _ in 0..rounds {
        let started = Instant::now();
        runtime.run(program);
        submit += started.elapsed().as_secs_f64() * 1e6;
        runtime.context().drain();
    }
    let mut step = 0.0;
    for _ in 0..rounds {
        step += runtime.run(program).seconds();
    }
    Timing {
        submit: submit / f64::from(rounds),
        step: step * 1e6 / f64::from(rounds),
    }
}

pub(crate) fn roundtripped(
    runtime: &Runtime,
    program: &Program,
    value: Value,
    rounds: u32,
) -> Timing {
    for _ in 0..WARMUP {
        runtime.run(program);
    }
    runtime.context().drain();
    let started = Instant::now();
    for _ in 0..rounds {
        runtime.run(program);
        let _ = runtime.read(program, value);
    }
    let round = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    let mut step = 0.0;
    for _ in 0..rounds {
        step += runtime.run(program).seconds();
    }
    Timing {
        submit: round,
        step: step * 1e6 / f64::from(rounds),
    }
}

pub(crate) fn program(
    runtime: &Runtime,
    graph: &neura::Graph,
    inputs: &[Value],
    held: &[&[f32]],
) -> Program {
    let weights = runtime.weights(graph);
    let program = runtime.compile(graph, &weights);
    for (input, values) in inputs.iter().zip(held) {
        runtime.write(&program, *input, values);
    }
    runtime.run(&program);
    program
}

pub(crate) struct Row {
    pub(crate) label: String,
    pub(crate) tiles: String,
    pub(crate) tasks: u32,
    pub(crate) waves: u32,
    pub(crate) timing: Timing,
    pub(crate) rate: String,
}

pub(crate) fn row(
    label: impl Into<String>,
    program: &Program,
    timing: Timing,
    rate: String,
) -> Row {
    Row {
        label: label.into(),
        tiles: tiles(program),
        tasks: program.task_count(),
        waves: program.wave_count(),
        timing,
        rate,
    }
}

pub(crate) fn tiles(program: &Program) -> String {
    program
        .matmul_geometries()
        .iter()
        .map(|(tile, count)| format!("{}x{}", name(*tile), count))
        .collect::<Vec<_>>()
        .join(" ")
}

fn name(tile: MatmulTile) -> String {
    format!(
        "{}x{}x{}/{}",
        tile.rows(),
        tile.columns(),
        tile.depth(),
        tile.threads()
    )
}

pub(crate) fn each(step: f64, count: u32, unit: &str) -> String {
    format!("{:.2} us a {unit}", step / f64::from(count.max(1)))
}

pub(crate) fn traffic(step: f64, bytes: u64, of: &str) -> String {
    let held = format!("{:.0} GB/s", bytes as f64 / step / 1e3);
    if of.is_empty() {
        return held;
    }
    format!("{held} of {of}")
}

pub(crate) fn report(section: &str, rows: &[Row]) {
    let width = rows
        .iter()
        .map(|row| row.label.len())
        .max()
        .unwrap_or(5)
        .max(5);
    println!("\n{section}");
    println!(
        "{:<width$} {:>20} {:>6} {:>6} {:>9} {:>9}  rate",
        "graph",
        "matmul tiles",
        "tasks",
        "waves",
        "submit us",
        "step us",
        width = width,
    );
    for entry in rows {
        println!(
            "{:<width$} {:>20} {:>6} {:>6} {:>9.1} {:>9.1}  {}",
            entry.label,
            entry.tiles,
            entry.tasks,
            entry.waves,
            entry.timing.submit,
            entry.timing.step,
            entry.rate,
            width = width,
        );
    }
}
