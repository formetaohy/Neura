use neura::{Graph, Init, Mlp, Program, Runtime, RuntimeRequest, Shape, mse_loss};
use std::time::Instant;

struct Timing {
    host: f64,
    waited: f64,
}

struct Measured {
    label: String,
    tasks: u32,
    waves: u32,
    work: u64,
    timing: Timing,
}

fn time(runtime: &Runtime, program: &Program, rounds: u32) -> Timing {
    for _ in 0..8 {
        runtime.run(program);
    }
    let started = Instant::now();
    for _ in 0..rounds {
        runtime.run(program);
    }
    let host = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    let started = Instant::now();
    for _ in 0..rounds {
        runtime.run(program);
        runtime.context().poll();
    }
    let waited = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    Timing { host, waited }
}

fn wide(runtime: &Runtime, elements: u32) -> Measured {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(elements));
    graph.relu(graph.mul(data, data));
    let program = runtime.compile(&graph);
    runtime.write(&program, data, &vec![0.5; elements as usize]);
    runtime.run(&program);
    Measured {
        label: format!("one rectifier over {elements} elements"),
        tasks: program.task_count(),
        waves: program.wave_count(),
        work: program.work(),
        timing: time(runtime, &program, 64),
    }
}

fn step(runtime: &Runtime, widths: &[u32], samples: u32) -> Measured {
    let graph = Graph::new();
    let model = Mlp::new(
        &graph,
        widths,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
    );
    let observations = graph.input(Shape::matrix(samples, widths[0]));
    let outputs = *widths.last().expect("a width");
    let targets = graph.input(Shape::matrix(samples, outputs));
    let loss = mse_loss(&graph, model.forward(&graph, observations), targets);
    graph.backward(loss);
    let program = runtime.compile(&graph);
    runtime.write(
        &program,
        observations,
        &vec![0.25; (samples * widths[0]) as usize],
    );
    runtime.write(&program, targets, &vec![0.5; (samples * outputs) as usize]);
    runtime.run(&program);
    Measured {
        label: format!("a step of {} layers at batch {samples}", widths.len() - 1),
        tasks: program.task_count(),
        waves: program.wave_count(),
        work: program.work(),
        timing: time(runtime, &program, 64),
    }
}

fn main() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        arena_bytes: 256 << 20,
        ..Default::default()
    }))
    .expect("a device to measure");
    let info = runtime.context().adapter_info();
    println!(
        "device: {} ({:?}, {:?})",
        info.name, info.backend, info.device_type
    );
    let narrow = [8, 32, 32, 8];
    let deep = [8, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 8];
    let measured = [
        wide(&runtime, 262_144),
        step(&runtime, &narrow, 8),
        step(&runtime, &narrow, 4096),
        step(&runtime, &deep, 64),
    ];
    println!(
        "{:<40} {:>7} {:>7} {:>11} {:>9} {:>9} {:>10}",
        "graph", "tasks", "waves", "work", "host us", "waited us", "per wave us",
    );
    for entry in &measured {
        println!(
            "{:<40} {:>7} {:>7} {:>11} {:>9.1} {:>9.1} {:>10.3}",
            entry.label,
            entry.tasks,
            entry.waves,
            entry.work,
            entry.timing.host,
            entry.timing.waited,
            entry.timing.host / f64::from(entry.waves.max(1)),
        );
    }
    println!(
        "{} device programs served every shape above",
        runtime.declared_kernels(),
    );
}
