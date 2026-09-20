use neura::{Adam, Graph, Init, Mlp, Program, Runtime, RuntimeRequest, Schedule, Shape, mse_loss};
use std::time::Instant;

const PRODUCT_ROWS: u32 = 1024;
const PRODUCT_DEPTH: u32 = 1024;
const PRODUCT_COLUMNS: u32 = 1024;

struct Timing {
    submit: f64,
    step: f64,
}

struct Measured {
    label: String,
    schedule: Schedule,
    tasks: u32,
    waves: u32,
    work: u64,
    timing: Timing,
}

fn time(runtime: &Runtime, program: &Program, rounds: u32) -> Timing {
    for _ in 0..8 {
        runtime.run(program);
    }
    let mut submit = 0.0;
    for _ in 0..rounds {
        drain(runtime);
        let started = Instant::now();
        runtime.run(program);
        submit += started.elapsed().as_secs_f64() * 1e6;
    }
    let submit = submit / f64::from(rounds);
    let started = Instant::now();
    for _ in 0..rounds {
        runtime.run(program);
        drain(runtime);
    }
    let step = started.elapsed().as_secs_f64() * 1e6 / f64::from(rounds);
    Timing { submit, step }
}

fn drain(runtime: &Runtime) {
    runtime.context().drain();
}

fn schedule_name(schedule: Schedule) -> String {
    let matmul = schedule.matmul();
    format!(
        "{}x{}x{}/{}",
        matmul.rows(),
        matmul.columns(),
        matmul.depth(),
        schedule.workgroup(),
    )
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
        schedule: program.schedule(),
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
    let gradients = graph.backward(loss);
    let mut optimizer = Adam::new(&graph, 0.005, 0.9, 0.999, 1e-8);
    optimizer.track_all(&graph, &model.parameters());
    optimizer.step(&graph, &gradients);
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
        schedule: program.schedule(),
        tasks: program.task_count(),
        waves: program.wave_count(),
        work: program.work(),
        timing: time(runtime, &program, 64),
    }
}

fn product(runtime: &Runtime, schedule: Schedule) -> (Measured, f64) {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(PRODUCT_ROWS, PRODUCT_DEPTH), Init::Zero);
    let right = graph.parameter(Shape::matrix(PRODUCT_DEPTH, PRODUCT_COLUMNS), Init::Zero);
    graph.retain(graph.matmul(left, right));
    let program = runtime.compile_with(&graph, schedule);
    runtime.run(&program);
    let timing = time(runtime, &program, 16);
    let flops =
        2.0 * f64::from(PRODUCT_ROWS) * f64::from(PRODUCT_DEPTH) * f64::from(PRODUCT_COLUMNS);
    let step = timing.step;
    let measured = Measured {
        label: format!("a {PRODUCT_ROWS}x{PRODUCT_DEPTH}x{PRODUCT_COLUMNS} product"),
        schedule: program.schedule(),
        tasks: program.task_count(),
        waves: program.wave_count(),
        work: program.work(),
        timing,
    };
    (measured, flops / (step * 1e-6) / 1e12)
}

fn main() {
    let runtime =
        pollster::block_on(Runtime::open(RuntimeRequest::default())).expect("a device to measure");
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
        "{:<40} {:>12} {:>7} {:>7} {:>11} {:>10} {:>9} {:>13}",
        "graph",
        "matmul/threads",
        "tasks",
        "waves",
        "work",
        "submit us",
        "step us",
        "per dispatch us",
    );
    for entry in &measured {
        println!(
            "{:<40} {:>12} {:>7} {:>7} {:>11} {:>10.1} {:>9.1} {:>13.2}",
            entry.label,
            schedule_name(entry.schedule),
            entry.tasks,
            entry.waves,
            entry.work,
            entry.timing.submit,
            entry.timing.step,
            entry.timing.step / f64::from(entry.waves.max(1)),
        );
    }
    println!();
    println!(
        "{:<40} {:>12} {:>9} {:>9} {:>13}",
        "schedule search", "matmul/threads", "tasks", "step us", "GFLOP/s",
    );
    for schedule in runtime.schedules() {
        let (entry, tflops) = product(&runtime, schedule);
        println!(
            "{:<40} {:>12} {:>9} {:>9.1} {:>13.1}",
            entry.label,
            schedule_name(entry.schedule),
            entry.tasks,
            entry.timing.step,
            tflops,
        );
    }
    let search = Graph::new();
    let left = search.parameter(Shape::matrix(PRODUCT_ROWS, PRODUCT_DEPTH), Init::Zero);
    let right = search.parameter(Shape::matrix(PRODUCT_DEPTH, PRODUCT_COLUMNS), Init::Zero);
    search.retain(search.matmul(left, right));
    let tuned = runtime.tune(&search);
    println!(
        "the device measured every schedule it offers and kept {}",
        schedule_name(tuned.schedule()),
    );
    println!(
        "{} device programs served every shape above",
        runtime.declared_kernels(),
    );
}
