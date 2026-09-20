use neura::wgpu::PollType;
use neura::{Graph, Init, Mlp, Program, Runtime, RuntimeRequest, Shape, mse_loss};
use std::time::Instant;

struct Timing {
    submit: f64,
    step: f64,
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
    runtime
        .context()
        .device()
        .poll(PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(30)),
        })
        .expect("the device finished the work it was handed");
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
        "{:<40} {:>7} {:>7} {:>11} {:>10} {:>9} {:>13}",
        "graph", "tasks", "waves", "work", "submit us", "step us", "per dispatch us",
    );
    for entry in &measured {
        println!(
            "{:<40} {:>7} {:>7} {:>11} {:>10.1} {:>9.1} {:>13.2}",
            entry.label,
            entry.tasks,
            entry.waves,
            entry.work,
            entry.timing.submit,
            entry.timing.step,
            entry.timing.step / f64::from(entry.waves.max(1)),
        );
    }
    println!(
        "{} device programs served every shape above",
        runtime.declared_kernels(),
    );
}
