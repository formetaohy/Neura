use neura::{
    Adam, Graph, Init, MatmulTile, Mlp, Precision, Profile, Program, Runtime, RuntimeRequest,
    Shape, mse_loss,
};
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
    profile: Profile,
    tiles: Vec<(MatmulTile, u32)>,
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

fn tile_name(tile: MatmulTile) -> String {
    format!(
        "{}x{}x{}/{}",
        tile.rows(),
        tile.columns(),
        tile.depth(),
        tile.threads(),
    )
}

fn tile_list(tiles: &[(MatmulTile, u32)]) -> String {
    tiles
        .iter()
        .map(|(tile, count)| format!("{}x{}", tile_name(*tile), count))
        .collect::<Vec<_>>()
        .join(" ")
}

fn measured(label: String, program: &Program, timing: Timing) -> Measured {
    Measured {
        label,
        profile: program.profile(),
        tiles: program.matmul_geometries(),
        tasks: program.task_count(),
        waves: program.wave_count(),
        work: program.work(),
        timing,
    }
}

fn wide(runtime: &Runtime, elements: u32) -> Measured {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(elements));
    graph.relu(graph.mul(data, data));
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &vec![0.5; elements as usize]);
    runtime.run(&program);
    let timing = time(runtime, &program, 64);
    measured(
        format!("one rectifier over {elements} elements"),
        &program,
        timing,
    )
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
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        observations,
        &vec![0.25; (samples * widths[0]) as usize],
    );
    runtime.write(&program, targets, &vec![0.5; (samples * outputs) as usize]);
    runtime.run(&program);
    let timing = time(runtime, &program, 64);
    measured(
        format!("a step of {} layers at batch {samples}", widths.len() - 1),
        &program,
        timing,
    )
}

fn act(runtime: &Runtime, widths: &[u32], agents: u32) -> Measured {
    let graph = Graph::new();
    let model = Mlp::new(
        &graph,
        widths,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
    );
    let observations = graph.input(Shape::matrix(agents, widths[0]));
    let actions = graph.argmax(model.forward(&graph, observations));
    graph.retain(actions);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        observations,
        &vec![0.25; (agents * widths[0]) as usize],
    );
    runtime.run(&program);
    let timing = time(runtime, &program, 64);
    measured(
        format!("{agents} agents acting through {} layers", widths.len() - 1),
        &program,
        timing,
    )
}

fn products(runtime: &Runtime, shapes: &[(u32, u32, u32)]) -> Measured {
    let graph = Graph::new();
    for (rows, depth, columns) in shapes {
        let left = graph.parameter(Shape::matrix(*rows, *depth), Init::Zero);
        let right = graph.parameter(Shape::matrix(*depth, *columns), Init::Zero);
        graph.retain(graph.matmul(left, right));
    }
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    let timing = time(runtime, &program, 16);
    let label = shapes
        .iter()
        .map(|(rows, depth, columns)| format!("{rows}x{depth}x{columns}"))
        .collect::<Vec<_>>()
        .join(" + ");
    measured(format!("products {label}"), &program, timing)
}

fn profile_name(profile: Profile) -> String {
    format!("{} threads", profile.workgroup())
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
        act(&runtime, &narrow, 4096),
        step(&runtime, &deep, 64),
        products(&runtime, &[(96, 256, 96), (1024, 1024, 1024)]),
    ];
    println!(
        "{:<40} {:>10} {:>22} {:>7} {:>7} {:>11} {:>10} {:>9} {:>13}",
        "graph",
        "workgroup",
        "matmul tiles",
        "tasks",
        "waves",
        "work",
        "submit us",
        "step us",
        "per dispatch us",
    );
    for entry in &measured {
        println!(
            "{:<40} {:>10} {:>22} {:>7} {:>7} {:>11} {:>10.1} {:>9.1} {:>13.2}",
            entry.label,
            profile_name(entry.profile),
            tile_list(&entry.tiles),
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
        "{:<40} {:>10} {:>22} {:>9} {:>13}",
        "profile search", "workgroup", "matmul tiles", "step us", "TFLOP/s",
    );
    for profile in runtime.profiles() {
        let graph = Graph::new();
        let left = graph.parameter(Shape::matrix(PRODUCT_ROWS, PRODUCT_DEPTH), Init::Zero);
        let right = graph.parameter(Shape::matrix(PRODUCT_DEPTH, PRODUCT_COLUMNS), Init::Zero);
        graph.retain(graph.matmul(left, right));
        let weights = runtime.weights(&graph, Precision::Single);
        let program = runtime.compile_with(&graph, &weights, profile);
        runtime.run(&program);
        let timing = time(&runtime, &program, 16);
        let flops =
            2.0 * f64::from(PRODUCT_ROWS) * f64::from(PRODUCT_DEPTH) * f64::from(PRODUCT_COLUMNS);
        println!(
            "{:<40} {:>10} {:>22} {:>9.1} {:>13.1}",
            format!("a {PRODUCT_ROWS}x{PRODUCT_DEPTH}x{PRODUCT_COLUMNS} product"),
            profile_name(profile),
            tile_list(&program.matmul_geometries()),
            timing.step,
            flops / (timing.step * 1e-6) / 1e12,
        );
    }
    let search = Graph::new();
    let left = search.parameter(Shape::matrix(PRODUCT_ROWS, PRODUCT_DEPTH), Init::Zero);
    let right = search.parameter(Shape::matrix(PRODUCT_DEPTH, PRODUCT_COLUMNS), Init::Zero);
    search.retain(search.matmul(left, right));
    let weights = runtime.weights(&search, Precision::Single);
    let tuned = runtime.tune(&search, &weights);
    println!(
        "the device measured every profile it offers and kept {}",
        profile_name(tuned.profile()),
    );
    println!(
        "{} device programs served every shape above",
        runtime.declared_kernels(),
    );
}
