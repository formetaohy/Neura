use neura::{Adam, Graph, Init, Mlp, Runtime, RuntimeRequest, Shape, mse_loss};
use std::time::Instant;

fn session(samples: u32) -> (Vec<f32>, Vec<f32>) {
    let mut observations = Vec::with_capacity(samples as usize * 4);
    let mut targets = Vec::with_capacity(samples as usize * 2);
    for sample in 0..samples {
        let phase = sample as f32 * 0.031;
        let x = phase.sin();
        let y = (phase * 1.7).cos();
        let speed = (phase * 0.3).sin() * 0.5 + 0.5;
        let throttle = (phase * 2.3).cos() * 0.5 + 0.5;
        observations.extend([x, y, speed, throttle]);
        targets.extend([(x - y) * 0.5, (speed - throttle) * 0.5]);
    }
    (observations, targets)
}

fn main() {
    let runtime =
        pollster::block_on(Runtime::open(RuntimeRequest::default())).expect("a device to train on");
    let graph = Graph::new();
    let model = Mlp::new(
        &graph,
        &[4, 32, 32, 2],
        Init::Uniform {
            low: -0.25,
            high: 0.25,
        },
    );
    let samples = 256;
    let observations = graph.input(Shape::matrix(samples, 4));
    let targets = graph.input(Shape::matrix(samples, 2));
    let prediction = model.forward(&graph, observations);
    graph.retain(prediction);
    let loss = mse_loss(&graph, prediction, targets);
    let gradients = graph.backward(loss);
    let mut optimizer = Adam::new(&graph, 0.005, 0.9, 0.999, 1e-8);
    optimizer.track_all(&graph, &model.parameters());
    optimizer.step(&graph, &gradients);
    let program = runtime.compile(&graph);
    let (observation_data, target_data) = session(samples);
    runtime.write(&program, observations, &observation_data);
    runtime.write(&program, targets, &target_data);
    for step in 0..400 {
        runtime.run(&program);
        if step % 100 == 0 || step == 399 {
            println!("step {step:>3}: loss {}", runtime.read(&program, loss)[0]);
        }
    }
    println!(
        "one step: {} tasks in {} waves, {} bytes of arena, {} device programs",
        program.task_count(),
        program.wave_count(),
        program.arena_bytes(),
        runtime.declared_kernels(),
    );
    for _ in 0..16 {
        runtime.run(&program);
    }
    let rounds = 128;
    let started = Instant::now();
    for _ in 0..rounds {
        runtime.run(&program);
    }
    let per_step = started.elapsed().as_secs_f64() * 1e6 / rounds as f64;
    println!("host side of a step: {per_step:.1} us over {rounds} steps");
    let produced = runtime
        .read(&program, prediction)
        .into_iter()
        .take(2)
        .collect::<Vec<_>>();
    let recorded = target_data.iter().take(2).copied().collect::<Vec<_>>();
    println!("two actions: {produced:?} where the recording says {recorded:?}");
}
