use crate::measure;
use neura::{
    AdamW, Conv2d, Element, Graph, HeadShape, Init, Mlp, MultiHeadAttention, Runtime, Shape,
    Window, mse_loss,
};

const NARROW: [u32; 4] = [8, 32, 32, 8];
const DEEP: [u32; 12] = [8, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 8];
const SOLO: u32 = 8;
const CROWD: u32 = 4096;
const SWEEP: u32 = 64;
const CHANNELS: u32 = 8;
const FEATURES: u32 = 16;
const IMAGES: u32 = 32;
const EDGE: u32 = 28;
const HEADS: u32 = 8;
const KEY_HEADS: u32 = 2;
const WIDTH: u32 = 128;
const STREAM: u32 = 64;

pub(crate) fn section(runtime: &Runtime) {
    let rows = [
        dense(runtime, &NARROW, SOLO),
        dense(runtime, &NARROW, CROWD),
        act(runtime, &NARROW, CROWD),
        dense(runtime, &DEEP, SWEEP),
        convolved(runtime),
        attended(runtime),
    ];
    measure::report("training", &rows);
}

fn dense(runtime: &Runtime, widths: &[u32], samples: u32) -> measure::Row {
    let graph = Graph::new();
    let model = Mlp::new(
        &graph,
        "model",
        widths,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    let observations = graph.input(Shape::matrix(samples, widths[0]), Element::Single);
    let outputs = *widths.last().expect("a width");
    let targets = graph.input(Shape::matrix(samples, outputs), Element::Single);
    let loss = mse_loss(&graph, model.forward(&graph, observations), targets);
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", 0.005, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&graph, &model.parameters());
    optimizer.step(&graph, &gradients);
    let held = vec![0.25; (samples * widths[0]) as usize];
    let wanted = vec![0.5; (samples * outputs) as usize];
    let program = measure::program(runtime, &graph, &[observations, targets], &[&held, &wanted]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, program.wave_count(), "wave");
    measure::row(
        format!("a step of {} layers at batch {samples}", widths.len() - 1),
        &program,
        timing,
        rate,
    )
}

fn act(runtime: &Runtime, widths: &[u32], agents: u32) -> measure::Row {
    let graph = Graph::new();
    let model = Mlp::new(
        &graph,
        "model",
        widths,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    let observations = graph.input(Shape::matrix(agents, widths[0]), Element::Single);
    let actions = graph.argmax(model.forward(&graph, observations));
    graph.retain(actions);
    let held = vec![0.25; (agents * widths[0]) as usize];
    let program = measure::program(runtime, &graph, &[observations], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, program.wave_count(), "wave");
    measure::row(
        format!("{agents} agents acting through {} layers", widths.len() - 1),
        &program,
        timing,
        rate,
    )
}

fn convolved(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let window = Window::new([3, 3], [1, 1], [0, 0], [1, 1]);
    let first = Conv2d::new(
        &graph,
        "first",
        [CHANNELS, FEATURES],
        1,
        window,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    let second = Conv2d::new(
        &graph,
        "second",
        [FEATURES, FEATURES],
        1,
        window,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    let observations = graph.input(Shape::of([IMAGES, CHANNELS, EDGE, EDGE]), Element::Single);
    let held = first.forward(&graph, observations);
    let hidden = second.forward(&graph, held);
    let edge = EDGE - 4;
    let targets = graph.input(Shape::of([IMAGES, FEATURES, edge, edge]), Element::Single);
    let loss = mse_loss(&graph, hidden, targets);
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", 0.005, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&graph, &first.parameters());
    optimizer.track_all(&graph, &second.parameters());
    optimizer.step(&graph, &gradients);
    let wanted_observations = vec![0.25; (IMAGES * CHANNELS * EDGE * EDGE) as usize];
    let wanted_targets = vec![0.5; (IMAGES * FEATURES * edge * edge) as usize];
    let program = measure::program(
        runtime,
        &graph,
        &[observations, targets],
        &[&wanted_observations, &wanted_targets],
    );
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, program.wave_count(), "wave");
    measure::row(
        format!("a step of 2 convolution layers at batch {IMAGES}"),
        &program,
        timing,
        rate,
    )
}

fn attended(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let origin = graph.fill(Shape::scalar(), 0.0);
    let attention = MultiHeadAttention::new(
        &graph,
        "attention",
        HeadShape::new(HEADS, KEY_HEADS, WIDTH),
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
        neura::AttentionOptions {
            scale: None,
            causal: true,
            origin: Some(origin),
            segments: None,
            reach: None,
            query_segments: None,
            bias: None,
        },
    );
    let observations = graph.input(Shape::of([1, 1, STREAM, WIDTH]), Element::Single);
    let targets = graph.input(Shape::of([HEADS, 1, STREAM, WIDTH]), Element::Single);
    let loss = mse_loss(&graph, attention.forward(&graph, observations), targets);
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", 0.005, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&graph, &attention.parameters());
    optimizer.step(&graph, &gradients);
    let held = vec![0.25; (STREAM * WIDTH) as usize];
    let wanted = vec![0.5; (HEADS * STREAM * WIDTH) as usize];
    let program = measure::program(runtime, &graph, &[observations, targets], &[&held, &wanted]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::each(timing.step, program.wave_count(), "wave");
    measure::row(
        format!("a step of attention at batch {STREAM}"),
        &program,
        timing,
        rate,
    )
}
