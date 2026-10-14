use crate::measure;
use neura::{Conv2d, Element, Graph, Init, Pool, Runtime, Shape, Window};

const CHANNELS: u32 = 8;
const OUTPUTS: u32 = 16;
const PLANES: u32 = 32;
const EDGE: u32 = 28;

pub(crate) fn section(runtime: &Runtime) {
    let rows = [
        convolved(runtime),
        pooled(runtime, Pool::Max),
        pooled(runtime, Pool::Mean),
    ];
    measure::report("vision paths", &rows);
}

fn convolved(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let window = Window::new([3, 3], [1, 1], [0, 0], [1, 1]);
    let layer = Conv2d::new(
        &graph,
        "layer",
        [CHANNELS, OUTPUTS],
        1,
        window,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    let image = graph.input(Shape::of([1, CHANNELS, EDGE, EDGE]), Element::Single);
    graph.retain(layer.forward(&graph, image));
    let held = vec![0.5; (CHANNELS * EDGE * EDGE) as usize];
    let program = measure::program(runtime, &graph, &[image], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let edge = EDGE - 2;
    let rate = measure::traffic(
        timing.step,
        u64::from(CHANNELS * EDGE * EDGE + OUTPUTS * edge * edge + OUTPUTS * CHANNELS * 9) * 4,
        "",
    );
    measure::row(
        format!("a convolution of {CHANNELS} into {OUTPUTS} channels over {EDGE}x{EDGE}"),
        &program,
        timing,
        rate,
    )
}

fn pooled(runtime: &Runtime, mode: Pool) -> measure::Row {
    let graph = Graph::new();
    let window = Window::new([2, 2], [2, 2], [0, 0], [1, 1]);
    let image = graph.input(Shape::of([1, PLANES, EDGE, EDGE]), Element::Single);
    graph.retain(graph.pool2d(image, window, mode));
    let held = vec![0.5; (PLANES * EDGE * EDGE) as usize];
    let program = measure::program(runtime, &graph, &[image], &[&held]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let edge = EDGE / 2;
    let rate = measure::traffic(
        timing.step,
        u64::from(PLANES * (EDGE * EDGE + edge * edge)) * 4,
        "",
    );
    measure::row(
        format!(
            "a {} pool over {PLANES} planes of {EDGE}x{EDGE}",
            mode.name()
        ),
        &program,
        timing,
        rate,
    )
}
