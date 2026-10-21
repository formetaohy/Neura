use neura_abi::Element;
use neura_graph::{DeltaInput, Graph, Shape, Value};
use neura_nn::DeltaScan;
use neura_runtime::{Runtime, RuntimeRequest};

const HEADS: u32 = 2;
const BATCH: u32 = 2;
const WIDTH: u32 = 3;
const ROWS: u32 = 6;
const CHUNK: u32 = 3;

fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32, what: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{what} element {index} came back as {actual} where {expected} was expected",
        );
    }
}

fn values() -> [Vec<f32>; 5] {
    let count = HEADS * BATCH * ROWS * WIDTH;
    let scalars = HEADS * BATCH * ROWS;
    let mut queries = random(count, 7);
    let mut keys = random(count, 11);
    for tensor in [&mut queries, &mut keys] {
        for row in tensor.chunks_mut(WIDTH as usize) {
            let length = row.iter().map(|value| value * value).sum::<f32>().sqrt();
            for value in row {
                *value /= length;
            }
        }
    }
    [
        queries,
        keys,
        random(count, 13),
        random(scalars, 17)
            .into_iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect(),
        random(scalars, 19)
            .into_iter()
            .map(|value| -value.abs())
            .collect(),
    ]
}

struct Walk {
    inputs: [Value; 6],
    out: Value,
    gradients: [Value; 6],
}

fn scanned(graph: &Graph) -> Walk {
    let query = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, WIDTH]), Element::Single);
    let key = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, WIDTH]), Element::Single);
    let value = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, WIDTH]), Element::Single);
    let beta = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, 1]), Element::Single);
    let log_decay = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, 1]), Element::Single);
    let state = graph.gradient_input(Shape::of([HEADS, BATCH, WIDTH, WIDTH]), Element::Single);
    let scan = DeltaScan::chunked(HEADS, WIDTH, BATCH, ROWS, CHUNK);
    let out = scan
        .forward(
            graph,
            state,
            DeltaInput {
                query,
                key,
                value,
                beta,
                log_decay,
            },
        )
        .attended;
    finished(graph, [query, key, value, beta, log_decay, state], out)
}

fn stepped(graph: &Graph) -> Walk {
    let query = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, WIDTH]), Element::Single);
    let key = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, WIDTH]), Element::Single);
    let value = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, WIDTH]), Element::Single);
    let beta = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, 1]), Element::Single);
    let log_decay = graph.gradient_input(Shape::of([HEADS, BATCH, ROWS, 1]), Element::Single);
    let state = graph.gradient_input(Shape::of([HEADS, BATCH, WIDTH, WIDTH]), Element::Single);
    let mut carried = state;
    let mut rows = Vec::with_capacity(ROWS as usize);
    for row in 0..ROWS {
        let step_query = graph.slice(query, 2, row, 1);
        let step_key = graph.slice(key, 2, row, 1);
        let step_value = graph.slice(value, 2, row, 1);
        let step_beta = graph.slice(beta, 2, row, 1);
        let step_decay = graph.slice(log_decay, 2, row, 1);
        let decayed = graph.mul(carried, graph.exp(step_decay));
        let memory = graph.matmul(step_key, decayed);
        let delta = graph.mul(graph.sub(step_value, memory), step_beta);
        let grown = graph.add(
            decayed,
            graph.matmul(graph.permute(step_key, [0, 1, 3, 2]), delta),
        );
        rows.push(graph.matmul(step_query, grown));
        carried = grown;
    }
    let out = graph.concat(&rows, 2);
    finished(graph, [query, key, value, beta, log_decay, state], out)
}

fn finished(graph: &Graph, inputs: [Value; 6], out: Value) -> Walk {
    graph.retain(out);
    let loss = graph.sum(out);
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let mut held = inputs.map(|held| gradients.of(held));
    for gradient in &mut held {
        graph.retain(*gradient);
    }
    Walk {
        inputs,
        out,
        gradients: held,
    }
}

fn run(runtime: &Runtime, graph: &Graph, walk: &Walk) -> (Vec<f32>, [Vec<f32>; 6]) {
    let weights = runtime.weights(graph);
    let program = runtime.compile(graph, &weights);
    for (held, values) in walk.inputs[..5].iter().zip(values()) {
        runtime.write(&program, *held, &values);
    }
    runtime.write(
        &program,
        walk.inputs[5],
        &vec![0.0; (HEADS * BATCH * WIDTH * WIDTH) as usize],
    );
    runtime.run(&program);
    let mut gradients = [
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
        Vec::new(),
    ];
    for (held, values) in walk.gradients.iter().zip(gradients.iter_mut()) {
        *values = runtime.read(&program, *held);
    }
    (runtime.read(&program, walk.out), gradients)
}

#[test]
fn a_chunked_scan_matches_the_recurrence_it_trains() {
    let runtime = Runtime::open(RuntimeRequest::default()).expect("a device that trains a scan");
    let chunked_graph = Graph::new();
    let chunked = scanned(&chunked_graph);
    let stepped_graph = Graph::new();
    let stepped = stepped(&stepped_graph);
    let (chunked_out, chunked_gradients) = run(&runtime, &chunked_graph, &chunked);
    let (stepped_out, stepped_gradients) = run(&runtime, &stepped_graph, &stepped);
    assert_close(&chunked_out, &stepped_out, 1e-4, "output");
    for (index, (chunked, stepped)) in chunked_gradients.iter().zip(&stepped_gradients).enumerate()
    {
        assert_close(chunked, stepped, 1e-4, &format!("gradient {index}"));
    }
}
