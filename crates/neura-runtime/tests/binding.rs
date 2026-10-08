use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{MemoryRequest, Program, Runtime, RuntimeRequest};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const WIDTH: u32 = 64;
const LAYERS: u32 = 3;
const BOUND: u32 = 4096;
const HEAP_BYTES: u64 = 2 << 20;

fn bounded(heap_bytes: u64) -> Runtime {
    Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes,
            ..Default::default()
        },
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

struct Model<'g> {
    input: Value<'g>,
    output: Value<'g>,
}

fn model<'g>(graph: &Graph<'g>, rows: u32, bound: Option<u32>) -> Model<'g> {
    let input = match bound {
        Some(bound) => graph.input(
            Shape::matrix(rows, WIDTH).freed(&[(2, graph.free(bound))]),
            Element::Single,
        ),
        None => graph.input(Shape::matrix(rows, WIDTH), Element::Single),
    };
    let mut value = input;
    for layer in 0..LAYERS {
        let weight = graph.named_parameter(
            &format!("w{layer}"),
            Shape::matrix(WIDTH, WIDTH),
            Init::Uniform {
                low: -0.2,
                high: 0.2,
            },
            Element::Single,
        );
        value = graph.relu(graph.matmul(value, weight));
    }
    graph.retain(value);
    Model {
        input,
        output: value,
    }
}

fn observations(rows: u32) -> Vec<f32> {
    (0..rows * WIDTH)
        .map(|at| ((at * 37) % 101) as f32 / 101.0 - 0.5)
        .collect()
}

fn fixed(runtime: &Runtime, rows: u32, data: &[f32]) -> Vec<f32> {
    let graph = Graph::new();
    let model = model(&graph, rows, None);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, model.input, data);
    runtime.run(&program);
    runtime.read(&program, model.output)
}

fn step(runtime: &Runtime, program: &Program, model: &Model<'_>, rows: u32) -> Vec<f32> {
    let data = observations(rows);
    runtime.bind(program, &[rows]);
    runtime.write(program, model.input, &data);
    runtime.run(program);
    let produced = runtime.read(program, model.output);
    assert_close(&produced, &fixed(runtime, rows, &data), 1e-4);
    produced
}

#[test]
fn a_bound_no_heap_can_hold_runs_the_binding_it_names() {
    let runtime = bounded(HEAP_BYTES);
    let graph = Graph::new();
    let family = model(&graph, BOUND, Some(BOUND));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert!(
        program.tensor_bytes() + weights.bytes() > HEAP_BYTES,
        "a bound of {BOUND} rows holds {} tensors beside {} weights where this heap holds {HEAP_BYTES}",
        program.tensor_bytes(),
        weights.bytes(),
    );
    for rows in [8u32, BOUND / 8, 512, 1] {
        runtime.bind(&program, &[rows]);
        assert!(
            program.arena_bytes() + weights.bytes() <= HEAP_BYTES,
            "a binding of {rows} rows holds {} bytes beside {} weights where this heap holds {HEAP_BYTES}",
            program.arena_bytes(),
            weights.bytes(),
        );
        step(&runtime, &program, &family, rows);
    }
}

#[test]
fn a_binding_grows_the_arena_a_later_binding_outgrows() {
    let runtime = open();
    let graph = Graph::new();
    let family = model(&graph, BOUND, Some(BOUND));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.bind(&program, &[1]);
    let narrow = program.arena_bytes();
    runtime.bind(&program, &[512]);
    let wide = program.arena_bytes();
    assert!(
        wide > narrow,
        "a binding of 512 rows holds {wide} bytes beside the {narrow} one row holds",
    );
    runtime.bind(&program, &[1]);
    assert_eq!(
        program.arena_bytes(),
        narrow,
        "a binding the tensors already hold walks the arena of its own lengths",
    );
    for rows in [BOUND / 8, 512, 1, BOUND / 8] {
        step(&runtime, &program, &family, rows);
    }
}

#[test]
fn a_binding_the_heap_cannot_hold_refuses() {
    let runtime = bounded(HEAP_BYTES);
    let graph = Graph::new();
    model(&graph, BOUND, Some(BOUND));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.bind(&program, &[8]);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.bind(&program, &[BOUND]);
    }));
    assert!(
        refused.is_err(),
        "a binding of {BOUND} rows holds {} tensors where this heap holds {HEAP_BYTES}",
        program.tensor_bytes(),
    );
}
