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

struct Training<'g> {
    inputs: [Value<'g>; 2],
    loss: Value<'g>,
}

fn training<'g>(graph: &Graph<'g>, rows: u32, bound: Option<u32>) -> Training<'g> {
    let (observations, targets) = match bound {
        Some(bound) => {
            let batch = graph.free(bound);
            let shape = Shape::matrix(bound, WIDTH).freed(&[(2, batch)]);
            (
                graph.input(shape, Element::Single),
                graph.input(shape, Element::Single),
            )
        }
        None => (
            graph.input(Shape::matrix(rows, WIDTH), Element::Single),
            graph.input(Shape::matrix(rows, WIDTH), Element::Single),
        ),
    };
    let mut carried = observations;
    let mut parameters = Vec::new();
    for layer in 0..LAYERS {
        let weight = graph.named_parameter(
            &format!("w{layer}"),
            Shape::matrix(WIDTH, WIDTH),
            Init::Uniform {
                low: -0.02,
                high: 0.02,
            },
            Element::Single,
        );
        let bias = graph.named_parameter(
            &format!("b{layer}"),
            Shape::matrix(1, WIDTH),
            Init::Zero,
            Element::Single,
        );
        parameters.push(weight);
        parameters.push(bias);
        carried = graph.relu(graph.add(graph.matmul(carried, weight), bias));
    }
    let difference = graph.sub(carried, targets);
    let loss = graph.sum(graph.mul(difference, difference));
    let gradients = graph.backward(loss);
    let rate = graph.fill(Shape::scalar(), -0.005);
    for parameter in parameters {
        graph.add_into(parameter, graph.mul(gradients.of(parameter), rate));
    }
    graph.retain(loss);
    Training {
        inputs: [observations, targets],
        loss,
    }
}

#[test]
fn a_bound_program_gates_the_waves_of_the_shape_it_binds() {
    let runtime = open();
    let bound = 32u32;
    let graph = Graph::new();
    let family = training(&graph, bound, Some(bound));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let probes = observations(bound);
    let targets = observations(bound);
    runtime.bind(&program, &[bound]);
    runtime.write(&program, family.inputs[0], &probes);
    runtime.write(&program, family.inputs[1], &targets);
    runtime.run(&program);
    let loss = runtime.read(&program, family.loss);
    let waves = program.wave_count();

    let graph = Graph::new();
    let fixed = training(&graph, bound, None);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, fixed.inputs[0], &probes);
    runtime.write(&program, fixed.inputs[1], &targets);
    runtime.run(&program);
    assert_close(&runtime.read(&program, fixed.loss), &loss, 1e-4);
    assert_eq!(
        program.wave_count(),
        waves,
        "a free program bound to the {bound} rows a static program declares gates {} waves where that program gates {waves}",
        program.wave_count(),
    );

    let graph = Graph::new();
    training(&graph, bound, Some(bound));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let tasks = program.task_count();
    for rows in [bound, bound / 4, 1, 0] {
        runtime.bind(&program, &[rows]);
        assert!(
            program.task_count() <= tasks,
            "a binding of {rows} rows walks {} tasks where the bound of {bound} walks {tasks}",
            program.task_count(),
        );
        assert!(
            program.wave_count() <= waves,
            "a binding of {rows} rows gates {} waves where the bound of {bound} gates {waves}",
            program.wave_count(),
        );
    }
    runtime.bind(&program, &[1]);
    assert!(
        program.task_count() < tasks,
        "a binding of one row walks every one of the {tasks} tasks the bound of {bound} rows walks",
    );
}

#[test]
fn a_program_walks_every_shape_of_a_family_once() {
    let runtime = open();
    let graph = Graph::new();
    let family = model(&graph, BOUND, Some(BOUND));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let derived = program.derived_encodings();
    for _ in 0..3 {
        step(&runtime, &program, &family, 64);
        assert_eq!(
            program.derived_encodings(),
            derived + 1,
            "a program plans a shape once, and every later binding of it walks the encoding the plan remembers",
        );
    }
    step(&runtime, &program, &family, 16);
    assert_eq!(program.derived_encodings(), derived + 2);
    for _ in 0..3 {
        step(&runtime, &program, &family, 64);
        step(&runtime, &program, &family, 16);
        assert_eq!(
            program.derived_encodings(),
            derived + 2,
            "a program that alternates two shapes derives neither of them twice",
        );
    }
    assert_eq!(
        program.remembered_encodings(),
        2,
        "a program remembers the two shapes it has walked",
    );
    assert!(
        program.remembered_bytes() > 0,
        "a program that walked two shapes holds no encoding of them",
    );
}
