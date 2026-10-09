use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_plan::{DEFAULT_ENCODING_BYTES, Plan};
use neura_runtime::{Budget, Profile};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const WIDTH: u32 = 16;
const SAMPLES: u32 = 64;
const LAYERS: usize = 8;
const BLOCK: usize = 2;

fn narrow() -> Profile {
    Profile::derive(Budget::BASELINE, None)[0]
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn data(elements: u32, scale: f32) -> Vec<f32> {
    (0..elements)
        .map(|element| (element as f32 * scale).sin())
        .collect()
}

fn layer<'g>(graph: &Graph<'g>, value: Value<'g>, parameters: &mut Vec<Value<'g>>) -> Value<'g> {
    let weight = graph.parameter(
        Shape::matrix(WIDTH, WIDTH),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let bias = graph.parameter(Shape::vector(WIDTH), Init::Zero, Element::Single);
    parameters.push(weight);
    parameters.push(bias);
    graph.relu(graph.add(graph.matmul(value, weight), bias))
}

fn stack<'g>(graph: &Graph<'g>, recomputed: bool) -> (Value<'g>, Value<'g>, Vec<Value<'g>>) {
    let input = graph.input(Shape::matrix(SAMPLES, WIDTH), Element::Single);
    let mut parameters = Vec::new();
    let mut value = input;
    let mut layers = 0;
    while layers < LAYERS {
        value = if recomputed {
            graph.recompute(|graph| {
                let mut inside = value;
                for _ in 0..BLOCK {
                    inside = layer(graph, inside, &mut parameters);
                }
                inside
            })
        } else {
            let mut inside = value;
            for _ in 0..BLOCK {
                inside = layer(graph, inside, &mut parameters);
            }
            inside
        };
        layers += BLOCK;
    }
    (input, value, parameters)
}

fn trained<'g>(
    graph: &Graph<'g>,
    recomputed: bool,
) -> (Value<'g>, Value<'g>, Value<'g>, Vec<Value<'g>>) {
    let (input, out, parameters) = stack(graph, recomputed);
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let weight_gradients = parameters
        .iter()
        .map(|parameter| gradients.of(*parameter))
        .collect::<Vec<_>>();
    for gradient in &weight_gradients {
        graph.retain(*gradient);
    }
    graph.retain(loss);
    graph.retain(out);
    (input, loss, out, weight_gradients)
}

#[test]
fn a_recomputed_stack_holds_less_of_the_forward_it_runs() {
    let plain = Graph::new();
    let (_, out, _) = stack(&plain, false);
    plain.backward(plain.sum(out));
    let recomputed = Graph::new();
    let (_, out, _) = stack(&recomputed, true);
    recomputed.backward(recomputed.sum(out));

    let plain = Plan::of(&plain, 256, narrow(), DEFAULT_ENCODING_BYTES);
    let recomputed = Plan::of(&recomputed, 256, narrow(), DEFAULT_ENCODING_BYTES);
    assert!(
        recomputed.arena_bytes() * 5 < plain.arena_bytes() * 4,
        "a stack of {LAYERS} layers keeps {} bytes of arena where {BLOCK} layer regions recompute it into {}",
        plain.arena_bytes(),
        recomputed.arena_bytes(),
    );
}

#[test]
fn a_recomputed_stack_matches_the_gradients_it_replaced() {
    let runtime = open();
    let plain_graph = Graph::new();
    let (plain_input, plain_loss, plain_out, plain_gradients) = trained(&plain_graph, false);
    let plain_weights = runtime.weights(&plain_graph);
    let plain_program = runtime.compile(&plain_graph, &plain_weights);

    let graph = Graph::new();
    let (input, loss, out, gradients_of) = trained(&graph, true);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);

    assert!(
        program.arena_bytes() * 5 < plain_program.arena_bytes() * 4,
        "a recomputed step keeps {} bytes of activations where the plain one keeps {}",
        program.arena_bytes(),
        plain_program.arena_bytes(),
    );
    let values = data(SAMPLES * WIDTH, 0.031);
    runtime.write(&program, input, &values);
    runtime.write(&plain_program, plain_input, &values);
    runtime.run(&program);
    runtime.run(&plain_program);
    assert_close(
        &runtime.read(&program, out),
        &runtime.read(&plain_program, plain_out),
        1e-5,
    );
    assert_close(
        &runtime.read(&program, loss),
        &runtime.read(&plain_program, plain_loss),
        1e-5,
    );
    let observed = runtime.read_many(&program, &gradients_of);
    let expected = runtime.read_many(&plain_program, &plain_gradients);
    for (observed, expected) in observed.iter().zip(&expected) {
        assert_close(observed, expected, 1e-5);
    }
}

#[test]
fn a_recomputed_block_reaches_the_readers_of_its_view() {
    let runtime = open();
    let plain = Graph::new();
    let (plain_input, plain_loss, plain_gradient) = viewed(&plain, false);
    let plain_weights = runtime.weights(&plain);
    let plain_program = runtime.compile(&plain, &plain_weights);

    let recomputed = Graph::new();
    let (input, loss, gradient) = viewed(&recomputed, true);
    let weights = runtime.weights(&recomputed);
    let program = runtime.compile(&recomputed, &weights);
    assert!(
        program.arena_bytes() * 5 < plain_program.arena_bytes() * 4,
        "a recomputed block keeps {} bytes where the block it replaces keeps {}",
        program.arena_bytes(),
        plain_program.arena_bytes(),
    );

    let values = data(SAMPLES * WIDTH, 0.017);
    runtime.write(&program, input, &values);
    runtime.write(&plain_program, plain_input, &values);
    runtime.run(&program);
    runtime.run(&plain_program);
    assert_close(
        &runtime.read(&program, loss),
        &runtime.read(&plain_program, plain_loss),
        1e-5,
    );
    assert_close(
        &runtime.read(&program, gradient),
        &runtime.read(&plain_program, plain_gradient),
        1e-5,
    );
}

fn viewed<'g>(graph: &Graph<'g>, recomputed: bool) -> (Value<'g>, Value<'g>, Value<'g>) {
    let (input, out, parameters) = stack(graph, recomputed);
    let flat = graph.reshape(out, Shape::vector(SAMPLES * WIDTH));
    let loss = graph.sum(graph.mul(flat, flat));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(parameters[0]);
    graph.retain(gradient);
    graph.retain(loss);
    (input, loss, gradient)
}

#[test]
fn a_recompute_region_refuses_an_update_in_place() {
    assert!(refuses(|| {
        let graph = Graph::new();
        let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
        let target = graph.resident(Shape::vector(4), Element::Single);
        graph.recompute(|graph| {
            graph.copy_into(target, weight);
            target
        });
    }));
}

#[test]
fn a_recompute_region_refuses_to_open_inside_another() {
    assert!(refuses(|| {
        let graph = Graph::new();
        let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
        graph.recompute(|graph| {
            let inner = graph.recompute(|graph| graph.relu(weight));
            graph.relu(inner)
        });
    }));
}

#[test]
fn a_recompute_region_refuses_a_tensor_of_its_own_author() {
    assert!(refuses(|| {
        let graph = Graph::new();
        let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
        let outside = graph.relu(weight);
        graph.recompute(|graph| {
            graph.relu(outside);
            outside
        });
    }));
}
