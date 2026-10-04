use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const BOUND: u32 = 8;
const WIDTH: u32 = 4;

fn data(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..count)
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn live_probe(bound: u32, live: u32) -> Vec<f32> {
    (0..bound)
        .map(|row| if row < live { 1.0 } else { 0.0 })
        .collect()
}

fn weights() -> Init {
    Init::Uniform {
        low: -0.25,
        high: 0.25,
    }
}

struct Extent {
    probe: Value<'static>,
    count: Value<'static>,
    tokens: Value<'static>,
    gate: Value<'static>,
    loss: Value<'static>,
    gradient: Value<'static>,
    gate_gradient: Value<'static>,
}

fn extent(
    graph: &Graph<'static>,
    count: Value<'static>,
    probe: Value<'static>,
    bound: u32,
) -> Extent {
    let gate = graph.gradient_input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let tokens = graph.gradient_input(Shape::of([1, 1, bound, WIDTH]), Element::Single);
    let rows = graph.trim(tokens, 2, count);
    let loss = graph.sum(graph.mul(rows, gate));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(tokens);
    let gate_gradient = gradients.of(gate);
    graph.retain(gradient);
    graph.retain(gate_gradient);
    Extent {
        probe,
        count,
        tokens,
        gate,
        loss,
        gradient,
        gate_gradient,
    }
}

fn device_counted(graph: &Graph<'static>, bound: u32) -> Extent {
    let probe = graph.input(Shape::of([1, 1, bound, 1]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    extent(graph, count, probe, bound)
}

fn host_counted(graph: &Graph<'static>, bound: u32) -> Extent {
    let count = graph.input(Shape::scalar(), Element::Single);
    extent(graph, count, count, bound)
}

fn reference(tokens: &[f32], gate: &[f32], live: u32) -> (f32, Vec<f32>, Vec<f32>) {
    let mut loss = 0.0;
    let mut gradient = vec![0.0f32; (BOUND * WIDTH) as usize];
    let mut gate_gradient = vec![0.0f32; WIDTH as usize];
    for row in 0..live {
        for column in 0..WIDTH {
            let at = (row * WIDTH + column) as usize;
            loss += tokens[at] * gate[column as usize];
            gradient[at] = gate[column as usize];
            gate_gradient[column as usize] += tokens[at];
        }
    }
    (loss, gradient, gate_gradient)
}

fn check(
    runtime: &Runtime,
    program: &Program<'_>,
    model: &Extent,
    tokens: &[f32],
    gate: &[f32],
    live: u32,
) {
    runtime.run(program);
    let (loss, gradient, gate_gradient) = reference(tokens, gate, live);
    assert_close(&runtime.read(program, model.loss), &[loss], 1e-3);
    assert_close(&runtime.read(program, model.gradient), &gradient, 0.0);
    assert_close(
        &runtime.read(program, model.gate_gradient),
        &gate_gradient,
        1e-3,
    );
}

#[test]
fn a_trimmed_tensor_lands_its_product_gradients_in_the_walk_that_owns_it() {
    let runtime = open();
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, BOUND, 1]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let tokens = graph.gradient_input(Shape::of([1, 1, BOUND, WIDTH]), Element::Single);
    let weight = graph.parameter(Shape::matrix(WIDTH, WIDTH), weights(), Element::Single);
    let rows = graph.trim(tokens, 2, count);
    let loss = graph.sum(graph.matmul(rows, weight));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(tokens);
    let weight_gradient = gradients.of(weight);
    graph.retain(gradient);
    graph.retain(weight_gradient);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let tokens_data = data(BOUND * WIDTH, 47);
    let weight_data = data(WIDTH * WIDTH, 53);
    runtime.write(&program, tokens, &tokens_data);
    runtime.write(&program, weight, &weight_data);
    for live in [BOUND, 4, 1] {
        runtime.write(&program, probe, &live_probe(BOUND, live));
        runtime.run(&program);
        let reference = Graph::new();
        let fixed = reference.gradient_input(Shape::matrix(live, WIDTH), Element::Single);
        let shared = reference.parameter(Shape::matrix(WIDTH, WIDTH), weights(), Element::Single);
        let fixed_loss = reference.sum(reference.matmul(fixed, shared));
        let fixed_gradients = reference.backward(fixed_loss);
        reference.retain(fixed_gradients.of(fixed));
        reference.retain(fixed_gradients.of(shared));
        let fixed_store = runtime.weights(&reference);
        let fixed_program = runtime.compile(&reference, &fixed_store);
        runtime.write(
            &fixed_program,
            fixed,
            &tokens_data[..(live * WIDTH) as usize],
        );
        runtime.write(&fixed_program, shared, &weight_data);
        runtime.run(&fixed_program);
        let mut expected = vec![0.0f32; (BOUND * WIDTH) as usize];
        expected[..(live * WIDTH) as usize]
            .copy_from_slice(&runtime.read(&fixed_program, fixed_gradients.of(fixed)));
        assert_close(&runtime.read(&program, gradient), &expected, 1e-4);
        assert_close(
            &runtime.read(&program, weight_gradient),
            &runtime.read(&fixed_program, fixed_gradients.of(shared)),
            1e-4,
        );
    }
}

#[test]
fn a_trimmed_parameter_descends_only_the_rows_the_count_leaves() {
    let runtime = open();
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, BOUND, 1]), Element::Single);
    let gate = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let weight = graph.parameter(
        Shape::of([1, 1, BOUND, WIDTH]),
        Init::Uniform {
            low: -0.25,
            high: 0.25,
        },
        Element::Single,
    );
    let rows = graph.trim(weight, 2, count);
    let loss = graph.sum(graph.mul(rows, gate));
    let gradients = graph.backward(loss);
    let rate = graph.fill(Shape::scalar(), -0.5);
    graph.add_into(weight, graph.mul(gradients.of(weight), rate));
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let gate_data = data(WIDTH, 41);
    let initial = data(BOUND * WIDTH, 43);
    runtime.write(&program, gate, &gate_data);
    for live in [BOUND, 3, 0] {
        runtime.write(&program, weight, &initial);
        runtime.write(&program, probe, &live_probe(BOUND, live));
        runtime.run(&program);
        let mut expected = initial.clone();
        for row in 0..live {
            for column in 0..WIDTH {
                let at = (row * WIDTH + column) as usize;
                expected[at] -= 0.5 * gate_data[column as usize];
            }
        }
        assert_close(&runtime.read(&program, weight), &expected, 1e-4);
    }
}

#[test]
fn a_trimmed_half_tensor_lands_its_gradient_through_an_image() {
    let runtime = open();
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, BOUND, 1]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let tokens = graph.gradient_input(Shape::of([1, 1, BOUND, WIDTH]), Element::Half);
    let rows = graph.trim(tokens, 2, count);
    let loss = graph.sum(rows);
    let gradients = graph.backward(loss);
    let gradient = gradients.of(tokens);
    graph.retain(gradient);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.write(&program, tokens, &data(BOUND * WIDTH, 71));
    for live in [BOUND, 3, 0] {
        runtime.write(&program, probe, &live_probe(BOUND, live));
        runtime.run(&program);
        let expected = (0..BOUND)
            .flat_map(|row| std::iter::repeat_n(if row < live { 1.0 } else { 0.0 }, WIDTH as usize))
            .collect::<Vec<f32>>();
        assert_close(&runtime.read(&program, gradient), &expected, 0.0);
    }
}

#[test]
fn a_trimmed_walk_beside_a_full_walk_lands_one_gradient() {
    let runtime = open();
    let graph = Graph::new();
    let probe = graph.input(Shape::of([1, 1, BOUND, 1]), Element::Single);
    let count = graph.sum_axis(probe, 2);
    let tokens = graph.gradient_input(Shape::of([1, 1, BOUND, WIDTH]), Element::Single);
    let rows = graph.trim(tokens, 2, count);
    let loss = graph.add(graph.sum(rows), graph.sum(tokens));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(tokens);
    graph.retain(gradient);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    for live in [BOUND, 3, 0] {
        runtime.write(&program, probe, &live_probe(BOUND, live));
        runtime.run(&program);
        let expected = (0..BOUND)
            .flat_map(|row| std::iter::repeat_n(if row < live { 2.0 } else { 1.0 }, WIDTH as usize))
            .collect::<Vec<f32>>();
        assert_close(&runtime.read(&program, gradient), &expected, 0.0);
    }
}

#[test]
fn a_recomputed_prefix_lands_its_gradient_in_the_storage_it_copies() {
    let runtime = open();
    let tokens = data(BOUND * WIDTH, 59);
    let other = data(BOUND * WIDTH, 61);
    let gate_data = data(WIDTH, 67);
    let mut measured = Vec::new();
    for recomputed in [false, true] {
        let graph = Graph::new();
        let probe = graph.input(Shape::of([1, 1, BOUND, 1]), Element::Single);
        let count = graph.sum_axis(probe, 2);
        let left = graph.gradient_input(Shape::of([1, 1, BOUND, WIDTH]), Element::Single);
        let right = graph.gradient_input(Shape::of([1, 1, BOUND, WIDTH]), Element::Single);
        let gate = graph.gradient_input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
        let loss = if recomputed {
            let product = graph.recompute(|graph| {
                let dense = graph.mul(left, right);
                graph.mul(graph.trim(dense, 2, count), gate)
            });
            graph.sum(product)
        } else {
            let dense = graph.mul(left, right);
            graph.sum(graph.mul(graph.trim(dense, 2, count), gate))
        };
        let slopes = graph.backward(loss);
        graph.retain(slopes.of(left));
        graph.retain(slopes.of(right));
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        runtime.write(&program, left, &tokens);
        runtime.write(&program, right, &other);
        runtime.write(&program, gate, &gate_data);
        let mut round = Vec::new();
        for live in [BOUND, 2, 0] {
            runtime.write(&program, probe, &live_probe(BOUND, live));
            runtime.run(&program);
            round.push((
                runtime.read(&program, slopes.of(left)),
                runtime.read(&program, slopes.of(right)),
            ));
        }
        measured.push(round);
    }
    for (kept, remade) in measured[0].iter().zip(&measured[1]) {
        assert_close(&kept.0, &remade.0, 1e-5);
        assert_close(&kept.1, &remade.1, 1e-5);
    }
}

#[test]
fn a_device_count_carries_the_gradient_of_a_trimmed_tensor() {
    let runtime = open();
    let graph = Graph::new();
    let model = device_counted(&graph, BOUND);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let tokens = data(BOUND * WIDTH, 11);
    let gate = data(WIDTH, 29);
    runtime.write(&program, model.tokens, &tokens);
    runtime.write(&program, model.gate, &gate);
    for live in [BOUND, 5, 1, 0] {
        runtime.write(&program, model.probe, &live_probe(BOUND, live));
        check(&runtime, &program, &model, &tokens, &gate, live);
    }
}

#[test]
fn a_count_the_host_writes_carries_the_gradient_of_a_trimmed_tensor() {
    let runtime = open();
    let graph = Graph::new();
    let model = host_counted(&graph, BOUND);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    let tokens = data(BOUND * WIDTH, 13);
    let gate = data(WIDTH, 31);
    runtime.write(&program, model.tokens, &tokens);
    runtime.write(&program, model.gate, &gate);
    for live in [BOUND, 6, 2, 0] {
        runtime.write(&program, model.count, &[live as f32]);
        check(&runtime, &program, &model, &tokens, &gate, live);
    }
}
