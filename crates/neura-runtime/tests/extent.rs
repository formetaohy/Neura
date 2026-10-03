use neura_abi::Element;
use neura_graph::{AttentionOptions, Free, Graph, Init, Pool, Shape, Value, Window};
use neura_runtime::{Program, Runtime};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

#[derive(Clone, Copy)]
enum Extent {
    Batch,
    Tokens,
}

struct Model<'g> {
    inputs: Vec<Value<'g>>,
    parameter: Value<'g>,
    output: Value<'g>,
    frees: Vec<(Extent, Free)>,
}

impl Model<'_> {
    fn binding(&self, batch: u32, tokens: u32) -> Vec<u32> {
        self.frees
            .iter()
            .map(|(extent, _)| match extent {
                Extent::Batch => batch,
                Extent::Tokens => tokens,
            })
            .collect()
    }

    fn elements(&self, shape: Shape, binding: &[u32]) -> u32 {
        let mut elements = 1u32;
        for axis in 0..4 {
            let dim = match shape.free(axis) {
                Some(slot) => {
                    let at = self
                        .frees
                        .iter()
                        .position(|(_, free)| free.slot() == slot)
                        .unwrap_or_else(|| panic!("free extent {slot} walks no binding"));
                    binding[at]
                }
                None => shape.dims()[axis as usize],
            };
            elements *= dim;
        }
        elements
    }
}

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

fn weights() -> Init {
    Init::Uniform {
        low: -0.25,
        high: 0.25,
    }
}

fn stream(
    graph: &Graph<'static>,
    batch: u32,
    tokens: u32,
    frees: Option<(Free, Free)>,
) -> Value<'static> {
    let shape = match frees {
        Some((batch_extent, token_extent)) => {
            Shape::of([1, batch, tokens, 8]).freed(&[(1, batch_extent), (2, token_extent)])
        }
        None => Shape::of([1, batch, tokens, 8]),
    };
    graph.input(shape, Element::Single)
}

fn attention_model(
    graph: &Graph<'static>,
    batch: u32,
    tokens: u32,
    dynamic: bool,
) -> Model<'static> {
    let batch_extent = graph.free(batch);
    let token_extent = graph.free(tokens);
    let frees = dynamic.then_some((batch_extent, token_extent));
    let input = stream(graph, batch, tokens, frees);
    let parameter = graph.parameter(Shape::of([2, 1, 8, 8]), weights(), Element::Single);
    let projected = graph.matmul(input, parameter);
    let attended = graph.attention(
        projected,
        projected,
        projected,
        AttentionOptions {
            scale: 0.25,
            causal: true,
            origin: None,
        },
    );
    let activated = graph.tanh(attended);
    let summed = graph.sum_rows(activated);
    let output = graph.mul(activated, summed);
    graph.retain(output);
    Model {
        inputs: vec![input],
        parameter,
        output,
        frees: frees
            .map(|(batch, tokens)| vec![(Extent::Batch, batch), (Extent::Tokens, tokens)])
            .unwrap_or_default(),
    }
}

fn reduction_model(
    graph: &Graph<'static>,
    batch: u32,
    tokens: u32,
    dynamic: bool,
) -> Model<'static> {
    let batch_extent = graph.free(batch);
    let token_extent = graph.free(tokens);
    let frees = dynamic.then_some((batch_extent, token_extent));
    let input = stream(graph, batch, tokens, frees);
    let parameter = graph.parameter(Shape::of([1, 1, 8, 8]), weights(), Element::Single);
    let squared = graph.mul(input, input);
    let total = graph.sum(squared);
    let rescaled = graph.mul(input, total);
    let projected = graph.matmul(rescaled, parameter);
    let half = graph.cast(projected, Element::Half);
    let exact = graph.cast(half, Element::Single);
    let folded = graph.sum_axis(exact, 3);
    let output = graph.softmax(exact);
    let decision = graph.argmax(output);
    graph.retain(output);
    graph.retain(folded);
    graph.retain(decision);
    Model {
        inputs: vec![input],
        parameter,
        output,
        frees: frees
            .map(|(batch, tokens)| vec![(Extent::Batch, batch), (Extent::Tokens, tokens)])
            .unwrap_or_default(),
    }
}

fn convolution_model(
    graph: &Graph<'static>,
    batch: u32,
    tokens: u32,
    dynamic: bool,
) -> Model<'static> {
    let batch_extent = graph.free(batch);
    let shape = if dynamic {
        Shape::of([batch, 16, tokens, 8]).freed(&[(0, batch_extent)])
    } else {
        Shape::of([batch, 16, tokens, 8])
    };
    let input = graph.input(shape, Element::Single);
    let parameter = graph.parameter(Shape::of([4, 16, 3, 3]), weights(), Element::Single);
    let convolved = graph.conv2d(input, parameter, Window::sliding([3, 3]));
    let pooled = graph.pool2d(convolved, Window::sliding([2, 2]), Pool::Mean);
    let transposed = graph.permute(pooled, [0, 1, 3, 2]);
    let scaled = graph.mul(transposed, transposed);
    let folded = graph.sum_axis(scaled, 2);
    let output = graph.mul(pooled, folded);
    graph.retain(output);
    Model {
        inputs: vec![input],
        parameter,
        output,
        frees: dynamic
            .then_some((Extent::Batch, batch_extent))
            .into_iter()
            .collect(),
    }
}

fn reference(
    runtime: &Runtime,
    build: fn(&Graph<'static>, u32, u32, bool) -> Model<'static>,
    batch: u32,
    tokens: u32,
    seed: u32,
) -> (Vec<f32>, Vec<f32>) {
    let graph = Graph::new();
    let model = build(&graph, batch, tokens, false);
    let store = runtime.weights(&graph);
    let program: Program<'_> = runtime.compile(&graph, &store);
    let observations = data(model.inputs[0].shape().elements(), seed);
    let parameter = data(model.parameter.shape().elements(), seed + 7);
    for input in &model.inputs {
        runtime.write(
            &program,
            *input,
            &observations[..input.shape().elements() as usize],
        );
    }
    runtime.write(&program, model.parameter, &parameter);
    runtime.run(&program);
    (runtime.read(&program, model.output), parameter)
}

fn family_runs_every_binding(
    build: fn(&Graph<'static>, u32, u32, bool) -> Model<'static>,
    bindings: &[(u32, u32)],
) {
    let runtime = open();
    let (batch, tokens) = bindings[0];
    let graph = Graph::new();
    let model = build(&graph, batch, tokens, true);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    assert!(
        program.dynamic(),
        "a graph of a free extent compiles into the program of every binding it names",
    );
    for (batch, tokens) in bindings {
        let seed = batch * 31 + tokens;
        let (expected, parameter) = reference(&runtime, build, *batch, *tokens, seed);
        let binding = model.binding(*batch, *tokens);
        runtime.bind(&program, &binding);
        for input in &model.inputs {
            runtime.write(
                &program,
                *input,
                &data(model.elements(input.shape(), &binding), seed),
            );
        }
        runtime.write(&program, model.parameter, &parameter);
        runtime.run(&program);
        let produced = runtime.read(&program, model.output);
        assert_eq!(
            produced.len(),
            expected.len(),
            "a binding of {batch} by {tokens} came back with {} numbers where {} were expected",
            produced.len(),
            expected.len(),
        );
        assert_close(&produced, &expected, 2e-3);
    }
    assert_eq!(
        runtime.declared_kernels(),
        1,
        "a shape family and every shape its references walk run one device program",
    );
}

#[test]
fn one_device_program_runs_every_binding_of_an_attention_family() {
    family_runs_every_binding(attention_model, &[(4, 6), (2, 3), (1, 1), (3, 5), (4, 6)]);
}

#[test]
fn a_reduction_and_a_narrow_tensor_follow_every_binding() {
    family_runs_every_binding(reduction_model, &[(2, 4), (1, 2), (2, 3), (1, 1)]);
}

#[test]
fn a_convolution_and_a_view_follow_every_binding() {
    family_runs_every_binding(convolution_model, &[(3, 8), (1, 8), (2, 8)]);
}

#[test]
fn a_zero_extent_runs_the_tasks_that_remain() {
    let runtime = open();
    let graph = Graph::new();
    let batch_extent = graph.free(3);
    let token_extent = graph.free(4);
    let input = graph.input(
        Shape::of([1, 3, 4, 8]).freed(&[(1, batch_extent), (2, token_extent)]),
        Element::Single,
    );
    let parameter = graph.parameter(Shape::of([2, 1, 8, 8]), weights(), Element::Single);
    let projected = graph.matmul(input, parameter);
    let output = graph.tanh(projected);
    graph.retain(output);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    runtime.bind(&program, &[3, 4]);
    runtime.write(&program, parameter, &data(2 * 8 * 8, 11));
    for batch in 0..=3 {
        for tokens in 0..=4 {
            runtime.bind(&program, &[batch, tokens]);
            runtime.write(&program, input, &data(batch * tokens * 8, 5));
            runtime.run(&program);
            let produced = runtime.read(&program, output);
            assert_eq!(
                produced.len() as u32,
                2 * batch * tokens * 8,
                "a binding of {batch} by {tokens} reads back {} numbers",
                produced.len(),
            );
            assert!(
                produced.iter().all(|value| value.abs() <= 1.0),
                "a tangent reads back a number beyond its range",
            );
        }
    }
}

#[test]
fn a_family_learns_through_every_binding() {
    let runtime = open();
    let graph = Graph::new();
    let batch = graph.free(4);
    let observations =
        graph.gradient_input(Shape::of([4, 6]).freed(&[(2, batch)]), Element::Single);
    let weight = graph.parameter(Shape::matrix(6, 3), weights(), Element::Single);
    let prediction = graph.matmul(observations, weight);
    let loss = graph.sum(graph.mul(prediction, prediction));
    let gradients = graph.backward(loss);
    graph.retain(loss);
    let store = runtime.weights(&graph);
    let family = runtime.compile(&graph, &store);
    let parameter = data(18, 13);
    runtime.bind(&family, &[4]);
    runtime.write(&family, weight, &parameter);
    for samples in [4, 2, 1, 3] {
        let graph = Graph::new();
        let fixed = graph.gradient_input(Shape::matrix(samples, 6), Element::Single);
        let shared = graph.parameter(Shape::matrix(6, 3), weights(), Element::Single);
        let prediction = graph.matmul(fixed, shared);
        let loss = graph.sum(graph.mul(prediction, prediction));
        let gradients_fixed = graph.backward(loss);
        graph.retain(loss);
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        let observations_data = data(samples * 6, 17);
        runtime.write(&program, fixed, &observations_data);
        runtime.write(&program, shared, &parameter);
        runtime.run(&program);
        let expected = runtime.read(&program, gradients_fixed.of(shared));

        runtime.bind(&family, &[samples]);
        runtime.write(&family, observations, &observations_data);
        runtime.run(&family);
        let produced = runtime.read(&family, gradients.of(weight));
        assert_eq!(produced.len(), expected.len());
        assert_close(&produced, &expected, 2e-3);
    }
}

#[test]
fn a_free_extent_stops_where_a_second_length_starts() {
    let graph = Graph::new();
    let batch = graph.free(4);
    let input = graph.input(
        Shape::of([1, 4, 8, 8]).freed(&[(1, batch)]),
        Element::Single,
    );
    let other = graph.input(Shape::of([1, 2, 8, 8]), Element::Single);
    assert!(refuses(|| {
        graph.concat(&[input, other], 1);
    }));
    assert!(refuses(|| {
        graph.slice(input, 1, 0, 2);
    }));
    assert!(refuses(|| {
        graph.mean_axis(input, 1);
    }));
    assert!(refuses(|| {
        graph.reshape(input, Shape::of([1, 4, 64]));
    }));
    let width = graph.free(8);
    let wide = graph.input(
        Shape::of([1, 4, 8, 8]).freed(&[(3, width)]),
        Element::Single,
    );
    assert!(refuses(|| {
        graph.rope(wide, None, 10_000.0);
    }));
    let rows = graph.free(8);
    let image = graph.input(Shape::of([1, 4, 8, 8]).freed(&[(2, rows)]), Element::Single);
    assert!(refuses(|| {
        graph.pool2d(image, Window::sliding([2, 2]), Pool::Mean);
    }));
    assert!(refuses(|| {
        graph.parameter(
            Shape::of([1, 4, 8, 8]).freed(&[(1, batch)]),
            weights(),
            Element::Single,
        );
    }));
    assert!(refuses(|| {
        graph.input(
            Shape::of([1, 4, 8, 8]).freed(&[(1, graph.free(8))]),
            Element::Single,
        );
    }));
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_program_of_free_extents_runs_no_binding_but_the_one_it_names() {
    let runtime = open();
    let graph = Graph::new();
    let batch = graph.free(4);
    let input = graph.input(
        Shape::of([1, 4, 8, 8]).freed(&[(1, batch)]),
        Element::Single,
    );
    let output = graph.tanh(input);
    graph.retain(output);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    assert!(
        refuses(|| {
            runtime.write(&program, input, &data(8, 3));
        }),
        "a family that names no binding reads no tensor",
    );
    runtime.bind(&program, &[4]);
    assert!(
        refuses(|| {
            runtime.bind(&program, &[5]);
        }),
        "a binding beyond the bound the graph declares stops where it stands",
    );
    assert!(
        refuses(|| {
            runtime.bind(&program, &[2, 2]);
        }),
        "a binding of the wrong number of extents stops where it stands",
    );
    runtime.write(&program, input, &data(4 * 8 * 8, 3));
    runtime.run(&program);
    let produced = runtime.read(&program, output);
    assert_eq!(produced.len(), 4 * 8 * 8);
}
