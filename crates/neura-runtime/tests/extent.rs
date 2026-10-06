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
            segments: None,
            reach: None,
            query_segments: None,
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
        let assembled = runtime.assembled_kernels();
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
            runtime.assembled_kernels(),
            assembled,
            "a binding of {batch} by {tokens} assembles a device program of its own",
        );
        assert_eq!(
            produced.len(),
            expected.len(),
            "a binding of {batch} by {tokens} came back with {} numbers where {} were expected",
            produced.len(),
            expected.len(),
        );
        assert_close(&produced, &expected, 2e-3);
    }
    assert!(
        program.matmul_geometries().len() < program.tiles().len(),
        "a plan that walks {} product tiles compiles the {} of the menu its plan never walks",
        program.matmul_geometries().len(),
        program.tiles().len(),
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
fn a_convolution_weight_gradient_weighs_every_position_a_binding_holds() {
    let runtime = open();
    let graph = Graph::new();
    let batch = graph.free(4);
    let input = graph.input(
        Shape::of([4, 2, 6, 6]).freed(&[(0, batch)]),
        Element::Single,
    );
    let filter = graph.parameter(Shape::of([3, 2, 3, 3]), weights(), Element::Single);
    let convolved = graph.conv2d(input, filter, Window::sliding([3, 3]));
    let loss = graph.sum(convolved);
    let gradients = graph.backward(loss);
    let gradient = gradients.of(filter);
    graph.retain(loss);
    let store = runtime.weights(&graph);
    let family = runtime.compile(&graph, &store);

    let filter_data = data(54, 17);
    for live in [4u32, 3, 1] {
        let graph = Graph::new();
        let fixed = graph.input(Shape::of([live, 2, 6, 6]), Element::Single);
        let shared = graph.parameter(Shape::of([3, 2, 3, 3]), weights(), Element::Single);
        let fixed_loss = graph.sum(graph.conv2d(fixed, shared, Window::sliding([3, 3])));
        let fixed_gradients = graph.backward(fixed_loss);
        let fixed_gradient = fixed_gradients.of(shared);
        graph.retain(fixed_loss);
        let store = runtime.weights(&graph);
        let reference = runtime.compile(&graph, &store);
        let observations = data(live * 72, 29);
        runtime.write(&reference, fixed, &observations);
        runtime.write(&reference, shared, &filter_data);
        runtime.run(&reference);
        let expected = runtime.read(&reference, fixed_gradient);

        runtime.bind(&family, &[live]);
        runtime.write(&family, input, &observations);
        runtime.write(&family, filter, &filter_data);
        runtime.run(&family);
        assert_close(&runtime.read(&family, gradient), &expected, 1e-5);
    }
}

#[test]
fn a_quantized_image_reads_back_the_quantum_its_storage_holds() {
    let runtime = open();
    let graph = Graph::new();
    let tokens = graph.free(6);
    let observations = graph.input(
        Shape::of([1, 1, 6, 4]).freed(&[(2, tokens)]),
        Element::Single,
    );
    let quantized = graph.quantize(observations, 0.25);
    let squared = graph.mul(quantized, quantized);
    graph.retain(quantized);
    graph.retain(squared);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);
    for tokens in [6u32, 4, 1, 3, 0] {
        runtime.bind(&program, &[tokens]);
        let source = data(tokens * 4, 17);
        runtime.write(&program, observations, &source);
        runtime.run(&program);
        let span = program.span(quantized);
        assert_eq!(
            span.payload_bytes(),
            Element::Int8.payload_words(u64::from(tokens * 4)) * 4,
            "a quantized image packs one word of every four numbers the binding names",
        );
        assert_eq!(
            span.table_offset(),
            Element::Int8.payload_words(6 * 4) * 4,
            "the quantum a quantized tensor reconstructs by stands where the layout its bound declares puts it, whatever the binding",
        );
        assert_eq!(span.image_bytes(), span.payload_bytes() + 4);
        let packed = source
            .iter()
            .map(|value| (value / 0.25).round().clamp(-127.0, 127.0) * 0.25)
            .collect::<Vec<_>>();
        let expected = packed.iter().map(|value| value * value).collect::<Vec<_>>();
        let produced = runtime.read_many(&program, &[quantized, squared]);
        assert_close(&produced[0], &packed, 1e-6);
        assert_close(&produced[1], &expected, 1e-6);
    }
}

fn axis_reference(source: &[f32], live: u32, columns: u32) -> Vec<f32> {
    (0..columns)
        .map(|column| {
            let mut total = 0.0;
            for row in 0..live {
                total += source[(row * columns + column) as usize];
            }
            total * total
        })
        .collect()
}

#[test]
fn a_long_axis_folds_in_stages_under_every_binding() {
    let runtime = open();
    for (bound, columns, bindings) in [
        (96u32, 512u32, vec![96u32, 40, 17, 3, 1]),
        (5000, 4, vec![5000, 3000, 40, 1, 0]),
    ] {
        let graph = Graph::new();
        let extent = graph.free(bound);
        let input = graph.input(
            Shape::of([1, bound, columns, 1]).freed(&[(1, extent)]),
            Element::Single,
        );
        let folded = graph.sum_axis(input, 1);
        let out = graph.mul(folded, folded);
        graph.retain(out);
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        for live in bindings {
            runtime.bind(&program, &[live]);
            let source = data(live * columns, 23 + live);
            runtime.write(&program, input, &source);
            runtime.run(&program);
            assert_close(
                &runtime.read(&program, out),
                &axis_reference(&source, live, columns),
                1e-2,
            );
        }
    }
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
fn a_mean_family_weighs_the_length_every_binding_holds() {
    let runtime = open();
    let graph = Graph::new();
    let rows = graph.free(8);
    let dynamic = Shape::of([1, 1, 8, 4]).freed(&[(2, rows)]);
    let (values, weight, mean, gradient) = mean_family(&graph, dynamic);
    let store = runtime.weights(&graph);
    let family = runtime.compile(&graph, &store);
    for live in [8u32, 5, 1] {
        let graph = Graph::new();
        let (fixed, shared, fixed_mean, fixed_gradient) =
            mean_family(&graph, Shape::of([1, 1, live, 4]));
        let values_data = data(live * 4, 31);
        let weight_data = data(4, 37);
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        runtime.write(&program, fixed, &values_data);
        runtime.write(&program, shared, &weight_data);
        runtime.run(&program);
        let expected_mean = runtime.read(&program, fixed_mean);
        let expected_gradient = runtime.read(&program, fixed_gradient);

        runtime.bind(&family, &[live]);
        runtime.write(&family, values, &values_data);
        runtime.write(&family, weight, &weight_data);
        runtime.run(&family);
        assert_close(&runtime.read(&family, mean), &expected_mean, 1e-5);
        assert_close(&runtime.read(&family, gradient), &expected_gradient, 1e-5);
    }
    runtime.bind(&family, &[0]);
    runtime.write(&family, values, &[]);
    runtime.run(&family);
    assert!(
        runtime
            .read(&family, mean)
            .iter()
            .all(|value| value.is_nan()),
        "a mean of no numbers divides an empty sum by an empty length",
    );
}

#[test]
fn a_fold_of_one_number_walks_the_length_a_binding_holds() {
    let runtime = open();
    let graph = Graph::new();
    let single = graph.free(1);
    let values = graph.gradient_input(
        Shape::of([1, 1, 4, 1]).freed(&[(3, single)]),
        Element::Single,
    );
    let weight = graph.input(Shape::of([1, 1, 4, 1]), Element::Single);
    let summed = graph.sum_rows(values);
    let mean = graph.mean_axis(values, 3);
    let loss = graph.sum(graph.mul(summed, weight));
    let gradients = graph.backward(loss);
    let gradient = gradients.of(values);
    graph.retain(summed);
    graph.retain(mean);
    graph.retain(loss);
    let store = runtime.weights(&graph);
    let family = runtime.compile(&graph, &store);

    let graph = Graph::new();
    let fixed = graph.gradient_input(Shape::of([1, 1, 4, 1]), Element::Single);
    let shared = graph.input(Shape::of([1, 1, 4, 1]), Element::Single);
    let fixed_summed = graph.sum_rows(fixed);
    let fixed_mean = graph.mean_axis(fixed, 3);
    let fixed_loss = graph.sum(graph.mul(fixed_summed, shared));
    let fixed_gradients = graph.backward(fixed_loss);
    let fixed_gradient = fixed_gradients.of(fixed);
    graph.retain(fixed_summed);
    graph.retain(fixed_mean);
    graph.retain(fixed_loss);
    let store = runtime.weights(&graph);
    let program = runtime.compile(&graph, &store);

    let values_data = data(4, 41);
    let weight_data = data(4, 43);
    runtime.write(&program, fixed, &values_data);
    runtime.write(&program, shared, &weight_data);
    runtime.run(&program);
    let expected_summed = runtime.read(&program, fixed_summed);
    let expected_mean = runtime.read(&program, fixed_mean);
    let expected_loss = runtime.read(&program, fixed_loss);
    let expected_gradient = runtime.read(&program, fixed_gradient);

    runtime.bind(&family, &[1]);
    runtime.write(&family, values, &values_data);
    runtime.write(&family, weight, &weight_data);
    runtime.run(&family);
    assert_close(&runtime.read(&family, summed), &expected_summed, 1e-5);
    assert_close(&runtime.read(&family, mean), &expected_mean, 1e-5);
    assert_close(&runtime.read(&family, loss), &expected_loss, 1e-5);
    assert_close(&runtime.read(&family, gradient), &expected_gradient, 1e-5);

    runtime.bind(&family, &[0]);
    runtime.write(&family, values, &[]);
    runtime.write(&family, weight, &weight_data);
    runtime.run(&family);
    assert_close(&runtime.read(&family, summed), &[0.0; 4], 1e-6);
    assert_close(&runtime.read(&family, loss), &[0.0], 1e-6);
    assert!(
        runtime
            .read(&family, mean)
            .iter()
            .all(|value| value.is_nan()),
        "a mean of the no numbers a free axis of one holds divides an empty sum by an empty length",
    );
    assert!(
        runtime.read(&family, gradient).is_empty(),
        "a gradient of a tensor that holds no number reads back no number",
    );
}

fn mean_family(
    graph: &Graph<'static>,
    shape: Shape,
) -> (
    Value<'static>,
    Value<'static>,
    Value<'static>,
    Value<'static>,
) {
    let values = graph.gradient_input(shape, Element::Single);
    let weight = graph.input(Shape::of([1, 1, 1, 4]), Element::Single);
    let mean = graph.mean_axis(values, 2);
    let loss = graph.sum(graph.mul(mean, weight));
    let gradients = graph.backward(loss);
    graph.retain(mean);
    graph.retain(loss);
    (values, weight, mean, gradients.of(values))
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
fn a_softmax_family_walks_every_width_a_binding_holds() {
    let runtime = open();
    for log in [false, true] {
        let graph = Graph::new();
        let width_extent = graph.free(8);
        let dynamic = Shape::of([1, 1, 4, 8]).freed(&[(3, width_extent)]);
        let (logits, weight, probabilities, gradient) = softmax_family(&graph, log, dynamic);
        let store = runtime.weights(&graph);
        let family = runtime.compile(&graph, &store);
        for width in [8u32, 5, 1, 0] {
            let logits_data = data(4 * width, 5);
            let weight_data = data(4 * width, 11);
            let (expected, expected_gradient) = if width == 0 {
                (Vec::new(), Vec::new())
            } else {
                let graph = Graph::new();
                let (logits, weight, probabilities, gradient) =
                    softmax_family(&graph, log, Shape::of([1, 1, 4, width]));
                let store = runtime.weights(&graph);
                let program = runtime.compile(&graph, &store);
                runtime.write(&program, logits, &logits_data);
                runtime.write(&program, weight, &weight_data);
                runtime.run(&program);
                (
                    runtime.read(&program, probabilities),
                    runtime.read(&program, gradient),
                )
            };
            runtime.bind(&family, &[width]);
            runtime.write(&family, logits, &logits_data);
            runtime.write(&family, weight, &weight_data);
            runtime.run(&family);
            assert_close(&runtime.read(&family, probabilities), &expected, 1e-4);
            assert_close(&runtime.read(&family, gradient), &expected_gradient, 1e-4);
        }
    }
}

fn softmax_family(
    graph: &Graph<'static>,
    log: bool,
    shape: Shape,
) -> (
    Value<'static>,
    Value<'static>,
    Value<'static>,
    Value<'static>,
) {
    let logits = graph.gradient_input(shape, Element::Single);
    let weight = graph.input(shape, Element::Single);
    let probabilities = if log {
        graph.log_softmax(logits)
    } else {
        graph.softmax(logits)
    };
    let loss = graph.sum(graph.mul(probabilities, weight));
    let gradients = graph.backward(loss);
    graph.retain(probabilities);
    graph.retain(loss);
    (logits, weight, probabilities, gradients.of(logits))
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
    assert!(
        refuses(|| {
            graph.concat(&[other, input], 1);
        }),
        "a free extent beside the head of a concatenation holds the shift every tensor behind it takes",
    );
    assert!(refuses(|| {
        graph.concat(&[other, input, other], 1);
    }));
    assert!(refuses(|| {
        graph.slice(input, 1, 0, 2);
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
    let taps = graph.free(3);
    let filter = graph.input(Shape::of([4, 4, 3, 3]).freed(&[(2, taps)]), Element::Single);
    let plain = graph.input(Shape::of([1, 4, 8, 8]), Element::Single);
    assert!(
        refuses(|| {
            graph.conv2d(plain, filter, Window::sliding([3, 3]));
        }),
        "a window walks the taps of its filter row by row, and a tap axis that walks a free extent hands the window a length a binding rules",
    );
    assert!(refuses(|| {
        graph.conv2d(image, filter, Window::sliding([3, 3]));
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

fn beside_family(
    graph: &Graph<'static>,
    rows: Shape,
    position: Shape,
) -> (
    Value<'static>,
    Value<'static>,
    Value<'static>,
    Value<'static>,
    neura_graph::Gradients<'static>,
) {
    let rows = graph.gradient_input(rows, Element::Single);
    let position = graph.gradient_input(position, Element::Single);
    let weight = graph.input(graph.shape(rows), Element::Single);
    let sum = graph.add(rows, position);
    let loss = graph.sum(graph.mul(sum, weight));
    let gradients = graph.backward(loss);
    graph.retain(sum);
    graph.retain(loss);
    (rows, position, weight, sum, gradients)
}

#[test]
fn a_walk_of_the_last_axis_a_binding_shrinks_reads_every_number() {
    const WIDTH: u32 = 8;
    const RANKS: u32 = 2;
    let runtime = open();
    let graph = Graph::new();
    let columns = graph.free(WIDTH);
    let shape = Shape::of([1, 1, RANKS, WIDTH]).freed(&[(3, columns)]);
    let rows = graph.gradient_input(shape, Element::Single);
    let weight = graph.input(shape, Element::Single);
    let loss = graph.sum(graph.mul(rows, weight));
    let gradients = graph.backward(loss);
    graph.retain(loss);
    let store = runtime.weights(&graph);
    let family = runtime.compile(&graph, &store);
    for live in [WIDTH, 3, 1] {
        let values = data(RANKS * live, 79);
        runtime.bind(&family, &[live]);
        runtime.write(&family, rows, &values);
        runtime.write(&family, weight, &values);
        runtime.run(&family);
        assert_close(&runtime.read(&family, gradients.of(rows)), &values, 1e-6);
    }
}

#[test]
fn a_static_operand_stands_beside_every_length_a_binding_holds() {
    const BOUND: u32 = 6;
    const WIDTH: u32 = 4;
    let runtime = open();
    let graph = Graph::new();
    let tokens = graph.free(BOUND);
    let (rows, position, weight, sum, gradients) = beside_family(
        &graph,
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, tokens)]),
        Shape::of([1, 1, BOUND, WIDTH]),
    );
    let store = runtime.weights(&graph);
    let family = runtime.compile(&graph, &store);
    let position_data = data(BOUND * WIDTH, 53);
    for live in [BOUND, BOUND - 1, 1] {
        let rows_data = data(live * WIDTH, 59);
        let weight_data = data(live * WIDTH, 61);
        let head = position_data[..(live * WIDTH) as usize].to_vec();
        let graph = Graph::new();
        let (fixed, fixed_position, shared, fixed_sum, fixed_gradients) = beside_family(
            &graph,
            Shape::of([1, 1, live, WIDTH]),
            Shape::of([1, 1, live, WIDTH]),
        );
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        runtime.write(&program, fixed, &rows_data);
        runtime.write(&program, fixed_position, &head);
        runtime.write(&program, shared, &weight_data);
        runtime.run(&program);
        let expected_sum = runtime.read(&program, fixed_sum);
        let expected_rows = runtime.read(&program, fixed_gradients.of(fixed));
        let expected_position = runtime.read(&program, fixed_gradients.of(fixed_position));

        runtime.bind(&family, &[live]);
        runtime.write(&family, rows, &rows_data);
        runtime.write(&family, position, &position_data);
        runtime.write(&family, weight, &weight_data);
        runtime.run(&family);
        assert_close(&runtime.read(&family, sum), &expected_sum, 1e-5);
        assert_close(
            &runtime.read(&family, gradients.of(rows)),
            &expected_rows,
            1e-5,
        );
        let produced = runtime.read(&family, gradients.of(position));
        assert_eq!(produced.len(), (BOUND * WIDTH) as usize);
        assert_close(
            &produced[..(live * WIDTH) as usize],
            &expected_position,
            1e-5,
        );
        assert!(
            produced[(live * WIDTH) as usize..]
                .iter()
                .all(|number| *number == 0.0),
            "a static operand stands beside the steps a binding names, and the walk hands it no gradient beyond the step it holds",
        );
    }
    runtime.bind(&family, &[0]);
    runtime.write(&family, rows, &[]);
    runtime.write(&family, position, &position_data);
    runtime.write(&family, weight, &[]);
    runtime.run(&family);
    assert!(runtime.read(&family, sum).is_empty());
    assert!(runtime.read(&family, gradients.of(rows)).is_empty());
    assert!(
        runtime
            .read(&family, gradients.of(position))
            .iter()
            .all(|number| *number == 0.0),
        "a walk of no steps holds no gradient",
    );
}

#[test]
fn a_static_operand_stands_beside_a_width_a_binding_holds() {
    const WIDTH: u32 = 8;
    const RANKS: u32 = 2;
    let runtime = open();
    let graph = Graph::new();
    let columns = graph.free(WIDTH);
    let rows_shape = Shape::of([1, 1, RANKS, WIDTH]).freed(&[(3, columns)]);
    let rows = graph.gradient_input(rows_shape, Element::Single);
    let every_row = graph.gradient_input(Shape::of([1, 1, RANKS, WIDTH]), Element::Single);
    let every_column = graph.gradient_input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let weight = graph.input(rows_shape, Element::Single);
    let sum = graph.add(graph.add(rows, every_row), every_column);
    let loss = graph.sum(graph.mul(sum, weight));
    let gradients = graph.backward(loss);
    graph.retain(sum);
    graph.retain(loss);
    let store = runtime.weights(&graph);
    let family = runtime.compile(&graph, &store);
    let row_data = data(RANKS * WIDTH, 67);
    let column_data = data(WIDTH, 71);
    for live in [WIDTH, WIDTH - 3, 1] {
        let rows_data = data(RANKS * live, 73);
        let weight_data = data(RANKS * live, 79);
        let head = (0..RANKS * live)
            .map(|index| row_data[(index / live * WIDTH + index % live) as usize])
            .collect::<Vec<f32>>();
        let graph = Graph::new();
        let fixed = graph.gradient_input(Shape::of([1, 1, RANKS, live]), Element::Single);
        let fixed_row = graph.gradient_input(Shape::of([1, 1, RANKS, live]), Element::Single);
        let fixed_column = graph.gradient_input(Shape::of([1, 1, 1, live]), Element::Single);
        let shared = graph.input(Shape::of([1, 1, RANKS, live]), Element::Single);
        let fixed_sum = graph.add(graph.add(fixed, fixed_row), fixed_column);
        let fixed_loss = graph.sum(graph.mul(fixed_sum, shared));
        let fixed_gradients = graph.backward(fixed_loss);
        graph.retain(fixed_sum);
        graph.retain(fixed_loss);
        let store = runtime.weights(&graph);
        let program = runtime.compile(&graph, &store);
        runtime.write(&program, fixed, &rows_data);
        runtime.write(&program, fixed_row, &head);
        runtime.write(&program, fixed_column, &column_data[..live as usize]);
        runtime.write(&program, shared, &weight_data);
        runtime.run(&program);
        let expected_sum = runtime.read(&program, fixed_sum);
        let expected_rows = runtime.read(&program, fixed_gradients.of(fixed));
        let expected_row = runtime.read(&program, fixed_gradients.of(fixed_row));
        let expected_column = runtime.read(&program, fixed_gradients.of(fixed_column));

        runtime.bind(&family, &[live]);
        runtime.write(&family, rows, &rows_data);
        runtime.write(&family, every_row, &row_data);
        runtime.write(&family, every_column, &column_data);
        runtime.write(&family, weight, &weight_data);
        runtime.run(&family);
        assert_close(&runtime.read(&family, sum), &expected_sum, 1e-5);
        assert_close(
            &runtime.read(&family, gradients.of(rows)),
            &expected_rows,
            1e-5,
        );
        let produced = runtime.read(&family, gradients.of(every_row));
        assert_eq!(produced.len(), (RANKS * WIDTH) as usize);
        for rank in 0..RANKS {
            let kept = &produced[(rank * WIDTH) as usize..(rank * WIDTH + live) as usize];
            let expected = &expected_row[(rank * live) as usize..((rank + 1) * live) as usize];
            assert_close(kept, expected, 1e-5);
            assert!(
                produced[(rank * WIDTH + live) as usize..((rank + 1) * WIDTH) as usize]
                    .iter()
                    .all(|number| *number == 0.0),
                "a static operand lands the gradient of every column the walk names, and no number beyond them",
            );
        }
        let produced = runtime.read(&family, gradients.of(every_column));
        assert_close(&produced[..live as usize], &expected_column, 1e-5);
        assert!(
            produced[live as usize..]
                .iter()
                .all(|number| *number == 0.0),
            "a row of static numbers stands beside the columns a binding names",
        );
    }
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
