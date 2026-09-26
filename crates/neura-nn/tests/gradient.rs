use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value, Window};
use neura_nn::{Conv2d, Embedding, LayerNorm, Linear, cross_entropy, mse_loss, policy_loss};
use neura_runtime::{Runtime, RuntimeRequest};

fn open() -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn sampled(elements: usize) -> impl Iterator<Item = usize> {
    let stride = (elements / 4).max(1);
    (0..elements).step_by(stride).take(4)
}

fn assert_slope(element: usize, analytic: f32, numeric: f32, elements: usize) {
    let slack = 1e-3 + 1e-2 * analytic.abs().max(numeric.abs());
    assert!(
        (numeric - analytic).abs() <= slack,
        "element {element} of a tensor of {elements} numbers: the tape gives {analytic} where the slope is {numeric}",
    );
}

#[test]
fn a_batched_product_gradient_matches_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let left = graph.parameter(
        Shape::of([2, 1, 3, 4]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let right = graph.parameter(
        Shape::of([2, 3, 4, 2]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let product = graph.matmul(left, right);
    assert_eq!(product.shape(), Shape::of([2, 3, 3, 2]));
    let targets = graph.input(Shape::of([2, 3, 3, 2]), Element::Single);
    let loss = mse_loss(&graph, product, targets);
    let gradients = graph.backward(loss);
    let parameters = [left, right];
    for parameter in &parameters {
        graph.retain(gradients.of(*parameter));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        targets,
        &(0..36)
            .map(|index| (index as f32 * 0.031).cos() * 0.5)
            .collect::<Vec<_>>(),
    );
    runtime.run(&program);
    for parameter in &parameters {
        let values = runtime.read(&program, *parameter);
        let analytic = runtime.read(&program, gradients.of(*parameter));
        assert_eq!(
            analytic.len(),
            values.len(),
            "the gradient of {} numbers reaches a parameter of {} numbers",
            analytic.len(),
            values.len(),
        );
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.1);
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            let numeric = (high - low) / (2.0 * step);
            assert_slope(element, analytic[element], numeric, values.len());
        }
        runtime.write(&program, *parameter, &values);
    }
}

#[test]
fn a_normalized_row_gradient_matches_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.parameter(
        Shape::of([2, 3, 5]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let layer = LayerNorm::new(
        &graph,
        5,
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        1e-5,
        Element::Single,
    );
    let targets = graph.input(Shape::of([2, 3, 5]), Element::Single);
    let loss = mse_loss(&graph, layer.forward(&graph, input), targets);
    let gradients = graph.backward(loss);
    let mut parameters = vec![input];
    parameters.extend(layer.parameters());
    for parameter in &parameters {
        graph.retain(gradients.of(*parameter));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        targets,
        &(0..30)
            .map(|index| (index as f32 * 0.043).sin() * 0.5)
            .collect::<Vec<_>>(),
    );
    runtime.run(&program);
    for parameter in &parameters {
        let values = runtime.read(&program, *parameter);
        let analytic = runtime.read(&program, gradients.of(*parameter));
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.1);
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            let numeric = (high - low) / (2.0 * step);
            assert_slope(element, analytic[element], numeric, values.len());
        }
        runtime.write(&program, *parameter, &values);
    }
}

#[test]
fn analytic_gradients_match_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let first = Linear::new(
        &graph,
        2,
        4,
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let second = Linear::new(
        &graph,
        4,
        1,
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let inputs = graph.input(Shape::matrix(4, 2), Element::Single);
    let targets = graph.input(Shape::matrix(4, 1), Element::Single);
    let hidden = graph.relu(first.forward(&graph, inputs));
    let loss = mse_loss(&graph, second.forward(&graph, hidden), targets);
    let gradients = graph.backward(loss);
    let parameters = first
        .parameters()
        .into_iter()
        .chain(second.parameters())
        .collect::<Vec<Value>>();
    for parameter in &parameters {
        graph.retain(gradients.of(*parameter));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        inputs,
        &[0.1, -0.4, 0.7, 0.2, -0.3, 0.9, 0.5, -0.8],
    );
    runtime.write(&program, targets, &[1.0, -0.5, 0.25, -0.75]);
    runtime.run(&program);
    for parameter in &parameters {
        let values = runtime.read(&program, *parameter);
        let analytic = runtime.read(&program, gradients.of(*parameter));
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.1);
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            let numeric = (high - low) / (2.0 * step);
            assert_slope(element, analytic[element], numeric, values.len());
        }
        runtime.write(&program, *parameter, &values);
    }
}

#[test]
fn analytic_gradients_of_a_deep_stack_match_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let model = neura_nn::Mlp::new(
        &graph,
        &[3, 5, 4, 2],
        Init::Uniform {
            low: -0.6,
            high: 0.6,
        },
        Element::Single,
    );
    let inputs = graph.input(Shape::matrix(4, 3), Element::Single);
    let targets = graph.input(Shape::matrix(4, 2), Element::Single);
    let loss = mse_loss(&graph, model.forward(&graph, inputs), targets);
    let gradients = graph.backward(loss);
    let parameters = model.parameters();
    for parameter in &parameters {
        graph.retain(gradients.of(*parameter));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        inputs,
        &[
            0.2, -0.5, 0.3, -0.7, 0.1, 0.6, 0.4, 0.8, -0.2, -0.9, 0.5, -0.3,
        ],
    );
    runtime.write(
        &program,
        targets,
        &[0.5, -0.5, -0.25, 0.75, 0.9, 0.1, -0.8, -0.4],
    );
    runtime.run(&program);
    for parameter in &parameters {
        let values = runtime.read(&program, *parameter);
        let analytic = runtime.read(&program, gradients.of(*parameter));
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.1);
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            let numeric = (high - low) / (2.0 * step);
            assert_slope(element, analytic[element], numeric, values.len());
        }
        runtime.write(&program, *parameter, &values);
    }
}

#[test]
fn analytic_gradients_of_a_tensor_wider_than_one_task_match_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let first = Linear::new(
        &graph,
        64,
        32,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    let second = Linear::new(
        &graph,
        32,
        16,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
    );
    let samples = 256;
    let inputs = graph.input(Shape::matrix(samples, 64), Element::Single);
    let targets = graph.input(Shape::matrix(samples, 16), Element::Single);
    let hidden = graph.relu(first.forward(&graph, inputs));
    let loss = mse_loss(&graph, second.forward(&graph, hidden), targets);
    let gradients = graph.backward(loss);
    let parameters = first
        .parameters()
        .into_iter()
        .chain(second.parameters())
        .collect::<Vec<Value>>();
    for parameter in &parameters {
        graph.retain(gradients.of(*parameter));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let inputs_data = (0..samples * 64)
        .map(|index| (index as f32 * 0.017).sin() * 0.5)
        .collect::<Vec<_>>();
    let targets_data = (0..samples * 16)
        .map(|index| (index as f32 * 0.031).cos() * 0.5)
        .collect::<Vec<_>>();
    runtime.write(&program, inputs, &inputs_data);
    runtime.write(&program, targets, &targets_data);
    runtime.run(&program);
    assert!(
        program.task_count() > graph.task_count() as u32,
        "a step this wide is tiled into more tasks than it names ops, and {} is too few",
        program.task_count(),
    );
    for parameter in &parameters {
        let values = runtime.read(&program, *parameter);
        let analytic = runtime.read(&program, gradients.of(*parameter));
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.05);
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            let numeric = (high - low) / (2.0 * step);
            assert_slope(element, analytic[element], numeric, values.len());
        }
        runtime.write(&program, *parameter, &values);
    }
}

#[test]
fn one_step_of_adam_moves_a_weight_against_its_gradient() {
    let runtime = open();
    let graph = Graph::new();
    let layer = Linear::new(
        &graph,
        2,
        1,
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let inputs = graph.input(Shape::matrix(4, 2), Element::Single);
    let targets = graph.input(Shape::matrix(4, 1), Element::Single);
    let loss = mse_loss(&graph, layer.forward(&graph, inputs), targets);
    let gradients = graph.backward(loss);
    let mut optimizer = neura_nn::Adam::new(&graph, 0.1, 0.9, 0.999, 1e-8);
    optimizer.track_all(&graph, &layer.parameters());
    optimizer.step(&graph, &gradients);
    graph.retain(loss);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        inputs,
        &[1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0],
    );
    runtime.write(&program, targets, &[1.0, -1.0, -1.0, 1.0]);
    runtime.run(&program);
    let before = runtime.read(&program, loss)[0];
    for _ in 0..200 {
        runtime.run(&program);
    }
    let after = runtime.read(&program, loss)[0];
    assert!(
        after < before,
        "two hundred steps of adam moved the loss from {before} to {after}",
    );
}

#[test]
fn the_cross_entropy_gradient_of_a_logit_is_its_probability_less_its_target() {
    let runtime = open();
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(4, 3), Init::Zero, Element::Single);
    let targets = graph.input(Shape::matrix(4, 3), Element::Single);
    let loss = cross_entropy(&graph, logits, targets);
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(logits));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let logits_data = vec![
        0.5, -1.0, 0.25, //
        -0.75, 1.5, 0.1, //
        2.0, -0.5, -1.25, //
        0.0, 0.0, 0.0,
    ];
    let targets_data = vec![
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        0.0, 0.0, 1.0, //
        1.0, 0.0, 0.0,
    ];
    runtime.write(&program, logits, &logits_data);
    runtime.write(&program, targets, &targets_data);
    runtime.run(&program);
    let mut expected = Vec::new();
    for row in 0..4 {
        let known = &logits_data[row * 3..row * 3 + 3];
        let largest = known.iter().copied().fold(f32::MIN, f32::max);
        let total = known
            .iter()
            .map(|value| (value - largest).exp())
            .sum::<f32>();
        for column in 0..3 {
            let probability = (known[column] - largest).exp() / total;
            expected.push((probability - targets_data[row * 3 + column]) / 4.0);
        }
    }
    let analytic = runtime.read(&program, gradients.of(logits));
    for (element, (actual, wanted)) in analytic.iter().zip(&expected).enumerate() {
        assert!(
            (actual - wanted).abs() < 1e-5,
            "logit {element} moved by {actual} where its probability less its target says {wanted}",
        );
    }
}

#[test]
fn an_embedding_gradient_matches_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let embedding = Embedding::new(
        &graph,
        6,
        4,
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let indices = graph.input(Shape::matrix(3, 1), Element::Single);
    let picked = embedding.forward(&graph, indices);
    let targets = graph.input(Shape::matrix(3, 4), Element::Single);
    let loss = mse_loss(&graph, picked, targets);
    let gradients = graph.backward(loss);
    let table = embedding.table();
    graph.retain(gradients.of(table));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let index_data = vec![2.0, 5.0, 2.0];
    let target_data = (0..12)
        .map(|index| (index as f32 * 0.077).cos() * 0.25)
        .collect::<Vec<_>>();
    runtime.write(&program, indices, &index_data);
    runtime.write(&program, targets, &target_data);
    runtime.run(&program);
    let values = runtime.read(&program, table);
    let analytic = runtime.read(&program, gradients.of(table));
    for element in [1, 9, 14, 22] {
        let step = 0.01 * values[element].abs().max(0.1);
        let mut probe = values.clone();
        probe[element] += step;
        runtime.write(&program, table, &probe);
        runtime.run(&program);
        let high = runtime.read(&program, loss)[0];
        probe[element] -= 2.0 * step;
        runtime.write(&program, table, &probe);
        runtime.run(&program);
        let low = runtime.read(&program, loss)[0];
        let numeric = (high - low) / (2.0 * step);
        assert_slope(element, analytic[element], numeric, values.len());
    }
    runtime.write(&program, table, &values);
}

#[test]
fn a_convolution_gradient_matches_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let inputs = graph.parameter(
        Shape::of([1, 16, 8, 8]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let conv = Conv2d::new(
        &graph,
        [16, 16],
        1,
        Window::new([3, 3], [1, 1], [1, 1]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let targets = graph.input(Shape::of([1, 16, 8, 8]), Element::Single);
    let predicted = conv.forward(&graph, inputs);
    assert_eq!(predicted.shape(), Shape::of([1, 16, 8, 8]));
    let loss = mse_loss(&graph, predicted, targets);
    let gradients = graph.backward(loss);
    let mut parameters = vec![inputs];
    parameters.extend(conv.parameters());
    for parameter in &parameters {
        graph.retain(gradients.of(*parameter));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        targets,
        &(0..1024)
            .map(|index| (index as f32 * 0.037).sin() * 0.5)
            .collect::<Vec<_>>(),
    );
    runtime.run(&program);
    for parameter in &parameters {
        let values = runtime.read(&program, *parameter);
        let analytic = runtime.read(&program, gradients.of(*parameter));
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.1);
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            let numeric = (high - low) / (2.0 * step);
            assert_slope(element, analytic[element], numeric, values.len());
        }
        runtime.write(&program, *parameter, &values);
    }
}

#[test]
fn a_depthwise_convolution_gradient_matches_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let channels = 8;
    let inputs = graph.parameter(
        Shape::of([1, channels, 5, 5]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let conv = Conv2d::new(
        &graph,
        [channels, channels],
        channels,
        Window::new([3, 3], [1, 1], [1, 1]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let targets = graph.input(Shape::of([1, channels, 5, 5]), Element::Single);
    let predicted = conv.forward(&graph, inputs);
    assert_eq!(predicted.shape(), Shape::of([1, 8, 5, 5]));
    let loss = mse_loss(&graph, predicted, targets);
    let gradients = graph.backward(loss);
    let mut parameters = vec![inputs];
    parameters.extend(conv.parameters());
    for parameter in &parameters {
        graph.retain(gradients.of(*parameter));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        targets,
        &(0..200)
            .map(|index| (index as f32 * 0.053).sin() * 0.5)
            .collect::<Vec<_>>(),
    );
    runtime.run(&program);
    for parameter in &parameters {
        let values = runtime.read(&program, *parameter);
        let analytic = runtime.read(&program, gradients.of(*parameter));
        assert_eq!(
            analytic.len(),
            values.len(),
            "the gradient of {} numbers reaches a tensor of {} numbers",
            analytic.len(),
            values.len(),
        );
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.1);
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            let numeric = (high - low) / (2.0 * step);
            assert_slope(element, analytic[element], numeric, values.len());
        }
        runtime.write(&program, *parameter, &values);
    }
}

#[test]
fn a_policy_gradient_matches_finite_differences() {
    let runtime = open();
    let graph = Graph::new();
    let layer = Linear::new(
        &graph,
        3,
        4,
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let observations = graph.input(Shape::matrix(5, 3), Element::Single);
    let action = graph.input(Shape::matrix(5, 1), Element::Single);
    let advantage = graph.input(Shape::matrix(5, 1), Element::Single);
    let logits = layer.forward(&graph, observations);
    let loss = policy_loss(&graph, logits, action, advantage);
    let gradients = graph.backward(loss);
    let parameters = layer.parameters();
    for parameter in &parameters {
        graph.retain(gradients.of(*parameter));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        observations,
        &(0..15)
            .map(|index| index as f32 * 0.1 - 0.7)
            .collect::<Vec<_>>(),
    );
    runtime.write(&program, action, &[0.0, 2.0, 1.0, 3.0, 0.0]);
    runtime.write(&program, advantage, &[1.0, -0.5, 0.25, -0.75, 2.0]);
    runtime.run(&program);
    for parameter in &parameters {
        let values = runtime.read(&program, *parameter);
        let analytic = runtime.read(&program, gradients.of(*parameter));
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.1);
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, *parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            let numeric = (high - low) / (2.0 * step);
            assert_slope(element, analytic[element], numeric, values.len());
        }
        runtime.write(&program, *parameter, &values);
    }
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn random(elements: u32, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..elements)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        })
        .collect()
}

#[test]
fn a_gradient_walks_back_through_a_view() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(
        Shape::matrix(2, 3),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let observations = graph.input(Shape::matrix(4, 2), Element::Single);
    let turned = graph.input(Shape::matrix(5, 3), Element::Single);
    let targets = graph.input(Shape::matrix(4, 3), Element::Single);
    let turned_targets = graph.input(Shape::matrix(5, 2), Element::Single);
    let squared_targets = graph.input(Shape::matrix(3, 4), Element::Single);
    let hidden = graph.relu(graph.matmul(observations, weight));
    let flipped = graph.transpose(hidden);
    let loss = graph.add(
        graph.add(
            mse_loss(&graph, graph.matmul(observations, weight), targets),
            mse_loss(
                &graph,
                graph.matmul(turned, graph.transpose(weight)),
                turned_targets,
            ),
        ),
        mse_loss(&graph, graph.mul(flipped, flipped), squared_targets),
    );
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(weight));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, observations, &random(8, 3));
    runtime.write(&program, turned, &random(15, 5));
    runtime.write(&program, targets, &random(12, 7));
    runtime.write(&program, turned_targets, &random(10, 11));
    runtime.write(&program, squared_targets, &random(12, 13));
    runtime.run(&program);
    let values = runtime.read(&program, weight);
    let analytic = runtime.read(&program, gradients.of(weight));
    assert_eq!(
        analytic.len(),
        values.len(),
        "the gradient of a weight reaches the weight it belongs to",
    );
    for element in sampled(values.len()) {
        let step = 0.01 * values[element].abs().max(0.1);
        let mut probe = values.clone();
        probe[element] += step;
        runtime.write(&program, weight, &probe);
        runtime.run(&program);
        let high = runtime.read(&program, loss)[0];
        probe[element] -= 2.0 * step;
        runtime.write(&program, weight, &probe);
        runtime.run(&program);
        let low = runtime.read(&program, loss)[0];
        let numeric = (high - low) / (2.0 * step);
        assert_slope(element, analytic[element], numeric, values.len());
    }
    runtime.write(&program, weight, &values);
}

#[test]
fn a_gradient_of_a_view_lands_on_the_tensor_that_owns_its_storage() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(3, 2), Init::Zero, Element::Single);
    let turned = graph.transpose(weight);
    let data = graph.input(Shape::matrix(4, 2), Element::Single);
    let loss = graph.sum(graph.matmul(data, turned));
    let gradients = graph.backward(loss);
    assert!(
        refuses(|| {
            let _ = gradients.of(turned);
        }),
        "a gradient landed on a view instead of the tensor that owns its storage",
    );
    graph.retain(gradients.of(weight));
}
