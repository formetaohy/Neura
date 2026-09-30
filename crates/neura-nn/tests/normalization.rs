use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_nn::{GroupNorm, RmsNorm, mse_loss};
use neura_runtime::{Runtime, RuntimeRequest};

fn open() -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.0,
        high: 1.0,
    }
    .samples(count, &mut entropy)
}

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index} came back as {actual} where {expected} was expected",
        );
    }
}

fn rms_reference(values: &[f32], scale: &[f32], rows: u32, columns: u32, floor: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; values.len()];
    for row in 0..rows as usize {
        let start = row * columns as usize;
        let mean = values[start..start + columns as usize]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / columns as f32;
        let deviation = (mean + floor).sqrt();
        for column in 0..columns as usize {
            out[start + column] = values[start + column] / deviation * scale[column];
        }
    }
    out
}

fn group_reference(
    values: &[f32],
    scale: &[f32],
    shift: &[f32],
    dims: [u32; 4],
    groups: u32,
    floor: f32,
) -> Vec<f32> {
    let [batches, channels, rows, columns] = dims;
    let per_group = channels / groups;
    let plane = (rows * columns) as usize;
    let mut out = vec![0.0f32; values.len()];
    for batch in 0..batches {
        for group in 0..groups {
            let mut elements = Vec::new();
            for channel in group * per_group..(group + 1) * per_group {
                let start = ((batch * channels + channel) * rows * columns) as usize;
                elements.extend_from_slice(&values[start..start + plane]);
            }
            let mean = elements.iter().sum::<f32>() / elements.len() as f32;
            let spread = elements
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f32>()
                / elements.len() as f32;
            let deviation = (spread + floor).sqrt();
            for channel in group * per_group..(group + 1) * per_group {
                let start = ((batch * channels + channel) * rows * columns) as usize;
                for index in 0..plane {
                    out[start + index] = (values[start + index] - mean) / deviation
                        * scale[channel as usize]
                        + shift[channel as usize];
                }
            }
        }
    }
    out
}

fn sampled(count: usize) -> Vec<usize> {
    let step = (count / 8).max(1);
    (0..count).step_by(step).collect()
}

fn assert_slope(element: usize, analytic: f32, numeric: f32, count: usize) {
    assert!(
        (analytic - numeric).abs() <= 2e-2 * (1.0 + analytic.abs() + numeric.abs()),
        "element {element} of {count} slopes {analytic} analytically and {numeric} numerically",
    );
}

#[test]
fn a_root_mean_square_normalizes_every_row() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::matrix(4, 6), Element::Single);
    let layer = RmsNorm::new(&graph, 6, Init::Zero, 1e-6, Element::Single);
    let out = layer.forward(&graph, input);
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(24, 3);
    let scale = random(6, 5);
    runtime.write(&program, input, &values);
    runtime.write(&program, layer.scale(), &scale);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, out),
        &rms_reference(&values, &scale, 4, 6, 1e-6),
        1e-4,
    );
}

#[test]
fn a_root_mean_square_gradient_matches_finite_differences() {
    let graph = Graph::new();
    let input = graph.input(Shape::matrix(4, 6), Element::Single);
    let target = graph.input(Shape::matrix(4, 6), Element::Single);
    let layer = RmsNorm::new(&graph, 6, Init::Zero, 1e-6, Element::Single);
    let loss = mse_loss(&graph, layer.forward(&graph, input), target);
    let gradient = graph.backward(loss).of(layer.scale());
    graph.retain(gradient);
    graph.retain(loss);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, input, &random(24, 23));
    runtime.write(&program, target, &random(24, 71));
    runtime.run(&program);
    let values = runtime.read(&program, layer.scale());
    let analytic = runtime.read(&program, gradient);
    for element in sampled(values.len()) {
        let step = 0.01;
        let mut probe = values.clone();
        probe[element] += step;
        runtime.write(&program, layer.scale(), &probe);
        runtime.run(&program);
        let high = runtime.read(&program, loss)[0];
        probe[element] -= 2.0 * step;
        runtime.write(&program, layer.scale(), &probe);
        runtime.run(&program);
        let low = runtime.read(&program, loss)[0];
        assert_slope(
            element,
            analytic[element],
            (high - low) / (2.0 * step),
            values.len(),
        );
    }
    runtime.write(&program, layer.scale(), &values);
}

#[test]
fn a_group_normalization_normalizes_each_group_of_channels() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 4, 3, 3]), Element::Single);
    let layer = GroupNorm::new(&graph, 4, 2, Init::Zero, 1e-5, Element::Single);
    let out = layer.forward(&graph, input);
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(72, 29);
    let scales = [0.5f32, -0.25, 1.0, 0.75];
    let shifts = [0.1f32, -0.2, 0.3, 0.0];
    runtime.write(&program, input, &values);
    runtime.write(&program, layer.scale(), &scales);
    runtime.write(&program, layer.shift(), &shifts);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, out),
        &group_reference(&values, &scales, &shifts, [2, 4, 3, 3], 2, 1e-5),
        1e-4,
    );
}

#[test]
fn a_group_normalization_gradient_matches_finite_differences() {
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 4, 3, 3]), Element::Single);
    let target = graph.input(Shape::of([2, 4, 3, 3]), Element::Single);
    let layer = GroupNorm::new(&graph, 4, 2, Init::Zero, 1e-5, Element::Single);
    let loss = mse_loss(&graph, layer.forward(&graph, input), target);
    let gradients = graph.backward(loss);
    let scaled = gradients.of(layer.scale());
    let shifted = gradients.of(layer.shift());
    graph.retain(scaled);
    graph.retain(shifted);
    graph.retain(loss);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, input, &random(72, 29));
    runtime.write(&program, target, &random(72, 43));
    runtime.run(&program);
    for (parameter, gradient) in [(layer.scale(), scaled), (layer.shift(), shifted)] {
        let values = runtime.read(&program, parameter);
        let analytic = runtime.read(&program, gradient);
        for element in sampled(values.len()) {
            let step = 0.01;
            let mut probe = values.clone();
            probe[element] += step;
            runtime.write(&program, parameter, &probe);
            runtime.run(&program);
            let high = runtime.read(&program, loss)[0];
            probe[element] -= 2.0 * step;
            runtime.write(&program, parameter, &probe);
            runtime.run(&program);
            let low = runtime.read(&program, loss)[0];
            assert_slope(
                element,
                analytic[element],
                (high - low) / (2.0 * step),
                values.len(),
            );
        }
        runtime.write(&program, parameter, &values);
    }
}

#[test]
fn a_group_normalization_keeps_its_scale_beside_its_shift() {
    let graph = Graph::new();
    let layer = GroupNorm::new(&graph, 4, 2, Init::Zero, 1e-5, Element::Single);
    assert_eq!(layer.parameters().len(), 2);
    assert_eq!(graph.shape(layer.scale()).dims(), [1, 4, 1, 1]);
    assert_eq!(graph.shape(layer.shift()).dims(), [1, 4, 1, 1]);
}
