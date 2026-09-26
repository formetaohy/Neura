use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Pool, Shape, Value, Window};
use neura_precision::{pack, unpack};
use neura_profile::{Budget, Profile};
use neura_program::Encoding;
use neura_runtime::{Runtime, RuntimeRequest};

#[path = "support/reference.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use reference::{matmul_reference, random};
use support::{assert_close, open};

fn narrow() -> Profile {
    Profile::derive(Budget::BASELINE)[0]
}

fn rounded(element: Element, values: &[f32]) -> Vec<f32> {
    unpack(element, values.len(), &pack(element, values))
}

#[test]
fn a_cast_keeps_the_numbers_it_narrows_on_the_tape() {
    let runtime = open();
    let graph = Graph::new();
    let data = graph.input(Shape::vector(64), Element::Single);
    let halves = graph.cast(data, Element::Half);
    let rectified = graph.relu(halves);
    let back = graph.cast(rectified, Element::Single);
    assert_eq!(graph.element(halves), Element::Half);
    assert_eq!(graph.element(rectified), Element::Half);
    assert_eq!(graph.element(back), Element::Single);
    graph.retain(halves);
    graph.retain(rectified);
    graph.retain(back);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(64, 5);
    runtime.write(&program, data, &values);
    runtime.run(&program);

    let halves_expected = rounded(Element::Half, &values);
    let mut rectified_expected = halves_expected.clone();
    for value in &mut rectified_expected {
        *value = value.max(0.0);
    }
    assert_eq!(
        runtime.read(&program, halves),
        halves_expected,
        "a cast narrows every number the tape carries",
    );
    assert_eq!(
        runtime.read(&program, rectified),
        rectified_expected,
        "an op over a narrow tensor packs the numbers it computes",
    );
    assert_eq!(
        runtime.read(&program, back),
        runtime.read(&program, rectified),
        "widening a narrow tensor copies the numbers it holds",
    );
}

#[test]
fn a_narrow_gather_feeds_a_narrow_product() {
    let runtime = open();
    let graph = Graph::new();
    let table = graph.parameter(
        Shape::matrix(4, 3),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Half,
    );
    let weight = graph.parameter(
        Shape::matrix(3, 2),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Half,
    );
    let indices = graph.input(Shape::matrix(2, 1), Element::Single);
    let rows = graph.gather(table, indices);
    let product = graph.matmul(rows, weight);
    assert_eq!(graph.element(rows), Element::Half);
    assert_eq!(graph.element(product), Element::Half);
    graph.retain(rows);
    graph.retain(product);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let table_values = random(12, 7);
    let weight_values = random(6, 11);
    runtime.write(&program, table, &table_values);
    runtime.write(&program, weight, &weight_values);
    runtime.write(&program, indices, &[2.0, 0.0]);
    runtime.run(&program);

    let table_values = rounded(Element::Half, &table_values);
    let weight_values = rounded(Element::Half, &weight_values);
    let mut rows_expected = Vec::new();
    for row in [2usize, 0] {
        rows_expected.extend_from_slice(&table_values[row * 3..row * 3 + 3]);
    }
    assert_close(&runtime.read(&program, rows), &rows_expected, 1e-4);
    assert_close(
        &runtime.read(&program, product),
        &matmul_reference(&rows_expected, &weight_values, 2, 3, 2),
        1e-2,
    );
}

#[test]
fn a_narrow_arena_holds_half_the_bytes_of_a_wide_one() {
    let wide_graph = Graph::new();
    let data = wide_graph.input(Shape::vector(1024), Element::Single);
    wide_graph.retain(wide_graph.relu(data));
    let narrow_graph = Graph::new();
    let data = narrow_graph.input(Shape::vector(1024), Element::Half);
    narrow_graph.retain(narrow_graph.relu(data));
    let wide = Encoding::of(&wide_graph, 256, narrow());
    let narrow = Encoding::of(&narrow_graph, 256, narrow());
    assert_eq!(wide.arena_bytes(), 8192);
    assert_eq!(narrow.arena_bytes(), 4096);
}

fn rounding_contract(backends: neura_gpu::Backends) {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        gpu: neura_gpu::GpuRequest {
            backends,
            ..Default::default()
        },
        readback_bytes: 1 << 20,
        ..Default::default()
    }))
    .expect("a device rounds the numbers the host would");
    let probes = probes();
    for element in [Element::Half, Element::Bfloat16] {
        let graph = Graph::new();
        let data = graph.input(Shape::vector(probes.len() as u32), Element::Single);
        let narrowed = graph.cast(data, element);
        graph.retain(narrowed);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, data, &probes);
        runtime.run(&program);
        let device = runtime.read(&program, narrowed);
        let host = unpack(element, probes.len(), &pack(element, &probes));
        for (index, (device, host)) in device.iter().zip(&host).enumerate() {
            assert_eq!(
                device.to_bits(),
                host.to_bits(),
                "element {index} of a {} tensor came back as {device} where the host packs {host}, from {}",
                element.name(),
                probes[index],
            );
        }
    }
}

fn probes() -> Vec<f32> {
    let mut probes = vec![
        -0.9993706,
        0.9993,
        1.0006,
        -1.0009,
        0.1,
        5.960_464_5e-8,
        6.0e-8,
        -5.960_464_5e-8,
        65504.0,
        65519.0,
        65520.0,
        65535.0,
        100000.0,
        -100000.0,
        0.0,
        -0.0,
        0.333_333_34,
        1024.5,
        1.0e-5,
        -3.402_823_5e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let sample = |count: u32, seed: u32| random(count, seed);
    probes.extend(sample(512, 13));
    for exponent in -40i32..40 {
        let scale = 2.0f32.powi(exponent);
        let seed = (exponent + 41) as u32;
        probes.extend(sample(4, seed).iter().map(|value| value * scale));
    }
    probes
}

#[cfg(any(target_os = "windows", target_os = "linux"))]
#[test]
fn vulkan_rounds_the_numbers_the_host_would() {
    rounding_contract(neura_gpu::Backends::VULKAN);
}

#[cfg(target_os = "windows")]
#[test]
fn dx12_rounds_the_numbers_the_host_would() {
    rounding_contract(neura_gpu::Backends::DX12);
}

#[cfg(target_os = "macos")]
#[test]
fn metal_rounds_the_numbers_the_host_would() {
    rounding_contract(neura_gpu::Backends::METAL);
}

fn narrow_stack<'g>(graph: &Graph<'g>, element: Element, data: Shape) -> [Value<'g>; 6] {
    let filter = graph.parameter(
        Shape::of([4, 2, 3, 3]),
        Init::Uniform {
            low: -0.3,
            high: 0.3,
        },
        element,
    );
    let channel_bias = graph.parameter(Shape::of([1, 4, 1, 1]), Init::Zero, element);
    let dense = graph.parameter(
        Shape::of([1, 1, 4, 4]),
        Init::Uniform {
            low: -0.3,
            high: 0.3,
        },
        element,
    );
    let keys = graph.parameter(
        Shape::of([1, 1, 9, 4]),
        Init::Uniform {
            low: -0.3,
            high: 0.3,
        },
        element,
    );
    let input = graph.input(data, element);
    let convolved = graph.add(
        graph.conv2d(input, filter, Window::sliding([3, 3])),
        channel_bias,
    );
    let pooled = graph.pool2d(
        graph.relu(convolved),
        Window::new([2, 2], [2, 2], [0, 0]),
        Pool::Mean,
    );
    let rows = graph.reshape(pooled, Shape::of([1, 1, 9, 4]));
    let projected = graph.matmul(rows, dense);
    let attended = graph.attention(
        projected,
        keys,
        keys,
        AttentionOptions {
            scale: 0.5,
            causal: false,
            origin: None,
        },
    );
    let out = graph.sum(graph.softmax(attended));
    [input, filter, channel_bias, dense, keys, out]
}

#[test]
fn a_narrow_model_meets_its_wide_twin() {
    let runtime = open();
    let shape = Shape::of([1, 2, 8, 8]);
    let wide_graph = Graph::new();
    let [
        wide,
        wide_filter,
        wide_bias,
        wide_dense,
        wide_keys,
        wide_out,
    ] = narrow_stack(&wide_graph, Element::Single, shape);
    wide_graph.retain(wide_out);
    let narrow_graph = Graph::new();
    let [
        narrow,
        narrow_filter,
        narrow_bias,
        narrow_dense,
        narrow_keys,
        narrow_out,
    ] = narrow_stack(&narrow_graph, Element::Half, shape);
    narrow_graph.retain(narrow_out);
    let wide_weights = runtime.weights(&wide_graph);
    let narrow_weights = runtime.weights(&narrow_graph);
    let wide_program = runtime.compile(&wide_graph, &wide_weights);
    let narrow_program = runtime.compile(&narrow_graph, &narrow_weights);
    let data = random(128, 3);
    runtime.write(&wide_program, wide, &data);
    runtime.write(&narrow_program, narrow, &data);
    for (wide_parameter, narrow_parameter, count, seed) in [
        (wide_filter, narrow_filter, 4 * 2 * 3 * 3, 5),
        (wide_bias, narrow_bias, 4, 7),
        (wide_dense, narrow_dense, 16, 11),
        (wide_keys, narrow_keys, 36, 13),
    ] {
        let values = random(count, seed);
        runtime.write(&wide_program, wide_parameter, &values);
        runtime.write(&narrow_program, narrow_parameter, &values);
    }
    runtime.run(&wide_program);
    runtime.run(&narrow_program);
    let wide_total = runtime.read(&wide_program, wide_out)[0];
    let narrow_total = runtime.read(&narrow_program, narrow_out)[0];
    assert!(
        (wide_total - narrow_total).abs() <= 1e-3 * (1.0 + wide_total.abs()),
        "a half precision stack of convolution, pool, product and attention summed {narrow_total} where its single precision twin summed {wide_total}",
    );
}
