use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Pool, Shape, Value, Window};
use neura_precision::{pack, unpack};
use neura_profile::{Budget, Profile};
use neura_program::{Encoding, Layout};
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
    unpack(element, values.len(), &pack(element, 1.0, values))
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
fn a_cast_packs_the_numbers_it_narrows_into_a_float_of_eight_bits() {
    for element in [Element::Fp8E4M3, Element::Fp8E5M2] {
        let runtime = open();
        let graph = Graph::new();
        let data = graph.input(Shape::vector(64), Element::Single);
        let packed = graph.cast(data, element);
        let squared = graph.mul(packed, packed);
        let back = graph.cast(squared, element);
        assert_eq!(graph.element(packed), element);
        assert_eq!(
            graph.element(squared),
            Element::Single,
            "an op over an eight bit float lands in single precision",
        );
        assert_eq!(graph.element(back), element);
        graph.retain(packed);
        graph.retain(squared);
        graph.retain(back);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let values = random(64, 5);
        runtime.write(&program, data, &values);
        runtime.run(&program);

        let packed_expected = rounded(element, &values);
        assert_eq!(
            runtime.read(&program, packed),
            packed_expected,
            "a cast narrows every number the tape carries",
        );
        let swept = runtime.read(&program, squared);
        let mut squared_expected = Vec::new();
        for value in &packed_expected {
            squared_expected.push(value * value);
        }
        assert_close(&swept, &squared_expected, 1e-6);
        let mut back_expected = Vec::new();
        for element_value in rounded(element, &squared_expected) {
            back_expected.push(element_value);
        }
        assert_eq!(
            runtime.read(&program, back),
            back_expected,
            "a product of eight bit floats packs back the way the host packs it",
        );
    }
}

#[test]
fn an_eight_bit_float_arena_holds_a_quarter_of_the_bytes_of_a_wide_one() {
    let wide_graph = Graph::new();
    let data = wide_graph.input(Shape::vector(1024), Element::Single);
    wide_graph.retain(wide_graph.relu(data));
    let narrow_graph = Graph::new();
    let data = narrow_graph.input(Shape::vector(1024), Element::Fp8E4M3);
    narrow_graph.retain(narrow_graph.relu(data));
    let wide = Encoding::of(&wide_graph, 256, narrow());
    let narrow = Encoding::of(&narrow_graph, 256, narrow());
    assert_eq!(wide.arena_bytes(), 8192);
    assert_eq!(narrow.arena_bytes(), 2048);
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

#[test]
fn a_quantized_tensor_carries_the_numbers_its_scale_places() {
    let runtime = open();
    let scale = 0.03125;
    let graph = Graph::new();
    let data = graph.input(Shape::vector(64), Element::Single);
    let quantized = graph.quantize(data, scale);
    let rectified = graph.relu(quantized);
    assert_eq!(graph.element(quantized), Element::Int8);
    assert_eq!(graph.element(rectified), Element::Int8);
    assert_eq!(graph.scale(rectified), scale);
    graph.retain(quantized);
    graph.retain(rectified);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(64, 9);
    runtime.write(&program, data, &values);
    runtime.run(&program);

    let quantized_expected = unpack(
        Element::Int8,
        values.len(),
        &pack(Element::Int8, scale, &values),
    );
    assert_eq!(
        runtime.read(&program, quantized),
        quantized_expected,
        "an int8 tensor carries the numbers a reader reconstructs through its scale",
    );
    let mut rectified_expected = quantized_expected;
    for value in &mut rectified_expected {
        *value = value.max(0.0);
    }
    assert_eq!(
        runtime.read(&program, rectified),
        rectified_expected,
        "an op over a quantized tensor lands in the numbers its storage places",
    );
}

#[test]
fn a_quantized_weight_feeds_a_product_a_quarter_of_the_bytes() {
    let runtime = open();
    let scale = 0.03125;
    let graph = Graph::new();
    let weight = graph.quantized_parameter(
        Shape::matrix(8, 4),
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        scale,
    );
    let input = graph.input(Shape::matrix(3, 8), Element::Single);
    let product = graph.matmul(input, weight);
    graph.retain(product);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let inputs = random(24, 17);
    let weights_values = random(32, 23);
    runtime.write(&program, input, &inputs);
    runtime.write(&program, weight, &weights_values);
    runtime.run(&program);

    let quantized = unpack(
        Element::Int8,
        weights_values.len(),
        &pack(Element::Int8, scale, &weights_values),
    );
    assert_close(
        &runtime.read(&program, product),
        &matmul_reference(&inputs, &quantized, 3, 8, 4),
        1e-4,
    );
    let single = Graph::new();
    single.parameter(
        Shape::matrix(8, 4),
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Single,
    );
    assert_eq!(
        Layout::of(&graph, runtime.alignment()).weights().words(),
        Element::Int8.storage_words(32),
        "a quantized weight packs four numbers a word beside the quantum they share",
    );
    assert_eq!(
        Layout::of(&single, runtime.alignment()).weights().words(),
        32,
    );
}

#[test]
fn an_eight_bit_float_weight_feeds_a_product_a_quarter_of_the_bytes() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(
        Shape::matrix(8, 4),
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Fp8E4M3,
    );
    let input = graph.input(Shape::matrix(3, 8), Element::Single);
    let product = graph.matmul(input, weight);
    graph.retain(product);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let inputs = random(24, 17);
    let weights_values = random(32, 23);
    runtime.write(&program, input, &inputs);
    runtime.write(&program, weight, &weights_values);
    runtime.run(&program);

    let packed = unpack(
        Element::Fp8E4M3,
        weights_values.len(),
        &pack(Element::Fp8E4M3, 1.0, &weights_values),
    );
    assert_close(
        &runtime.read(&program, product),
        &matmul_reference(&inputs, &packed, 3, 8, 4),
        1e-3,
    );
    let single = Graph::new();
    single.parameter(
        Shape::matrix(8, 4),
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Single,
    );
    assert_eq!(
        weights.bytes() * 4,
        runtime.weights(&single).bytes(),
        "an eight bit float weight store holds a quarter of the bytes a single precision one holds",
    );
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
    for (element, probes) in [
        (Element::Half, probes()),
        (Element::Bfloat16, probes()),
        (Element::Fp8E4M3, fp8_probes()),
        (Element::Fp8E5M2, fp8_probes()),
    ] {
        let graph = Graph::new();
        let data = graph.input(Shape::vector(probes.len() as u32), Element::Single);
        let narrowed = graph.cast(data, element);
        graph.retain(narrowed);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, data, &probes);
        runtime.run(&program);
        let device = runtime.read(&program, narrowed);
        let host = unpack(element, probes.len(), &pack(element, 1.0, &probes));
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
    let scale = 0.015625;
    let probes = quantized_probes(scale);
    let graph = Graph::new();
    let data = graph.input(Shape::vector(probes.len() as u32), Element::Single);
    let quantized = graph.quantize(data, scale);
    graph.retain(quantized);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &probes);
    runtime.run(&program);
    let device = runtime.read(&program, quantized);
    let host = unpack(
        Element::Int8,
        probes.len(),
        &pack(Element::Int8, scale, &probes),
    );
    for (index, (device, host)) in device.iter().zip(&host).enumerate() {
        assert_eq!(
            device.to_bits(),
            host.to_bits(),
            "element {index} of an int8 tensor came back as {device} where the host places {host}, from {}",
            probes[index],
        );
    }
}

fn fp8_probes() -> Vec<f32> {
    let mut probes = vec![
        0.0,
        -0.0,
        1.0,
        -1.0,
        0.5,
        -0.5,
        1.0 / 512.0,
        -1.0 / 512.0,
        1.0 / 1024.0,
        1.0 / 256.0,
        1.0 / 65536.0,
        -1.0 / 65536.0,
        1.0 / 131072.0,
        448.0,
        -448.0,
        700.0,
        -700.0,
        57344.0,
        -57344.0,
        70000.0,
        -70000.0,
        1.175_494_4e-38,
        1.0e30,
        -1.0e30,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let sample = |count: u32, seed: u32| random(count, seed);
    probes.extend(sample(512, 37));
    for exponent in -24i32..10 {
        let scale = 2.0f32.powi(exponent);
        let seed = (exponent + 25) as u32;
        probes.extend(sample(4, seed).iter().map(|value| value * scale));
    }
    probes
}

fn quantized_probes(scale: f32) -> Vec<f32> {
    let mut probes = vec![
        0.0,
        -0.0,
        scale / 2.0,
        -scale / 2.0,
        scale * (1.0 + f32::EPSILON),
        -scale * (1.0 + f32::EPSILON),
        126.5 * scale,
        -126.5 * scale,
        127.0 * scale,
        128.0 * scale,
        -127.0 * scale,
        -128.0 * scale,
        1.0e30,
        -1.0e30,
        scale,
        -scale,
    ];
    let sample = |count: u32, seed: u32| random(count, seed);
    probes.extend(sample(512, 29).iter().map(|value| value * scale * 64.0));
    probes.extend(sample(512, 31).iter().map(|value| value * scale * 0.5));
    probes
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

#[test]
fn a_four_bit_weight_feeds_a_product_a_seventh_of_the_bytes() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.block_quantized_parameter(
        Shape::matrix(300, 4),
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Int4,
    );
    let input = graph.input(Shape::matrix(2, 300), Element::Single);
    let product = graph.matmul(input, weight);
    graph.retain(product);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let inputs = random(600, 17);
    let weight_values = random(1200, 23);
    runtime.write(&program, input, &inputs);
    runtime.write(&program, weight, &weight_values);
    runtime.run(&program);

    let quantized = unpack(
        Element::Int4,
        weight_values.len(),
        &pack(Element::Int4, 1.0, &weight_values),
    );
    assert_close(
        &runtime.read(&program, product),
        &matmul_reference(&inputs, &quantized, 2, 300, 4),
        1e-4,
    );
    assert_eq!(
        runtime.read(&program, weight),
        quantized,
        "a block quantized weight reads back the numbers its blocks place",
    );
    assert_eq!(
        Layout::of(&graph, runtime.alignment()).weights().words(),
        Element::Int4.storage_words(1200),
        "a four bit weight packs eight numbers a word beside one quantum of every 128",
    );
}

#[test]
fn a_four_bit_float_weight_feeds_a_product_its_blocks_place() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.block_quantized_parameter(
        Shape::matrix(300, 4),
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Fp4E2M1,
    );
    let input = graph.input(Shape::matrix(2, 300), Element::Single);
    let product = graph.matmul(input, weight);
    graph.retain(product);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let inputs = random(600, 17);
    let weight_values = random(1200, 23);
    runtime.write(&program, input, &inputs);
    runtime.write(&program, weight, &weight_values);
    runtime.run(&program);

    let quantized = unpack(
        Element::Fp4E2M1,
        weight_values.len(),
        &pack(Element::Fp4E2M1, 1.0, &weight_values),
    );
    assert_close(
        &runtime.read(&program, product),
        &matmul_reference(&inputs, &quantized, 2, 300, 4),
        1e-4,
    );
    assert_eq!(
        runtime.read(&program, weight),
        quantized,
        "a four bit float weight reads back the numbers its blocks place",
    );
    assert_eq!(
        Layout::of(&graph, runtime.alignment()).weights().words(),
        Element::Fp4E2M1.storage_words(1200),
        "a four bit float weight packs eight numbers a word beside one quantum of every 32",
    );
    assert!(
        Element::Fp4E2M1.storage_words(1200) < Element::Half.storage_words(1200),
        "a four bit float weight weighs a quarter of the half precision storage it replaces",
    );
}

#[test]
fn a_seeded_four_bit_weight_decodes_as_the_host_reads_it() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.block_quantized_parameter(
        Shape::matrix(150, 8),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Int4,
    );
    let input = graph.input(Shape::matrix(3, 150), Element::Single);
    let product = graph.matmul(input, weight);
    graph.retain(product);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let inputs = random(450, 29);
    runtime.write(&program, input, &inputs);
    runtime.run(&program);

    let seeded = runtime.read(&program, weight);
    assert_close(
        &runtime.read(&program, product),
        &matmul_reference(&inputs, &seeded, 3, 150, 8),
        1e-4,
    );
    assert!(
        seeded.iter().any(|value| value.abs() > 1e-3),
        "the seeded weight holds numbers rather than the zeros a missing quantum would place",
    );
}
