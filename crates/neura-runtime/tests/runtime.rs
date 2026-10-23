use neura_abi::Element;
use neura_gguf::Gguf;
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value, Window};
use neura_precision::{pack, unpack};
use neura_runtime::MemoryRequest;
use support::attention::{Shapes, attention_forward};
use support::convolution::{conv2d_input_reference, conv2d_reference, conv2d_weight_reference};
use support::{assert_close, gguf_bytes, open, opened, random, rounded};

mod support;

#[test]
fn a_narrow_product_walks_the_quantized_weight_a_gguf_stores() {
    let (depth, columns) = (2048u32, 1024u32);
    for element in [Element::Q8_0, Element::Q4K, Element::Q5K, Element::Q6K] {
        let runtime = open();
        let profile = runtime.default_profile();
        let values = random(columns * depth, 23);
        let payload = pack(element, 1.0, &values);
        let shape = [u64::from(columns), u64::from(depth)];
        let weight = unpack(element, values.len(), &payload);
        for rows in [1u32, 2u32, 4u32] {
            let graph = Graph::new();
            let left = graph.input(Shape::matrix(rows, depth), Element::Single);
            let stored = graph.named_block_quantized_parameter(
                element.name(),
                Shape::matrix(columns, depth),
                Init::Zero,
                element,
            );
            let out = graph.matmul(left, graph.permute(stored, [0, 1, 3, 2]));
            graph.retain(out);
            let weights = runtime.load(
                &graph,
                &Gguf::decode(&gguf_bytes(32, &[(element, &shape, &payload)])),
            );
            let program = runtime.compile_with(&graph, &weights, profile);
            if rows == 2 && element.block().is_some() {
                assert!(
                    program
                        .matmul_geometries()
                        .iter()
                        .any(|(tile, _)| tile.rows() == 2),
                    "a two row {} product folds both rows through one weight read",
                    element.name(),
                );
            }
            if rows == 4 && matches!(element, Element::Q4K | Element::Q5K | Element::Q6K) {
                assert!(
                    program
                        .matmul_geometries()
                        .iter()
                        .any(|(tile, _)| tile.rows() == 4),
                    "a four row {} product folds all four rows through one weight read",
                    element.name(),
                );
            }
            let inputs = random(rows * depth, 29);
            runtime.write(&program, left, &inputs);
            runtime.run(&program);
            let mut expected = Vec::with_capacity((rows * columns) as usize);
            for row in 0..rows as usize {
                for column in 0..columns as usize {
                    let mut total = 0.0;
                    for step in 0..depth as usize {
                        total += weight[column * depth as usize + step]
                            * inputs[row * depth as usize + step];
                    }
                    expected.push(total);
                }
            }
            assert_close(&runtime.read(&program, out), &expected, 1e-3);
        }
    }
}

#[test]
fn a_wide_product_walks_the_quantized_weight_a_gguf_stores() {
    let (rows, depth, columns) = (8u32, 512u32, 4096u32);
    for element in [Element::Q8_0, Element::Q4K, Element::Q5K, Element::Q6K] {
        let runtime = open();
        let graph = Graph::new();
        let left = graph.input(Shape::matrix(rows, depth), Element::Single);
        let stored = graph.named_block_quantized_parameter(
            element.name(),
            Shape::matrix(columns, depth),
            Init::Zero,
            element,
        );
        let out = graph.matmul(left, graph.permute(stored, [0, 1, 3, 2]));
        graph.retain(out);
        let values = random(columns * depth, 23);
        let payload = pack(element, 1.0, &values);
        let shape = [u64::from(columns), u64::from(depth)];
        let weights = runtime.load(
            &graph,
            &Gguf::decode(&gguf_bytes(32, &[(element, &shape, &payload)])),
        );
        let program = runtime.compile_with(&graph, &weights, runtime.default_profile());
        if element.packs_codes() {
            assert!(
                program
                    .matmul_geometries()
                    .iter()
                    .any(|(tile, _)| tile.stages_a_run()),
                "a wide product of {} walks the codes it stages through the panel a staged tile keeps",
                element.name(),
            );
        }
        let inputs = random(rows * depth, 29);
        runtime.write(&program, left, &inputs);
        runtime.run(&program);
        let weight = unpack(element, values.len(), &payload);
        let mut expected = Vec::with_capacity((rows * columns) as usize);
        for row in 0..rows as usize {
            for column in 0..columns as usize {
                let mut total = 0.0;
                for step in 0..depth as usize {
                    total += weight[column * depth as usize + step]
                        * inputs[row * depth as usize + step];
                }
                expected.push(total);
            }
        }
        assert_close(&runtime.read(&program, out), &expected, 1e-3);
    }
}

#[test]
fn a_grouped_product_walks_the_rows_each_segment_closes() {
    let runtime = open();
    let (bound, depth, columns, segments) = (16u32, 8u32, 4u32, 4u32);
    let lengths = [2.0f32, 0.0, 5.0, 1.0];
    let graph = Graph::new();
    let lengths_value = graph.input(Shape::vector(segments), Element::Single);
    let ragged = graph.ragged(bound, lengths_value);
    let input = graph.input(
        Shape::of([1, 1, bound, depth]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let first = graph.named_parameter(
        "first",
        Shape::of([segments, 1, depth, columns]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let second = graph.named_parameter(
        "second",
        Shape::of([segments, 1, columns, depth]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let middle = graph.grouped_matmul(input, first, ragged.offsets);
    let total = graph.sum(graph.grouped_matmul(middle, second, ragged.offsets));
    for value in [middle, total, first, second] {
        graph.retain(value);
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = random(bound * depth, 17);
    runtime.write(&program, input, &held);
    runtime.write(&program, lengths_value, &lengths);
    runtime.run(&program);
    let first_held = runtime.read(&program, first);
    let second_held = runtime.read(&program, second);
    let starts = [0usize, 2, 2, 7];
    let total_rows = lengths.iter().map(|length| *length as usize).sum::<usize>();
    let mut middle_expected = vec![0.0f32; total_rows * columns as usize];
    for segment in 0..segments as usize {
        for row in 0..lengths[segment] as usize {
            for column in 0..columns as usize {
                let mut sum = 0.0;
                for step in 0..depth as usize {
                    sum += held[(starts[segment] + row) * depth as usize + step]
                        * first_held[(segment * depth as usize + step) * columns as usize + column];
                }
                middle_expected[(starts[segment] + row) * columns as usize + column] = sum;
            }
        }
    }
    assert_close(&runtime.read(&program, middle), &middle_expected, 1e-4);
    let mut total_expected = 0.0;
    for segment in 0..segments as usize {
        for row in 0..lengths[segment] as usize {
            for column in 0..depth as usize {
                for step in 0..columns as usize {
                    total_expected += middle_expected
                        [(starts[segment] + row) * columns as usize + step]
                        * second_held
                            [(segment * columns as usize + step) * depth as usize + column];
                }
            }
        }
    }
    assert_close(&runtime.read(&program, total), &[total_expected], 1e-3);
}

#[test]
fn a_gather_copies_the_rows_its_indices_name() {
    let (rows, vocabulary, width) = (64u32, 4096u32, 128u32);
    let runtime = open();
    let graph = Graph::new();
    let table = graph.parameter(
        Shape::of([1, 1, vocabulary, width]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let indices = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
    let gathered = graph.gather(table, indices);
    graph.retain(gathered);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let stored = (0..vocabulary * width)
        .map(|at| (at as f32 * 0.001).sin())
        .collect::<Vec<f32>>();
    let walked = (0..rows)
        .map(|at| ((at * 97 + 13) % vocabulary) as f32)
        .collect::<Vec<f32>>();
    runtime.write(&program, table, &stored);
    runtime.write(&program, indices, &walked);
    runtime.run(&program);
    let expected = walked
        .iter()
        .flat_map(|row| {
            let start = *row as usize * width as usize;
            stored[start..start + width as usize].iter().copied()
        })
        .collect::<Vec<f32>>();
    assert_close(&runtime.read(&program, gathered), &expected, 0.0);
}

#[test]
fn a_segment_sum_adds_the_rows_each_plane_closes() {
    let columns = 128u32;
    let lengths = [
        37.0f32, 23.0, 41.0, 17.0, 31.0, 29.0, 43.0, 19.0, 37.0, 11.0, 47.0, 13.0, 31.0, 23.0,
        41.0, 19.0,
    ];
    let bound = lengths.iter().sum::<f32>() as u32;
    let planes = lengths.len() as u32;
    let runtime = open();
    let graph = Graph::new();
    let lengths_value = graph.input(Shape::vector(planes), Element::Single);
    let ragged = graph.ragged(bound, lengths_value);
    let values = graph.input(
        Shape::of([1, 1, bound, columns]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let summed = graph.segment_sum(values, ragged);
    graph.retain(summed);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let stored = random(bound * columns, 31);
    runtime.write(&program, values, &stored);
    runtime.write(&program, lengths_value, &lengths);
    runtime.run(&program);
    let mut expected = Vec::with_capacity((planes * columns) as usize);
    let mut start = 0usize;
    for length in lengths {
        for column in 0..columns as usize {
            let mut total = 0.0;
            for row in 0..length as usize {
                total += stored[(start + row) * columns as usize + column];
            }
            expected.push(total);
        }
        start += length as usize;
    }
    assert_close(&runtime.read(&program, summed), &expected, 1e-4);
}

#[test]
fn a_sectioned_top_k_sorts_every_band_it_walks() {
    const ROWS: u32 = 1;
    const WIDTH: u32 = 32768;
    const KEEP: u32 = 64;
    let graph = Graph::new();
    let logits = graph.input(Shape::of([1, 1, ROWS, WIDTH]), Element::Single);
    let (values, indices) = graph.top_k(logits, KEEP);
    graph.retain(values);
    graph.retain(indices);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = random(ROWS * WIDTH, 37);
    runtime.write(&program, logits, &held);
    runtime.run(&program);
    let mut order = (0..WIDTH).collect::<Vec<u32>>();
    order.sort_by(|left, right| {
        held[*right as usize]
            .partial_cmp(&held[*left as usize])
            .expect("a random logit is a number")
            .then(left.cmp(right))
    });
    let expected_values = order[..KEEP as usize]
        .iter()
        .map(|at| held[*at as usize])
        .collect::<Vec<f32>>();
    let expected_indices = order[..KEEP as usize]
        .iter()
        .map(|at| *at as f32)
        .collect::<Vec<f32>>();
    assert_close(&runtime.read(&program, values), &expected_values, 1e-6);
    assert_eq!(runtime.read(&program, indices), expected_indices);
}

#[test]
fn a_gated_wave_waits_for_every_wave_before_it() {
    const ELEMENTS: u32 = 1 << 20;
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 1, 1, ELEMENTS]), Element::Single);
    let total = graph.sum(input);
    graph.retain(total);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let mut program = runtime.compile(&graph, &weights);
    let values = vec![1.0; ELEMENTS as usize];
    runtime.write(&program, input, &values);
    runtime.run(&program);
    assert_close(&runtime.read(&program, total), &[ELEMENTS as f32], 0.0);
    assert!(program.wave_count() > 1);
    runtime.write(&program, input, &values);
    {
        let mut session = runtime.session(&mut program);
        while session.submit(1) {}
        session.wait();
    }
    assert_close(&runtime.read(&program, total), &[ELEMENTS as f32], 0.0);
}

#[test]
fn a_cursor_resumes_the_slice_it_paused_once_that_slice_completes() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 1, 8, 8192]), Element::Single);
    let first = graph.softmax(input);
    let second = graph.softmax(first);
    graph.retain(second);
    let weights = runtime.weights(&graph);
    let mut program = runtime.compile(&graph, &weights);
    let values = random(8 * 8192, 41);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let expected = runtime.read(&program, second);
    let waves = program.wave_count();
    assert!(waves > 1);
    runtime.write(&program, input, &values);
    let mut session = runtime.session(&mut program);
    session.submit(1);
    let cursor = session.pause();
    assert!(cursor.wave() > 0 && !cursor.done());
    while !cursor.ready() {
        std::hint::spin_loop();
    }
    let mut resumed = runtime.resume(&mut program, cursor);
    while resumed.submit(1) {}
    resumed.wait();
    assert_eq!(resumed.wave(), waves);
    assert_close(&runtime.read(&program, second), &expected, 1e-6);
}

#[test]
fn a_session_walks_a_plan_a_slice_at_a_time() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 1, 8, 8192]), Element::Single);
    let first = graph.softmax(input);
    let second = graph.softmax(first);
    graph.retain(second);
    let weights = runtime.weights(&graph);
    let mut program = runtime.compile(&graph, &weights);
    let values = random(8 * 8192, 41);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let expected = runtime.read(&program, second);
    let waves = program.wave_count();
    assert!(
        waves > 1,
        "a pair of softmaxes walks {waves} waves over {} tasks",
        program.task_count(),
    );
    for budget in [1, 2, 3] {
        runtime.write(&program, input, &values);
        {
            let mut session = runtime.session(&mut program);
            while session.submit(budget) {}
            session.wait();
            assert_eq!(session.wave(), waves);
        }
        assert_close(&runtime.read(&program, second), &expected, 1e-6);
    }
}

fn peaked(rows: u32, columns: u32, sections: u32, peak: f32) -> Vec<f32> {
    let mut values = vec![-peak; (rows * columns) as usize];
    let span = columns / sections;
    for row in 0..rows {
        for section in 0..sections {
            let at = row * columns + section * span + (span / 2 + section) % span;
            values[at as usize] = peak - section as f32;
        }
    }
    values
}

fn softmax_reference(input: &[f32], columns: u32) -> Vec<f32> {
    let mut out = vec![0.0f32; input.len()];
    for (row, chunk) in input.chunks(columns as usize).enumerate() {
        let largest = chunk.iter().copied().fold(f32::MIN, f32::max);
        let total: f64 = chunk
            .iter()
            .map(|value| f64::from(value - largest).exp())
            .sum();
        for (column, value) in chunk.iter().enumerate() {
            out[row * columns as usize + column] =
                (f64::from(value - largest).exp() / total) as f32;
        }
    }
    out
}

#[test]
fn a_starved_softmax_parts_its_columns_between_the_tasks_of_its_rows() {
    let (rows, columns) = (1u32, 4096u32);
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(rows, columns), Init::Zero, Element::Single);
    let probabilities = graph.softmax(logits);
    graph.retain(probabilities);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = peaked(rows, columns, 4, 8.0);
    runtime.write(&program, logits, &values);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, probabilities),
        &softmax_reference(&values, columns),
        1e-6,
    );
}

fn rotate(
    rows: u32,
    width: u32,
    origin: u32,
    base: f32,
    values: &[f32],
    inverse: bool,
) -> Vec<f32> {
    let half = (width / 2) as usize;
    let width = width as usize;
    let mut out = values.to_vec();
    for row in 0..rows as usize {
        for column in 0..width {
            let high = column >= half;
            let channel = if high { column - half } else { column };
            let angle =
                (origin + row as u32) as f32 * base.powf(-2.0 * channel as f32 / width as f32);
            let (sine, cosine) = angle.sin_cos();
            let partner = if high { column - half } else { column + half };
            let value = values[row * width + column];
            let other = values[row * width + partner];
            let sign = if high != inverse { 1.0 } else { -1.0 };
            out[row * width + column] = value * cosine + sign * other * sine;
        }
    }
    out
}

fn rotate_planes(lengths: &[f32], width: u32, base: f32, inverse: bool, image: &[f32]) -> Vec<f32> {
    let mut out = Vec::new();
    let mut start = 0u32;
    for length in lengths {
        let length = *length as u32;
        let at = (start * width) as usize;
        out.extend(rotate(
            length,
            width,
            0,
            base,
            &image[at..at + (length * width) as usize],
            inverse,
        ));
        start += length;
    }
    out
}

#[test]
fn a_packed_rope_places_every_row_in_the_plane_its_offsets_close() {
    const BOUND: u32 = 8;
    const WIDTH: u32 = 4;
    const BASE: f32 = 10000.0;
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(4), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let packed = graph.gradient_input(
        Shape::of([1, 1, BOUND, WIDTH]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let turned = graph.rope(packed, None, Some(graph.knob(BASE)));
    graph.retain(turned);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    for walked in [
        [3.0f32, 3.0, 2.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [2.0, 1.0, 3.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [4.0, 2.0, 2.0, 0.0],
    ] {
        let image = random(BOUND * WIDTH, 71);
        runtime.write(&program, lengths, &walked);
        runtime.write(&program, packed, &image);
        runtime.run(&program);
        assert_close(
            &runtime.read(&program, turned),
            &rotate_planes(&walked, WIDTH, BASE, false, &image),
            1e-4,
        );
    }
}

#[test]
fn a_causal_attention_block_scores_every_row_it_weights() {
    let shapes = Shapes {
        heads: 4,
        key_heads: 2,
        batch: 2,
        queries: 5,
        keys: 5,
        width: 3,
        causal: true,
        origin: 0,
        reach: 0,
        scale: 0.5,
    };
    let graph = Graph::new();
    let tensor = |heads: u32, tokens: u32| {
        graph.parameter(
            Shape::of([heads, shapes.batch, tokens, shapes.width]),
            Init::Zero,
            Element::Single,
        )
    };
    let queries = tensor(shapes.heads, shapes.queries);
    let keys = tensor(shapes.key_heads, shapes.keys);
    let values = tensor(shapes.key_heads, shapes.keys);
    let out = graph.attention(
        queries,
        keys,
        values,
        AttentionOptions {
            bias: None,
            scale: Some(graph.knob(shapes.scale)),
            causal: shapes.causal,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let queries_data = random(
        shapes.heads * shapes.batch * shapes.queries * shapes.width,
        17,
    );
    let keys_data = random(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        29,
    );
    let values_data = random(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        43,
    );
    runtime.write(&program, queries, &queries_data);
    runtime.write(&program, keys, &keys_data);
    runtime.write(&program, values, &values_data);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, out),
        &attention_forward(shapes, &queries_data, &keys_data, &values_data),
        1e-5,
    );
}

const ROWS: u32 = 6;
const DEPTH: u32 = 4;
const COLUMNS: u32 = 3;

#[test]
fn a_gradient_input_receives_the_slope_of_every_row_it_feeds() {
    let runtime = open();
    let graph = Graph::new();
    let observations = graph.gradient_input(Shape::matrix(ROWS, DEPTH), Element::Single);
    let weight = graph.parameter(Shape::matrix(DEPTH, COLUMNS), Init::Zero, Element::Single);
    let loss = graph.sum(graph.matmul(observations, weight));
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(observations));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data = random(ROWS * DEPTH, 17);
    let kernel = random(DEPTH * COLUMNS, 23);
    runtime.write(&program, observations, &data);
    runtime.write(&program, weight, &kernel);
    runtime.run(&program);
    let mut expected = vec![0.0f32; (ROWS * DEPTH) as usize];
    for row in 0..ROWS {
        for depth in 0..DEPTH {
            let mut total = 0.0;
            for column in 0..COLUMNS {
                total += kernel[(depth * COLUMNS + column) as usize];
            }
            expected[(row * DEPTH + depth) as usize] = total;
        }
    }
    assert_close(
        &runtime.read(&program, gradients.of(observations)),
        &expected,
        1e-5,
    );
}

#[test]
fn a_gradient_input_walks_the_slope_the_device_descends() {
    let runtime = open();
    let graph = Graph::new();
    let observations = graph.gradient_input(Shape::matrix(ROWS, DEPTH), Element::Single);
    let target = graph.input(Shape::matrix(ROWS, COLUMNS), Element::Single);
    let weight = graph.parameter(Shape::matrix(DEPTH, COLUMNS), Init::Zero, Element::Single);
    let prediction = graph.matmul(observations, weight);
    let difference = graph.sub(prediction, target);
    let loss = graph.sum(graph.mul(difference, difference));
    let gradients = graph.backward(loss);
    let rate = graph.fill(Shape::scalar(), -1e-4);
    graph.add_into(observations, graph.mul(gradients.of(observations), rate));
    graph.retain(loss);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data = random(ROWS * DEPTH, 29);
    let kernel = random(DEPTH * COLUMNS, 31);
    let target_data = random(ROWS * COLUMNS, 37);
    runtime.write(&program, observations, &data);
    runtime.write(&program, weight, &kernel);
    runtime.write(&program, target, &target_data);
    runtime.run(&program);
    let before = runtime.read(&program, loss)[0];
    runtime.run(&program);
    let after = runtime.read(&program, loss)[0];
    assert!(
        after < before,
        "a step against the slope lowered the loss from {before} to {after}",
    );
    let moved = runtime.read(&program, observations);
    assert_ne!(moved, data, "the descent moved the observation it followed");
}

#[test]
fn a_gradient_input_receives_the_slope_of_the_numbers_a_slice_and_a_join_name() {
    let runtime = open();
    let graph = Graph::new();
    let observations = graph.gradient_input(Shape::of([1, 1, 1, 8]), Element::Single);
    let sliced = graph.slice(observations, 3, 2, 2);
    let parts = (0..2)
        .map(|head| graph.slice(observations, 3, head * 4, 2))
        .collect::<Vec<_>>();
    let joined = graph.concat(&parts, 0);
    let loss = graph.add(
        graph.sum(graph.mul(sliced, sliced)),
        graph.sum(graph.mul(joined, joined)),
    );
    let gradients = graph.backward(loss);
    let held = gradients.of(observations);
    graph.retain(held);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data = random(8, 41);
    runtime.write(&program, observations, &data);
    runtime.run(&program);
    let mut expected = vec![0.0f32; 8];
    for at in [0, 1, 2, 3, 4, 5] {
        expected[at] = 2.0 * data[at];
    }
    assert_close(&runtime.read(&program, held), &expected, 1e-5);
}

#[test]
fn a_quantized_tensor_reads_back_the_blocks_a_device_packs() {
    for shape in [Shape::matrix(5, 96), Shape::matrix(3, 33), Shape::vector(7)] {
        let runtime = open();
        let graph = Graph::new();
        let data = graph.input(shape, Element::Single);
        let blocks = graph.block_quantize(data, Element::Q8_0);
        graph.retain(blocks);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let values = random(shape.elements(), 7);
        runtime.write(&program, data, &values);
        runtime.run(&program);
        assert_eq!(
            runtime.read(&program, blocks),
            rounded(Element::Q8_0, &values),
            "a device packs {shape:?} into the blocks its host packs",
        );
    }
}

fn writing(element: Element, rows: u32, width: u32) {
    let runtime = open();
    let graph = Graph::new();
    let state = graph.resident(Shape::matrix(rows, width), element);
    let indices = graph.input(Shape::matrix(rows, 1), Element::Single);
    let updates = graph.input(Shape::matrix(rows, width), Element::Single);
    let packed = if element == Element::Single {
        updates
    } else if element.block().is_some() {
        graph.block_quantize(updates, element)
    } else {
        graph.cast(updates, element)
    };
    graph.write_into(state, indices, packed);
    graph.retain(state);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(rows * width, 11);
    let seats: Vec<f32> = (0..rows).map(|row| (row * 2 % rows) as f32).collect();
    runtime.write(&program, indices, &seats);
    runtime.write(&program, updates, &values);
    runtime.run(&program);
    let held = runtime.read(&program, state);
    let mut expected = vec![0.0f32; values.len()];
    for (row, seat) in seats.iter().enumerate() {
        let at = *seat as usize * width as usize;
        let from = row * width as usize;
        expected[at..at + width as usize]
            .copy_from_slice(&rounded(element, &values[from..from + width as usize]));
    }
    assert_eq!(
        held,
        expected,
        "a {rows}x{width} {} state holds the rows a scatter wrote",
        element.name()
    );
}

#[test]
fn a_state_takes_the_rows_a_device_writes() {
    writing(Element::Q8_0, 8, 128);
    writing(Element::Half, 5, 34);
    writing(Element::Single, 4, 96);
}

#[test]
fn a_convolution_walks_the_window_it_is_handed() {
    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::of([2, 3, 6, 6]), Element::Single);
    let filter = graph.parameter(Shape::of([4, 3, 3, 3]), Init::Zero, Element::Single);
    let bias = graph.parameter(Shape::of([1, 4, 1, 1]), Init::Zero, Element::Single);
    let window = Window::new([3, 3], [2, 2], [1, 1], [1, 1]);
    let convolved = graph.conv2d(input, filter, window);
    let (rows, columns, channels) = (3usize, 3usize, 4usize);
    assert_eq!(convolved.shape(), Shape::of([2, 4, 3, 3]));
    let activated = graph.relu(graph.add(convolved, bias));
    graph.retain(activated);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let input_values = random(216, 5);
    let filter_values = random(108, 11);
    let bias_values = random(4, 17);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.write(&program, bias, &bias_values);
    runtime.run(&program);
    let expected = conv2d_reference(&input_values, &filter_values, [2, 3, 6, 6], 4, window)
        .iter()
        .enumerate()
        .map(|(index, value)| (value + bias_values[index / (rows * columns) % channels]).max(0.0))
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, activated), &expected, 1e-4);
}

#[test]
fn a_cache_scores_every_head_of_the_chunk_it_holds() {
    const HEADS: u32 = 4;
    const KEY_HEADS: u32 = 1;
    const WIDTH: u32 = 256;
    const CAPACITY: u32 = 64;
    const ROWS: u32 = 8;
    let runtime = open();
    let graph = Graph::new();
    let cursor = graph.input(Shape::scalar(), Element::Single);
    let values = graph.input(Shape::of([KEY_HEADS, 1, ROWS, WIDTH]), Element::Single);
    let keys = graph.fill(Shape::of([KEY_HEADS, 1, ROWS, WIDTH]), 0.0);
    let queries = graph.fill(Shape::of([HEADS, 1, ROWS, WIDTH]), 0.0);
    let held = Shape::of([KEY_HEADS, 1, CAPACITY, WIDTH]);
    let cache_keys = graph.state(held, Init::Zero, Element::Single);
    let cache_values = graph.state(held, Init::Zero, Element::Single);
    let heads = (0..KEY_HEADS)
        .map(|head| graph.fill(Shape::scalar(), (head * CAPACITY) as f32))
        .collect::<Vec<Value>>();
    let rows = (0..ROWS)
        .map(|row| graph.fill(Shape::scalar(), row as f32))
        .collect::<Vec<Value>>();
    let seats = graph.add(
        graph.concat(&heads, 0),
        graph.add(graph.concat(&rows, 2), cursor),
    );
    graph.write_into(cache_keys, seats, keys);
    graph.write_into(cache_values, seats, values);
    let attended = graph.attention(
        queries,
        cache_keys,
        cache_values,
        AttentionOptions {
            bias: None,
            scale: Some(graph.knob(1.0)),
            causal: true,
            origin: Some(cursor),
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(attended);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, cursor, &[0.0]);
    let values_data = random(KEY_HEADS * ROWS * WIDTH, 53);
    runtime.write(&program, values, &values_data);
    runtime.run(&program);
    let mut expected = vec![0.0f32; (HEADS * ROWS * WIDTH) as usize];
    for head in 0..HEADS as usize {
        for row in 0..ROWS as usize {
            for step in 0..WIDTH as usize {
                let seen = (0..=row)
                    .map(|at| values_data[at * WIDTH as usize + step])
                    .sum::<f32>();
                expected[(head * ROWS as usize + row) * WIDTH as usize + step] =
                    seen / (row + 1) as f32;
            }
        }
    }
    assert_close(&runtime.read(&program, attended), &expected, 1e-4);
}

#[test]
fn a_tiled_convolution_walks_every_window_it_fits() {
    for (shape, outputs, groups, window) in [
        (
            [2u32, 4, 17, 19],
            40u32,
            1u32,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [1, 64, 8, 8],
            8,
            1,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2u32, 4, 24, 24],
            8u32,
            1u32,
            Window::new([3, 3], [1, 1], [0, 0], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            8,
            1,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            6,
            1,
            Window::new([3, 3], [2, 2], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            6,
            1,
            Window::new([3, 3], [1, 1], [2, 2], [2, 2]),
        ),
        (
            [2, 4, 24, 24],
            8,
            2,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            4,
            4,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 8, 21, 23],
            3,
            1,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            8,
            1,
            Window::new([5, 5], [1, 1], [2, 2], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            8,
            1,
            Window::new([1, 1], [1, 1], [0, 0], [1, 1]),
        ),
    ] {
        let runtime = open();
        let graph = Graph::new();
        let [batch, channels, rows, columns] = shape;
        let input = graph.gradient_input(Shape::of(shape), Element::Single);
        let local = channels / groups;
        let filter = graph.parameter(
            Shape::of([outputs, local, window.reach_rows(), window.reach_columns()]),
            Init::Zero,
            Element::Single,
        );
        let convolved = graph.conv2d(input, filter, window);
        graph.retain(convolved);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let input_values = random(batch * channels * rows * columns, 5);
        let filter_values = random(
            outputs * local * window.reach_rows() * window.reach_columns(),
            11,
        );
        runtime.write(&program, input, &input_values);
        runtime.write(&program, filter, &filter_values);
        runtime.run(&program);
        let expected = conv2d_reference(&input_values, &filter_values, shape, outputs, window);
        assert_close(&runtime.read(&program, convolved), &expected, 1e-3);
        let slope = graph.input(convolved.shape(), Element::Single);
        let loss = graph.sum(graph.mul(convolved, slope));
        let gradients = graph.backward(loss);
        let walked = gradients.of(input);
        graph.retain(walked);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let slope_values = random(convolved.shape().elements(), 7);
        runtime.write(&program, input, &input_values);
        runtime.write(&program, filter, &filter_values);
        runtime.write(&program, slope, &slope_values);
        runtime.run(&program);
        let expected = conv2d_input_reference(
            &filter_values,
            &slope_values,
            shape,
            outputs,
            groups,
            window,
        );
        assert_close(&runtime.read(&program, walked), &expected, 1e-3);
    }
}

#[test]
fn a_tiled_convolution_transpose_scatters_the_input_it_reads() {
    for (shape, outputs, groups, window) in [
        (
            [2u32, 4, 12, 12],
            10u32,
            2u32,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 12, 12],
            10,
            2,
            Window::new([3, 3], [2, 2], [1, 1], [1, 1]),
        ),
        (
            [2, 6, 11, 13],
            12,
            3,
            Window::new([3, 3], [1, 1], [2, 2], [2, 2]),
        ),
    ] {
        let runtime = open();
        let graph = Graph::new();
        let input = graph.input(Shape::of(shape), Element::Single);
        let local = outputs / groups;
        let filter = graph.parameter(
            Shape::of([shape[1], local, window.reach_rows(), window.reach_columns()]),
            Init::Zero,
            Element::Single,
        );
        let scattered = graph.conv2d_transpose(input, filter, window, groups);
        graph.retain(scattered);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let input_values = random(shape.iter().product(), 5);
        let filter_values = random(
            shape[1] * local * window.reach_rows() * window.reach_columns(),
            11,
        );
        runtime.write(&program, input, &input_values);
        runtime.write(&program, filter, &filter_values);
        runtime.run(&program);
        let expected = conv2d_input_reference(
            &filter_values,
            &input_values,
            scattered.shape().dims(),
            shape[1],
            groups,
            window,
        );
        assert_close(&runtime.read(&program, scattered), &expected, 1e-3);
    }
}

#[test]
fn a_tiled_convolution_walks_the_filter_a_resident_page_holds() {
    let runtime = opened(neura_runtime::MemoryRequest {
        readback_bytes: 4 << 20,
        resident_weight_bytes: Some(32 << 10),
        ..Default::default()
    });
    let graph = Graph::new();
    let shape = [1u32, 32, 12, 12];
    let outputs = 32u32;
    let window = Window::new([3, 3], [1, 1], [1, 1], [1, 1]);
    let input = graph.input(Shape::of(shape), Element::Single);
    let filter = graph.parameter(Shape::of([outputs, 32, 3, 3]), Init::Zero, Element::Single);
    let convolved = graph.conv2d(input, filter, window);
    graph.retain(convolved);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert!(
        program.weight_windows() > 1,
        "the filter spans more than one page"
    );
    let input_values = random(shape[1] * shape[2] * shape[3], 5);
    let filter_values = random(outputs * 32 * 9, 11);
    runtime.write(&program, input, &input_values);
    runtime.write(&program, filter, &filter_values);
    runtime.run(&program);
    let expected = conv2d_reference(&input_values, &filter_values, shape, outputs, window);
    assert_close(&runtime.read(&program, convolved), &expected, 1e-3);
}

#[test]
fn a_tiled_convolution_weight_walks_every_window_it_fits() {
    for (shape, outputs, groups, window) in [
        (
            [2u32, 4, 24, 24],
            8u32,
            1u32,
            Window::new([3, 3], [1, 1], [0, 0], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            8,
            1,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            6,
            1,
            Window::new([3, 3], [2, 2], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            6,
            1,
            Window::new([3, 3], [1, 1], [2, 2], [2, 2]),
        ),
        (
            [2, 4, 24, 24],
            8,
            2,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            4,
            4,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 8, 21, 23],
            3,
            1,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            8,
            1,
            Window::new([5, 5], [1, 1], [2, 2], [1, 1]),
        ),
        (
            [2, 4, 24, 24],
            8,
            1,
            Window::new([1, 1], [1, 1], [0, 0], [1, 1]),
        ),
        (
            [2, 4, 17, 19],
            40,
            1,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
        (
            [2, 5, 23, 25],
            8,
            1,
            Window::new([3, 3], [1, 1], [1, 1], [1, 1]),
        ),
    ] {
        let runtime = open();
        let graph = Graph::new();
        let channels = shape[1];
        let input = graph.input(Shape::of(shape), Element::Single);
        let local = channels / groups;
        let filter = graph.parameter(
            Shape::of([outputs, local, window.reach_rows(), window.reach_columns()]),
            Init::Zero,
            Element::Single,
        );
        let convolved = graph.conv2d(input, filter, window);
        let slope = graph.input(convolved.shape(), Element::Single);
        let loss = graph.sum(graph.mul(convolved, slope));
        let gradients = graph.backward(loss);
        let walked = gradients.of(filter);
        graph.retain(walked);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let input_values = random(shape.iter().product(), 5);
        let slope_values = random(convolved.shape().elements(), 7);
        runtime.write(&program, input, &input_values);
        runtime.write(&program, slope, &slope_values);
        runtime.run(&program);
        let expected =
            conv2d_weight_reference(&input_values, &slope_values, shape, outputs, groups, window);
        assert_close(&runtime.read(&program, walked), &expected, 1e-3);
    }
}

#[test]
fn a_heavy_wave_walks_the_slices_its_work_names() {
    const ELEMENTS: u32 = 1 << 22;
    let runtime = opened(MemoryRequest {
        readback_bytes: 32 << 20,
        heap_bytes: 64 << 20,
        ..Default::default()
    });
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 1, 1, ELEMENTS]), Element::Single);
    let output = graph.relu(input);
    graph.retain(output);
    let weights = runtime.weights(&graph);
    let mut program = runtime.compile(&graph, &weights);
    let total = (0..program.wave_count())
        .map(|wave| program.wave_work(wave, 1))
        .sum::<u64>();
    assert_eq!(total, program.work());
    assert_eq!(program.wave_work(0, program.wave_count()), total);
    assert!(program.wave_count() > 2);
    let values = random(ELEMENTS, 53);
    runtime.write(&program, input, &values);
    runtime.run(&program);
    let expected = values
        .iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<f32>>();
    assert_close(&runtime.read(&program, output), &expected, 0.0);
    runtime.write(&program, input, &values);
    {
        let mut session = runtime.session(&mut program);
        while session.submit(1) {}
        session.wait();
    }
    assert_close(&runtime.read(&program, output), &expected, 0.0);
}

#[test]
fn a_measured_plan_restores_the_state_it_timed() {
    let runtime = open();
    let graph = Graph::new();
    let left = graph.input(Shape::matrix(2, 1024), Element::Single);
    let weight = graph.named_block_quantized_parameter(
        "weight",
        Shape::matrix(8192, 1024),
        Init::Zero,
        Element::Q8_0,
    );
    let state = graph.named_state(
        "state",
        Shape::of([1, 1, 1, 1024]),
        Init::Zero,
        Element::Single,
    );
    let product = graph.matmul(left, graph.permute(weight, [0, 1, 3, 2]));
    graph.retain(product);
    graph.copy_into(
        state,
        graph.add(state, graph.fill(Shape::of([1, 1, 1, 1024]), 1.0)),
    );
    let weights = runtime.weights(&graph);
    let program = runtime.compile_measured(&graph, &weights);
    assert_eq!(runtime.read(&program, state), vec![0.0; 1024]);
    runtime.write(&program, left, &vec![0.5; 2048]);
    runtime.run(&program);
    assert_eq!(runtime.read(&program, state), vec![1.0; 1024]);
}

#[test]
fn a_pinned_schedule_streams_only_the_pages_it_does_not_hold() {
    let plain = crate::support::open();
    let paged = crate::support::opened(neura_runtime::MemoryRequest {
        resident_weight_bytes: Some(64 << 14),
        ..Default::default()
    });
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 1, 1, 128]), Element::Single);
    let mut value = input;
    let mut weights = Vec::new();
    for layer in 0..40u32 {
        let weight = graph.parameter(Shape::of([1, 1, 128, 128]), Init::Zero, Element::Single);
        weights.push((weight, crate::support::random(128 * 128, layer + 3)));
        value = graph.matmul(value, weight);
    }
    graph.retain(value);
    let plain_weights = plain.weights(&graph);
    let plain_program = plain.compile(&graph, &plain_weights);
    let paged_weights = paged.weights(&graph);
    let paged_program = paged.compile(&graph, &paged_weights);
    assert!(
        paged_program.weight_windows() > 1,
        "the weights outrun the store"
    );
    let input_values = crate::support::random(128, 7);
    for (weight, held) in &weights {
        plain.write(&plain_program, *weight, held);
        paged.write(&paged_program, *weight, held);
    }
    plain.write(&plain_program, input, &input_values);
    paged.write(&paged_program, input, &input_values);
    paged.run(&paged_program);
    assert!(
        paged_program.pinned_pages() > 0,
        "the paged schedule pins no page",
    );
    for _ in 0..3 {
        plain.run(&plain_program);
        paged.run(&paged_program);
        assert_close(
            &paged.read(&paged_program, value),
            &plain.read(&plain_program, value),
            1e-4,
        );
    }
}

#[test]
fn a_pinned_schedule_updates_the_pages_it_holds() {
    const WIDTH: u32 = 128;
    const LAYERS: u32 = 24;
    let plain = crate::support::open();
    let paged = crate::support::opened(neura_runtime::MemoryRequest {
        resident_weight_bytes: Some(64 << 14),
        ..Default::default()
    });
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let target = graph.input(Shape::of([1, 1, 1, WIDTH]), Element::Single);
    let mut value = input;
    let mut parameters = Vec::new();
    for layer in 0..LAYERS {
        let weight = graph.parameter(Shape::of([1, 1, WIDTH, WIDTH]), Init::Zero, Element::Single);
        let mut held = crate::support::random(WIDTH * WIDTH, layer + 13);
        for (at, number) in held.iter_mut().enumerate() {
            let (row, column) = (at as u32 / WIDTH, at as u32 % WIDTH);
            *number = if row == column { 0.9 } else { *number * 0.005 };
        }
        parameters.push((weight, held));
        value = graph.matmul(value, weight);
    }
    let difference = graph.sub(value, target);
    let loss = graph.sum(graph.mul(difference, difference));
    let gradients = graph.backward(loss);
    let rate = graph.fill(Shape::scalar(), -1e-3);
    for (parameter, _) in &parameters {
        graph.add_into(*parameter, graph.mul(gradients.of(*parameter), rate));
    }
    graph.retain(loss);
    let plain_weights = plain.weights(&graph);
    let plain_program = plain.compile(&graph, &plain_weights);
    let paged_weights = paged.weights(&graph);
    let paged_program = paged.compile(&graph, &paged_weights);
    assert!(
        paged_program.weight_windows() > 1,
        "the weights outrun the store"
    );
    let input_values = crate::support::random(WIDTH, 19);
    let target_values = crate::support::random(WIDTH, 23);
    for (parameter, held) in &parameters {
        plain.write(&plain_program, *parameter, held);
        paged.write(&paged_program, *parameter, held);
    }
    plain.write(&plain_program, input, &input_values);
    paged.write(&paged_program, input, &input_values);
    plain.write(&plain_program, target, &target_values);
    paged.write(&paged_program, target, &target_values);
    for _ in 0..3 {
        plain.run(&plain_program);
        paged.run(&paged_program);
    }
    assert!(
        paged_program.pinned_pages() > 0,
        "the paged schedule pins no page",
    );
    assert_close(
        &paged.read(&paged_program, loss),
        &plain.read(&plain_program, loss),
        1e-4,
    );
    let (held, _) = parameters[0];
    assert_close(
        &paged.read(&paged_program, held),
        &plain.read(&plain_program, held),
        1e-4,
    );
}

#[test]
fn a_paged_program_loads_the_rows_a_host_declares() {
    const ROWS: u32 = 512;
    const WIDTH: u32 = 128;
    let plain = crate::support::open();
    let paged = crate::support::opened(neura_runtime::MemoryRequest {
        resident_weight_bytes: Some(160 << 10),
        ..Default::default()
    });
    let graph = Graph::new();
    let table = graph.parameter(Shape::of([1, 1, ROWS, WIDTH]), Init::Zero, Element::Single);
    let indices = graph.input(Shape::of([1, 1, 2, 1]), Element::Single);
    let gathered = graph.gather(table, indices);
    graph.retain(gathered);
    let held = crate::support::random(ROWS * WIDTH, 31);
    let plain_weights = plain.weights(&graph);
    let plain_program = plain.compile(&graph, &plain_weights);
    let paged_weights = paged.weights(&graph);
    let paged_program = paged.compile(&graph, &paged_weights);
    plain.write(&plain_program, table, &held);
    paged.write(&paged_program, table, &held);
    for rows in [[7u32, 300u32], [1, 511], [64, 64]] {
        let wanted = rows.map(|row| row as f32);
        plain.write(&plain_program, indices, &wanted);
        paged.write(&paged_program, indices, &wanted);
        paged.declare_rows(&paged_program, table, &rows);
        plain.run(&plain_program);
        paged.run(&paged_program);
        assert_close(
            &paged.read(&paged_program, gathered),
            &plain.read(&plain_program, gathered),
            1e-6,
        );
    }
}
