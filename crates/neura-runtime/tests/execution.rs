use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_profile::{Budget, Profile};
use neura_runtime::{Runtime, RuntimeRequest};

#[path = "support/reference.rs"]
mod reference;
#[path = "support/softmax.rs"]
mod softmax;
#[path = "support/mod.rs"]
mod support;

use reference::{matmul_reference, random};
use softmax::{log_softmax_reference, softmax_reference};
use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn independent_tasks_collapse_into_one_dispatch() {
    let graph = Graph::new();
    let wide = graph.parameter(Shape::vector(65_536), Init::Zero, Element::Single);
    let activated = graph.relu(wide);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(
        program.task_count(),
        32,
        "a rectifier this wide is handed to the device in bounded chunks",
    );
    assert_eq!(program.dispatch_count(), 1);
    runtime.run(&program);
    let out = runtime.read(&program, activated);
    assert_eq!(out.len(), 65_536);
    assert!(out.iter().all(|value| *value == 0.0));
}

#[test]
fn a_matmul_matches_a_cpu_reference() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(7, 5), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(5, 9), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let left_data = random(35, 11);
    let right_data = random(45, 29);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    let product = runtime.read(&program, out);
    assert_close(
        &product,
        &matmul_reference(&left_data, &right_data, 7, 5, 9),
        1e-5,
    );
}

#[test]
fn a_product_that_splits_its_depth_matches_a_cpu_reference() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(4, 512), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(512, 8), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert!(
        program
            .matmul_geometries()
            .iter()
            .any(|(_, tasks)| *tasks > 1),
        "a product of four by eight tiles none of the device's breadth, so its depth splits",
    );
    assert_eq!(
        program.dispatch_count(),
        2,
        "a fold reads every slot of the depth"
    );
    let left_data = random(4 * 512, 41);
    let right_data = random(512 * 8, 43);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, out),
        &matmul_reference(&left_data, &right_data, 4, 512, 8),
        1e-4,
    );
}

#[test]
fn a_split_product_keeps_its_epilogue_and_its_planes() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([2, 1, 4, 512]), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(512, 8), Init::Zero, Element::Single);
    let bias = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let out = graph.relu(graph.add(graph.matmul(left, right), bias));
    assert_eq!(out.shape(), Shape::of([2, 1, 4, 8]));
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let left_data = random(2 * 4 * 512, 47);
    let right_data = random(512 * 8, 53);
    let bias_data = random(8, 59);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.write(&program, bias, &bias_data);
    runtime.run(&program);
    let mut expected = Vec::new();
    for plane in 0..2usize {
        let product = matmul_reference(
            &left_data[plane * 4 * 512..(plane + 1) * 4 * 512],
            &right_data,
            4,
            512,
            8,
        );
        expected.extend(
            product
                .iter()
                .enumerate()
                .map(|(index, value)| (value + bias_data[index % 8]).max(0.0)),
        );
    }
    assert_close(&runtime.read(&program, out), &expected, 1e-4);
}

#[test]
fn a_product_pairs_the_batch_its_operands_share() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([2, 1, 3, 4]), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::of([1, 3, 4, 2]), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    assert_eq!(out.shape(), Shape::of([2, 3, 3, 2]));
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let left_data = random(24, 11);
    let right_data = random(24, 29);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    let mut expected = vec![0.0f32; 36];
    for row in 0..2usize {
        for column in 0..3usize {
            let plane = row * 3 + column;
            let product = matmul_reference(
                &left_data[row * 12..row * 12 + 12],
                &right_data[column * 8..column * 8 + 8],
                3,
                4,
                2,
            );
            expected[plane * 6..plane * 6 + 6].copy_from_slice(&product);
        }
    }
    assert_close(&runtime.read(&program, out), &expected, 1e-5);
}

#[test]
fn a_product_spreads_one_operand_over_every_plane() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::of([2, 3, 7, 5]), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(5, 9), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    assert_eq!(out.shape(), Shape::of([2, 3, 7, 9]));
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let left_data = random(2 * 3 * 7 * 5, 31);
    let right_data = random(45, 37);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    let mut expected = Vec::new();
    for plane in 0..6usize {
        expected.extend(matmul_reference(
            &left_data[plane * 35..plane * 35 + 35],
            &right_data,
            7,
            5,
            9,
        ));
    }
    assert_close(&runtime.read(&program, out), &expected, 1e-5);
}

#[test]
fn a_row_fold_sums_every_row_of_every_plane() {
    let graph = Graph::new();
    let data = graph.parameter(Shape::of([2, 3, 5, 7]), Init::Zero, Element::Single);
    let sums = graph.sum_rows(data);
    assert_eq!(sums.shape(), Shape::of([2, 3, 5, 1]));
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(2 * 3 * 5 * 7, 17);
    runtime.write(&program, data, &values);
    runtime.run(&program);
    let expected = values
        .chunks(7)
        .map(|row| row.iter().sum::<f32>())
        .collect::<Vec<_>>();
    assert_eq!(expected.len(), 30);
    assert_close(&runtime.read(&program, sums), &expected, 1e-4);
}

#[test]
fn a_row_fold_walks_a_view_through_its_strides() {
    let graph = Graph::new();
    let matrix = graph.parameter(Shape::matrix(7, 3), Init::Zero, Element::Single);
    let sums = graph.sum_rows(graph.transpose(matrix));
    assert_eq!(sums.shape(), Shape::matrix(3, 1));
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(21, 23);
    runtime.write(&program, matrix, &values);
    runtime.run(&program);
    let expected = (0..3)
        .map(|column| (0..7).map(|row| values[row * 3 + column]).sum::<f32>())
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, sums), &expected, 1e-4);
}

#[test]
fn a_masked_attention_block_rides_one_tape() {
    let heads = 2usize;
    let tokens = 3u32;
    let width = 2u32;
    let graph = Graph::new();
    let queries = graph.parameter(
        Shape::of([heads as u32, 1, tokens, width]),
        Init::Zero,
        Element::Single,
    );
    let keys = graph.parameter(
        Shape::of([heads as u32, 1, tokens, width]),
        Init::Zero,
        Element::Single,
    );
    let values = graph.parameter(
        Shape::of([heads as u32, 1, tokens, width]),
        Init::Zero,
        Element::Single,
    );
    let mask = graph.parameter(Shape::matrix(tokens, tokens), Init::Zero, Element::Single);
    let scores = graph.mul(
        graph.matmul(queries, graph.transpose(keys)),
        graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
    );
    let weighted = graph.softmax(graph.add(scores, mask));
    let out = graph.matmul(weighted, values);
    assert_eq!(out.shape(), Shape::of([heads as u32, 1, tokens, width]));
    graph.retain(out);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let elements = heads * tokens as usize * width as usize;
    let data = random(elements as u32 * 3, 41);
    let mask_values = (0..tokens * tokens)
        .map(|index| {
            if index % tokens <= index / tokens {
                0.0
            } else {
                -1.0e9
            }
        })
        .collect::<Vec<_>>();
    let (values_data, rest) = data.split_at(elements);
    let (keys_data, queries_data) = rest.split_at(elements);
    runtime.write(&program, queries, queries_data);
    runtime.write(&program, keys, keys_data);
    runtime.write(&program, values, values_data);
    runtime.write(&program, mask, &mask_values);
    runtime.run(&program);
    let scale = 1.0 / (width as f32).sqrt();
    let mut expected = Vec::new();
    for head in 0..heads {
        for row in 0..tokens as usize {
            let mut logits = Vec::new();
            for column in 0..tokens as usize {
                let at = (head * tokens as usize + row) * width as usize;
                let other = (head * tokens as usize + column) * width as usize;
                let dot = (0..width as usize)
                    .map(|step| queries_data[at + step] * keys_data[other + step])
                    .sum::<f32>();
                logits.push(dot * scale + mask_values[row * tokens as usize + column]);
            }
            let largest = logits.iter().copied().fold(f32::MIN, f32::max);
            let total = logits
                .iter()
                .map(|value| (value - largest).exp())
                .sum::<f32>();
            let probabilities = logits
                .iter()
                .map(|value| (value - largest).exp() / total)
                .collect::<Vec<_>>();
            for step in 0..width as usize {
                let mut sum = 0.0;
                for (column, probability) in probabilities.iter().enumerate() {
                    let at = (head * tokens as usize + column) * width as usize;
                    sum += probability * values_data[at + step];
                }
                expected.push(sum);
            }
        }
    }
    assert_close(&runtime.read(&program, out), &expected, 1e-5);
}

#[test]
fn a_bias_broadcasts_over_every_row() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(4, 8), Element::Single);
    let bias = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let scale = graph.parameter(Shape::scalar(), Init::Zero, Element::Single);
    let shifted = graph.add(data, bias);
    let scaled = graph.mul(shifted, scale);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data_values = random(32, 7);
    let bias_values = random(8, 13);
    runtime.write(&program, data, &data_values);
    runtime.write(&program, bias, &bias_values);
    runtime.write(&program, scale, &[3.0]);
    runtime.run(&program);
    let out = runtime.read(&program, scaled);
    let expected = data_values
        .iter()
        .enumerate()
        .map(|(index, value)| (value + bias_values[index % 8]) * 3.0)
        .collect::<Vec<_>>();
    assert_close(&out, &expected, 1e-5);
}

#[test]
fn a_softmax_row_sums_to_one() {
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(6, 9), Init::Zero, Element::Single);
    let probabilities = graph.softmax(logits);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let logits_values = random(54, 3);
    runtime.write(&program, logits, &logits_values);
    runtime.run(&program);
    let out = runtime.read(&program, probabilities);
    assert_close(&out, &softmax_reference(&logits_values, 9), 1e-5);
    for row in out.chunks(9) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }
}

#[test]
fn the_loss_gradient_of_a_matmul_is_the_column_sum_of_its_operand() {
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(4, 3), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(3, 5), Init::Zero, Element::Single);
    let loss = graph.sum(graph.matmul(left, right));
    let grads = graph.backward(loss);
    graph.retain(grads.of(left));
    graph.retain(grads.of(right));
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let left_data = random(12, 5);
    let right_data = random(15, 17);
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    let left_grad = runtime.read(&program, grads.of(left));
    let right_grad = runtime.read(&program, grads.of(right));
    let mut expected_left = vec![0.0f32; 12];
    for row in 0..4u32 {
        for column in 0..3u32 {
            expected_left[(row * 3 + column) as usize] = (0..5u32)
                .map(|other| right_data[(column * 5 + other) as usize])
                .sum();
        }
    }
    let mut expected_right = vec![0.0f32; 15];
    for step in 0..3u32 {
        for column in 0..5u32 {
            expected_right[(step * 5 + column) as usize] = (0..4u32)
                .map(|row| left_data[(row * 3 + step) as usize])
                .sum();
        }
    }
    assert_close(&left_grad, &expected_left, 1e-4);
    assert_close(&right_grad, &expected_right, 1e-4);
    let loss_value = runtime.read(&program, loss);
    let expected_loss = left_data
        .iter()
        .zip(expected_left.iter())
        .map(|(value, grad)| value * grad)
        .sum::<f32>();
    assert_close(&loss_value, &[expected_loss], 1e-3);
}

#[test]
fn a_bias_gradient_folds_every_row_it_was_added_to() {
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(5, 4), Element::Single);
    let bias = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let loss = graph.sum(graph.add(data, bias));
    let grads = graph.backward(loss);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &random(20, 23));
    runtime.run(&program);
    assert_close(&runtime.read(&program, grads.of(bias)), &[5.0; 4], 1e-5);
}

#[test]
fn a_rectifier_gradient_keeps_the_sign_of_its_input() {
    let graph = Graph::new();
    let data = graph.parameter(Shape::vector(8), Init::Zero, Element::Single);
    let loss = graph.sum(graph.relu(data));
    let grads = graph.backward(loss);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = vec![-2.0, -1.0, 0.0, 1.0, 2.0, -0.5, 0.5, 3.0];
    runtime.write(&program, data, &values);
    runtime.run(&program);
    let expected = values
        .iter()
        .map(|value| f32::from(*value > 0.0))
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, grads.of(data)), &expected, 1e-6);
}

#[test]
fn the_loss_gradient_of_a_softmax_row_vanishes() {
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(4, 6), Init::Zero, Element::Single);
    let loss = graph.sum(graph.softmax(logits));
    let grads = graph.backward(loss);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, logits, &random(24, 31));
    runtime.run(&program);
    let gradient = runtime.read(&program, grads.of(logits));
    assert_close(&gradient, &[0.0; 24], 1e-5);
}

#[test]
fn a_square_root_and_its_reciprocal_ride_the_same_tape() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let root = graph.sqrt(data);
    let reciprocal = graph.recip(root);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[1.0, 4.0, 9.0, 16.0]);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, reciprocal),
        &[1.0, 0.5, 1.0 / 3.0, 0.25],
        1e-6,
    );
}

#[test]
fn a_device_program_carries_every_tile_of_its_profile_whatever_the_batch() {
    let runtime = open();
    let mut menus: Vec<Vec<neura_profile::MatmulTile>> = Vec::new();
    let mut results = Vec::new();
    let mut programs = Vec::new();
    for batch in [8u32, 64, 8, 17, 512] {
        let graph = Graph::new();
        let weight = graph.parameter(
            Shape::matrix(5, 3),
            Init::Uniform {
                low: -0.5,
                high: 0.5,
            },
            Element::Single,
        );
        let data = graph.input(Shape::matrix(batch, 5), Element::Single);
        let out = graph.softmax(graph.matmul(data, weight));
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        if !menus.contains(&program.tiles().to_vec()) {
            menus.push(program.tiles().to_vec());
        }
        assert_eq!(
            program.tiles(),
            program.profile().tiles(),
            "a device program carries the whole menu of the profile it compiles for",
        );
        assert_eq!(
            runtime.declared_kernels(),
            menus.len(),
            "a device program is assembled once per menu, not once per shape",
        );
        let data_values = random(batch * 5, batch);
        runtime.write(&program, data, &data_values);
        runtime.run(&program);
        results.push(runtime.read(&program, out));
        programs.push((program, out));
    }
    assert_eq!(results[0].len(), 8 * 3);
    assert_eq!(results[1].len(), 64 * 3);
    assert_eq!(
        menus.len(),
        1,
        "every batch of one model walks the menu of one device program",
    );
    assert_eq!(
        runtime.assembled_kernels(),
        1,
        "a shape never assembles a second device program",
    );
    for (index, (program, out)) in programs.iter().enumerate() {
        runtime.run(program);
        assert_eq!(
            &runtime.read(program, *out),
            &results[index],
            "a program keeps the results of the shape it was written for",
        );
    }
}

#[test]
fn an_update_in_place_replays_on_every_run() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let update = graph.fill(Shape::vector(4), 0.25);
    graph.add_into(weight, update);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, weight, &[1.0; 4]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, weight), &[1.25; 4], 1e-6);
    runtime.run(&program);
    assert_close(&runtime.read(&program, weight), &[1.5; 4], 1e-6);
}

#[test]
fn a_write_lands_after_every_write_it_follows() {
    let graph = Graph::new();
    let state = graph.resident(Shape::vector(4), Element::Single);
    let data = graph.input(Shape::vector(4), Element::Single);
    let mut deep = graph.relu(data);
    for _ in 0..5 {
        deep = graph.relu(deep);
    }
    let shallow = graph.fill(Shape::vector(4), 7.0);
    graph.add_into(state, deep);
    graph.copy_into(state, shallow);
    let out = graph.relu(state);
    graph.retain(state);
    graph.retain(out);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[1.0; 4]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &[7.0; 4], 1e-6);
    assert_close(&runtime.read(&program, state), &[7.0; 4], 1e-6);
}

#[test]
fn a_chain_of_updates_rides_one_dispatch() {
    let graph = Graph::new();
    let source = graph.input(Shape::vector(2048), Element::Single);
    let mut value = source;
    let half = graph.fill(Shape::vector(2048), 0.5);
    for _ in 1..16 {
        value = graph.mul(value, half);
    }
    let out = graph.relu(value);
    graph.retain(out);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(
        program.dispatch_count(),
        1,
        "a chain of dependent work is dispatched once",
    );
    runtime.write(&program, source, &vec![0.5; 2048]);
    runtime.run(&program);
    let expected = 0.5f32.powi(16);
    assert_close(&runtime.read(&program, out), &[expected; 2048], 1e-6);
}

#[test]
fn a_tape_runs_a_whole_training_step_in_one_submission() {
    let graph = Graph::new();
    let weight = graph.parameter(
        Shape::matrix(4, 3),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let bias = graph.parameter(Shape::vector(3), Init::Zero, Element::Single);
    let data = graph.input(Shape::matrix(6, 4), Element::Single);
    let hidden = graph.relu(graph.add(graph.matmul(data, weight), bias));
    let loss = graph.sum(hidden);
    let grads = graph.backward(loss);
    let learning_rate = graph.fill(Shape::scalar(), -0.001);
    let weight_step = graph.mul(grads.of(weight), learning_rate);
    let bias_step = graph.mul(grads.of(bias), learning_rate);
    graph.add_into(weight, weight_step);
    graph.add_into(bias, bias_step);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &random(24, 41));
    runtime.run(&program);
    let first = (runtime.read(&program, loss), runtime.read(&program, weight));
    runtime.run(&program);
    let second = (runtime.read(&program, loss), runtime.read(&program, weight));
    assert!(
        second.0[0] < first.0[0],
        "a step of descent lowered the loss from {} to {}",
        first.0[0],
        second.0[0],
    );
    assert_ne!(second.1, first.1, "the step moved the weights");
    assert!(
        program.dispatch_count() < program.task_count(),
        "{} tasks were dispatched in {} dispatches",
        program.task_count(),
        program.dispatch_count(),
    );
}

#[test]
fn reading_two_tensors_costs_one_submission() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let doubled = graph.mul(data, graph.fill(Shape::vector(4), 2.0));
    let shifted = graph.add(doubled, graph.fill(Shape::vector(4), 1.0));
    graph.retain(doubled);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[1.0, 2.0, 3.0, 4.0]);
    runtime.run(&program);
    let values = runtime.read_many(&program, &[doubled, shifted]);
    assert_close(&values[0], &[2.0, 4.0, 6.0, 8.0], 1e-6);
    assert_close(&values[1], &[3.0, 5.0, 7.0, 9.0], 1e-6);
}

#[test]
fn a_readout_holds_the_run_it_was_pulled_from() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let doubled = graph.mul(data, graph.fill(Shape::vector(4), 2.0));
    graph.retain(doubled);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[1.0, 2.0, 3.0, 4.0]);
    runtime.run(&program);
    let first = runtime.pull(&program, &[doubled]);
    runtime.write(&program, data, &[5.0, 6.0, 7.0, 8.0]);
    runtime.run(&program);
    assert_close(
        &runtime.collect(first).pop().expect("one tensor came back"),
        &[2.0, 4.0, 6.0, 8.0],
        1e-6,
    );
    assert_close(
        &runtime.read(&program, doubled),
        &[10.0, 12.0, 14.0, 16.0],
        1e-6,
    );
}

#[test]
fn a_pull_without_a_collect_runs_out_of_readbacks() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let out = graph.mul(data, graph.fill(Shape::vector(4), 1.0));
    graph.retain(out);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let mut pulled = Vec::new();
    for slot in 0..runtime.readback_slots() {
        runtime.run(&program);
        pulled.push((slot, runtime.pull(&program, &[out])));
    }
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.pull(&program, &[out])
    }));
    assert!(
        outcome.is_err(),
        "every readback of the runtime was in flight, and a pull was accepted",
    );
    let (slot, readout) = pulled.pop().expect("a readback was pulled");
    runtime.collect(readout);
    let next = runtime.pull(&program, &[out]);
    runtime.collect(next);
    assert!(slot < runtime.readback_slots());
}

#[test]
fn a_graph_without_tasks_is_refused_by_the_runtime() {
    let graph = Graph::new();
    graph.input(Shape::vector(4), Element::Single);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.compile(&graph, &weights)
    }));
    assert!(
        outcome.is_err(),
        "a program with an empty tape was compiled"
    );
}

#[test]
fn a_tensor_wider_than_the_staging_buffer_is_refused_by_a_read() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4096), Element::Single);
    let out = graph.mul(data, graph.fill(Shape::vector(4096), 1.0));
    let runtime = pollster::block_on(Runtime::open(neura_runtime::RuntimeRequest {
        readback_bytes: 256,
        ..Default::default()
    }))
    .expect("a device with a small staging buffer");
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| runtime.read(&program, out)));
    assert!(
        outcome.is_err(),
        "a read wider than the staging buffer was accepted"
    );
}

#[test]
fn a_reclaimed_temporary_is_refused_and_a_retained_one_reads_back() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let scaled = graph.mul(data, graph.fill(Shape::vector(4), 2.0));
    let squared = graph.mul(scaled, scaled);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[1.0, 2.0, 3.0, 4.0]);
    runtime.run(&program);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.read(&program, scaled)
    }));
    assert!(
        refused.is_err(),
        "a temporary whose storage a later task took over was handed back",
    );
    assert_close(
        &runtime.read(&program, squared),
        &[4.0, 16.0, 36.0, 64.0],
        1e-6,
    );
}

#[test]
fn a_retained_gradient_reads_back_after_the_step_that_consumed_it() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Constant(1.0), Element::Single);
    let data = graph.input(Shape::vector(4), Element::Single);
    let loss = graph.sum(graph.mul(data, weight));
    let grads = graph.backward(loss);
    let gradient = grads.of(weight);
    graph.retain(gradient);
    let step = graph.mul(gradient, graph.fill(Shape::vector(4), -0.5));
    graph.add_into(weight, step);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[1.0, 2.0, 3.0, 4.0]);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, gradient),
        &[1.0, 2.0, 3.0, 4.0],
        1e-6,
    );
    assert_close(
        &runtime.read(&program, weight),
        &[0.5, 0.0, -0.5, -1.0],
        1e-6,
    );
    assert_close(&runtime.read(&program, loss), &[10.0], 1e-5);
}

#[test]
fn a_parameter_read_before_any_run_holds_its_seed() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Constant(2.5), Element::Single);
    let out: Value = graph.relu(weight);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_close(&runtime.read(&program, weight), &[2.5; 4], 1e-6);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &[2.5; 4], 1e-6);
}

#[test]
fn a_program_binds_exactly_the_memory_its_tape_lays_out() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(1024), Element::Single);
    let out = graph.relu(graph.mul(data, data));
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[2.0; 1024]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out)[..4], &[4.0; 4], 1e-6);
    assert_eq!(program.arena_bytes(), 8192);
    assert_eq!(program.tensor_bytes(), program.arena_bytes());
    assert_eq!(program.heap_bytes(), runtime.heap_bytes());
    assert!(
        program.device_bytes() >= program.tensor_bytes(),
        "a tape carries {} bytes of device memory beside its {} byte tensors",
        program.device_bytes(),
        program.tensor_bytes(),
    );
}

#[test]
fn two_programs_of_one_model_share_their_weights_and_hold_their_own_arena() {
    let runtime = open();
    let first = Graph::new();
    let weight = first.parameter(Shape::vector(4), Init::Constant(2.5), Element::Single);
    let scaled = first.mul(weight, first.fill(Shape::vector(4), 3.0));
    let weights = runtime.weights(&first);
    let a = runtime.compile(&first, &weights);
    runtime.run(&a);
    assert_close(&runtime.read(&a, scaled), &[7.5; 4], 1e-6);

    let second = Graph::new();
    let shared = second.parameter(Shape::vector(4), Init::Constant(2.5), Element::Single);
    let data = second.input(Shape::vector(16), Element::Single);
    let doubled = second.mul(data, second.fill(Shape::vector(16), 2.0));
    let b = runtime.compile(&second, &weights);
    runtime.write(&b, data, &[1.0; 16]);
    runtime.run(&b);
    assert_close(&runtime.read(&b, doubled), &[2.0; 16], 1e-6);

    runtime.write(&a, weight, &[8.0; 4]);
    assert_close(&runtime.read(&b, shared), &[8.0; 4], 1e-6);

    runtime.run(&a);
    assert_close(&runtime.read(&a, scaled), &[24.0; 4], 1e-6);
    assert_ne!(
        a.span(scaled).offset,
        b.span(doubled).offset,
        "two programs of one model share one tensor region",
    );
    assert_eq!(
        a.weights().offset(),
        b.weights().offset(),
        "two programs of one model hold two weight stores",
    );
    assert_eq!(
        a.heap().allocation(),
        b.heap().allocation(),
        "two programs of one runtime were handed two heaps",
    );
    assert!(
        a.arena_bytes() < b.arena_bytes(),
        "a plan of {} bytes reached as wide as a plan of {} bytes",
        a.arena_bytes(),
        b.arena_bytes(),
    );
}

#[test]
fn a_fresh_program_holds_zeros_until_the_host_writes() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let out = graph.mul(data, graph.fill(Shape::vector(4), 2.0));
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &[0.0; 4], 1e-6);
    runtime.write(&program, data, &[1.0, 2.0, 3.0, 4.0]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &[2.0, 4.0, 6.0, 8.0], 1e-6);
}

#[test]
fn every_profile_the_device_offers_runs_the_same_matmul() {
    let runtime = open();
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(37, 19), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(19, 43), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    let transposed = graph.matmul(graph.transpose(right), graph.transpose(left));
    let left_data = random(37 * 19, 11);
    let right_data = random(19 * 43, 29);
    let expected = matmul_reference(&left_data, &right_data, 37, 19, 43);
    let mut transposed_right = vec![0.0f32; 19 * 43];
    for row in 0..19usize {
        for column in 0..43usize {
            transposed_right[column * 19 + row] = right_data[row * 43 + column];
        }
    }
    let mut transposed_left = vec![0.0f32; 37 * 19];
    for row in 0..37usize {
        for column in 0..19usize {
            transposed_left[column * 37 + row] = left_data[row * 19 + column];
        }
    }
    let expected_transposed = matmul_reference(&transposed_right, &transposed_left, 43, 19, 37);
    for profile in runtime.profiles() {
        let weights = runtime.weights(&graph);
        let program = runtime.compile_with(&graph, &weights, profile);
        assert_eq!(program.profile(), profile);
        runtime.write(&program, left, &left_data);
        runtime.write(&program, right, &right_data);
        runtime.run(&program);
        assert_close(&runtime.read(&program, out), &expected, 1e-4);
        assert_close(
            &runtime.read(&program, transposed),
            &expected_transposed,
            1e-4,
        );
    }
}

#[test]
fn every_tile_of_a_profile_runs_its_own_matmul() {
    let runtime = open();
    let profile = runtime.default_profile();
    let tiles = profile.tiles();
    for index in 0..tiles.len() {
        let tile = tiles[index];
        let profile = neura_profile::Profile::of(&tiles[index..index + 1]);
        let graph = Graph::new();
        let left = graph.parameter(
            Shape::matrix(tile.rows(), tile.depth()),
            Init::Zero,
            Element::Single,
        );
        let right = graph.parameter(
            Shape::matrix(tile.depth(), tile.columns()),
            Init::Zero,
            Element::Single,
        );
        let out = graph.matmul(left, right);
        let weights = runtime.weights(&graph);
        let program = runtime.compile_with(&graph, &weights, profile);
        assert_eq!(program.tiles(), &[tile]);
        assert_eq!(
            program.matmul_geometries(),
            vec![(tile, 1)],
            "a product of {}x{} is not tiled by its own geometry",
            tile.rows(),
            tile.columns(),
        );
        let left_data = random(tile.rows() * tile.depth(), 5);
        let right_data = random(tile.depth() * tile.columns(), 9);
        let expected = matmul_reference(
            &left_data,
            &right_data,
            tile.rows(),
            tile.depth(),
            tile.columns(),
        );
        runtime.write(&program, left, &left_data);
        runtime.write(&program, right, &right_data);
        runtime.run(&program);
        assert_close(&runtime.read(&program, out), &expected, 1e-4);
    }
}

#[test]
fn a_product_that_splits_its_depth_keeps_the_menu_of_one_program() {
    let runtime = open();
    for depth in [8u32, 4096] {
        let graph = Graph::new();
        let left = graph.parameter(Shape::matrix(8, depth), Init::Zero, Element::Single);
        let right = graph.parameter(Shape::matrix(depth, 32), Init::Zero, Element::Single);
        graph.retain(graph.matmul(left, right));
        let weights = runtime.weights(&graph);
        runtime.compile(&graph, &weights);
    }
    assert_eq!(
        runtime.assembled_kernels(),
        1,
        "a product that folds its depth walks the program its shape without a fold walks",
    );
}

#[test]
fn a_device_pool_of_sixteen_kibibytes_drops_the_widest_profile() {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        gpu: neura_gpu::GpuRequest::default().minimum_limits(),
        readback_bytes: 1 << 16,
        ..Default::default()
    }))
    .expect("a device with the baseline pool");
    let profiles = runtime.profiles();
    assert!(!profiles.is_empty(), "the baseline pool fits no profile");
    assert!(
        profiles
            .iter()
            .all(|profile| profile.shared_bytes() <= 16 * 1024),
        "a profile asks for more than the baseline pool",
    );
    assert!(
        profiles.len() < Profile::derive(Budget::of(1024, 48 << 10)).len(),
        "the baseline pool must drop a profile the wide pool keeps",
    );
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 4), Init::Zero, Element::Single);
    let data = graph.input(Shape::matrix(8, 4), Element::Single);
    let out = graph.matmul(data, weight);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(program.profile(), *profiles.last().expect("a profile"));
    assert!(
        refuses(|| {
            let widest = *Profile::derive(Budget::of(1024, 48 << 10))
                .last()
                .expect("a profile");
            let _ = runtime.compile_with(&graph, &weights, widest);
        }),
        "a profile the device cannot hold was compiled",
    );
    runtime.run(&program);
    assert_eq!(runtime.read(&program, out).len(), 32);
}

#[test]
fn tuning_measures_every_profile_the_device_offers() {
    let runtime = open();
    let graph = Graph::new();
    let left = graph.parameter(Shape::matrix(64, 32), Init::Zero, Element::Single);
    let right = graph.parameter(Shape::matrix(32, 64), Init::Zero, Element::Single);
    let out = graph.matmul(left, right);
    let left_data = random(64 * 32, 3);
    let right_data = random(32 * 64, 7);
    let expected = matmul_reference(&left_data, &right_data, 64, 32, 64);
    let weights = runtime.weights(&graph);
    let program = runtime.tune(&graph, &weights);
    assert!(
        runtime.profiles().contains(&program.profile()),
        "a tuned program carries a profile the device offers",
    );
    assert_eq!(
        runtime.declared_kernels(),
        runtime.profiles().len(),
        "tuning declares one device program per profile it measures",
    );
    runtime.write(&program, left, &left_data);
    runtime.write(&program, right, &right_data);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &expected, 1e-4);
}

#[test]
fn an_operand_folded_into_a_subtraction_keeps_its_side() {
    let graph = Graph::new();
    let left = graph.input(Shape::vector(4), Element::Single);
    let right = graph.input(Shape::vector(4), Element::Single);
    let bias = graph.input(Shape::vector(4), Element::Single);
    let difference = graph.sub(bias, graph.mul(left, right));
    let quotient = graph.div(bias, graph.mul(left, right));
    graph.retain(difference);
    graph.retain(quotient);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert_eq!(
        program.task_count(),
        2,
        "each product rides the operand it was folded into",
    );
    runtime.write(&program, left, &[1.0, 2.0, 3.0, 4.0]);
    runtime.write(&program, right, &[0.5, 0.25, -1.0, 2.0]);
    runtime.write(&program, bias, &[10.0, 20.0, 30.0, 40.0]);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, difference),
        &[9.5, 19.5, 33.0, 32.0],
        1e-6,
    );
    assert_close(
        &runtime.read(&program, quotient),
        &[20.0, 40.0, -10.0, 5.0],
        1e-6,
    );
}

#[test]
fn a_log_softmax_row_holds_its_log_probabilities() {
    let graph = Graph::new();
    let logits = graph.parameter(Shape::matrix(6, 9), Init::Zero, Element::Single);
    let log_probabilities = graph.log_softmax(logits);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let logits_values = random(54, 3);
    runtime.write(&program, logits, &logits_values);
    runtime.run(&program);
    let out = runtime.read(&program, log_probabilities);
    assert_close(&out, &log_softmax_reference(&logits_values, 9), 1e-5);
    for row in out.chunks(9) {
        assert!(
            (row.iter().map(|value| value.exp()).sum::<f32>() - 1.0).abs() < 1e-5,
            "a log probability row exponentiates to one",
        );
    }
}

#[test]
fn a_log_softmax_rides_the_shape_of_its_rows() {
    let runtime = open();
    let mut widths = Vec::new();
    for columns in [4u32, 64] {
        let graph = Graph::new();
        let logits = graph.parameter(Shape::matrix(3, columns), Init::Zero, Element::Single);
        let out = graph.log_softmax(logits);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let data = random(3 * columns, columns);
        runtime.write(&program, logits, &data);
        runtime.run(&program);
        assert_close(
            &runtime.read(&program, out),
            &log_softmax_reference(&data, columns),
            1e-5,
        );
        widths.push(runtime.read(&program, out).len());
        assert_eq!(runtime.declared_kernels(), 1);
    }
    assert_eq!(widths, vec![12, 192]);
}

#[test]
fn a_view_reads_back_through_the_strides_it_was_transposed_to() {
    let runtime = open();
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(3, 5), Element::Single);
    let table = graph.parameter(Shape::matrix(5, 3), Init::Zero, Element::Single);
    let shifted = graph.add(data, graph.transpose(table));
    graph.retain(shifted);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data_values = random(15, 5);
    let table_values = random(15, 11);
    runtime.write(&program, data, &data_values);
    runtime.write(&program, table, &table_values);
    runtime.run(&program);
    let expected = (0..15)
        .map(|index| {
            let row = index / 5;
            let column = index % 5;
            data_values[index] + table_values[column * 3 + row]
        })
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, shifted), &expected, 1e-6);
}

#[test]
fn a_tensor_of_every_rank_reads_the_row_it_broadcasts() {
    let runtime = open();
    let graph = Graph::new();
    for dims in [vec![2u32, 3, 4], vec![2, 3, 4, 5]] {
        let shape = Shape::of(&dims);
        let elements = dims.iter().product::<u32>();
        let columns = *dims.last().expect("a last axis");
        let data = graph.input(shape, Element::Single);
        let bias = graph.parameter(Shape::vector(columns), Init::Zero, Element::Single);
        let shifted = graph.add(data, bias);
        graph.retain(shifted);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let data_values = random(elements, elements);
        let bias_values = random(columns, columns + 7);
        runtime.write(&program, data, &data_values);
        runtime.write(&program, bias, &bias_values);
        runtime.run(&program);
        let expected = data_values
            .iter()
            .enumerate()
            .map(|(index, value)| value + bias_values[index % columns as usize])
            .collect::<Vec<_>>();
        assert_close(&runtime.read(&program, shifted), &expected, 1e-6);
    }
}

#[test]
fn a_task_reads_a_leaf_before_the_task_that_rewrites_it() {
    let graph = Graph::new();
    let state = graph.resident(Shape::vector(4), Element::Single);
    let bias = graph.input(Shape::vector(4), Element::Single);
    let read = graph.mul(state, bias);
    let patch = graph.fill(Shape::vector(4), 7.0);
    graph.copy_into(state, patch);
    let out = graph.relu(read);
    graph.retain(out);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, bias, &[1.0, 2.0, 3.0, 4.0]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &[0.0; 4], 1e-6);
    assert_close(&runtime.read(&program, state), &[7.0; 4], 1e-6);
}

#[test]
fn a_task_reads_a_parameter_before_the_task_that_updates_it() {
    let graph = Graph::new();
    let weight = graph.parameter(Shape::vector(4), Init::Constant(0.5), Element::Single);
    let bias = graph.input(Shape::vector(4), Element::Single);
    let read = graph.mul(weight, bias);
    let step = graph.fill(Shape::vector(4), 1.0);
    graph.add_into(weight, step);
    let out = graph.relu(read);
    graph.retain(out);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, bias, &[1.0, 2.0, 3.0, 4.0]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &[0.5, 1.0, 1.5, 2.0], 1e-6);
    assert_close(&runtime.read(&program, weight), &[1.5; 4], 1e-6);
}

#[test]
fn a_storage_a_view_reads_keeps_the_task_that_writes_it() {
    let graph = Graph::new();
    let left = graph.input(Shape::matrix(2, 3), Element::Single);
    let right = graph.input(Shape::matrix(2, 3), Element::Single);
    let product = graph.mul(left, right);
    let flipped = graph.transpose(product);
    let doubled = graph.add(product, graph.fill(Shape::matrix(2, 3), 1.0));
    let columns = graph.sum_rows(flipped);
    graph.retain(doubled);
    graph.retain(columns);
    let runtime = open();
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let left_values = random(6, 3);
    let right_values = random(6, 5);
    runtime.write(&program, left, &left_values);
    runtime.write(&program, right, &right_values);
    runtime.run(&program);
    let products = left_values
        .iter()
        .zip(&right_values)
        .map(|(left, right)| left * right)
        .collect::<Vec<_>>();
    let expected = products.iter().map(|value| value + 1.0).collect::<Vec<_>>();
    assert_close(&runtime.read(&program, doubled), &expected, 1e-6);
    let expected = (0..3)
        .map(|column| (0..2).map(|row| products[row * 3 + column]).sum::<f32>())
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, columns), &expected, 1e-6);
}
