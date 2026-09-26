use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};

#[path = "support/decision.rs"]
mod decision;
#[path = "support/reference.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use decision::{
    argmax_reference, counts, gather_reference, gumbel_reference, one_hot_reference,
    scatter_reference,
};
use reference::{matmul_reference, random};
use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn tied(count: u32, seed: u32) -> Vec<f32> {
    random(count, seed)
        .into_iter()
        .map(|value| (value * 4.0).round() / 4.0)
        .collect()
}

#[test]
fn every_row_picks_the_largest_element_it_holds() {
    let runtime = open();
    let rows = 257;
    let classes = 100;
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(rows, classes), Element::Single);
    let action = graph.argmax(logits);
    graph.retain(action);
    let weights = runtime.weights(&graph);
    let data = tied(rows * classes, 7);
    let expected = argmax_reference(&data, rows, classes);
    for profile in runtime.profiles() {
        let program = runtime.compile_with(&graph, &weights, profile);
        runtime.write(&program, logits, &data);
        runtime.run(&program);
        assert_eq!(
            runtime.read(&program, action),
            expected,
            "the {} row strategy of {profile:?} picks another element",
            classes,
        );
    }
}

#[test]
fn a_row_wider_than_the_workgroup_folds_with_it() {
    let runtime = open();
    let rows = 5;
    let classes = 1000;
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(rows, classes), Element::Single);
    let action = graph.argmax(logits);
    graph.retain(action);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let mut data = tied(rows * classes, 21);
    data[3 * classes as usize + 501] = 100.0;
    data[3 * classes as usize + 999] = 100.0;
    runtime.write(&program, logits, &data);
    runtime.run(&program);
    let expected = argmax_reference(&data, rows, classes);
    assert_eq!(expected[3], 501.0, "a tie keeps the first of its maxima");
    assert_eq!(runtime.read(&program, action), expected);
}

#[test]
fn a_row_of_one_class_picks_that_class() {
    let runtime = open();
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(9, 1), Element::Single);
    let action = graph.argmax(logits);
    graph.retain(action);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, logits, &tied(9, 3));
    runtime.run(&program);
    assert_eq!(runtime.read(&program, action), vec![0.0; 9]);
}

#[test]
fn an_action_carries_the_index_of_its_row() {
    let runtime = open();
    let rows = 128;
    let classes = 64;
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(rows, classes), Element::Single);
    let action = graph.argmax(logits);
    let mask = graph.one_hot(action, classes);
    let picked = graph.gather(logits, action);
    graph.retain(mask);
    graph.retain(picked);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data = tied(rows * classes, 33);
    runtime.write(&program, logits, &data);
    runtime.run(&program);
    let expected = argmax_reference(&data, rows, classes);
    assert_close(
        &runtime.read(&program, mask),
        &one_hot_reference(&expected, classes),
        0.0,
    );
    let picked_rows = expected
        .iter()
        .flat_map(|row| {
            &data[*row as usize * classes as usize..(*row as usize + 1) * classes as usize]
        })
        .copied()
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, picked), &picked_rows, 0.0);
}

#[test]
fn an_index_list_lights_the_class_it_names() {
    let runtime = open();
    let rows = 6;
    let classes = 4;
    let graph = Graph::new();
    let indices = graph.input(Shape::matrix(rows, 1), Element::Single);
    let mask = graph.one_hot(indices, classes);
    graph.retain(mask);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let index_data = vec![0.0, 3.0, 1.0, 3.0, 2.0, 0.0];
    runtime.write(&program, indices, &index_data);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, mask),
        &one_hot_reference(&index_data, classes),
        0.0,
    );
}

#[test]
fn a_gather_copies_the_rows_its_index_list_names() {
    let runtime = open();
    let rows = 5;
    let width = 3;
    let picks = 7;
    let graph = Graph::new();
    let table = graph.input(Shape::matrix(rows, width), Element::Single);
    let indices = graph.input(Shape::matrix(picks, 1), Element::Single);
    let picked = graph.gather(table, indices);
    graph.retain(picked);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let table_data = random(rows * width, 5);
    let index_data = vec![4.0, 0.0, 4.0, 2.0, 1.0, 0.0, 3.0];
    runtime.write(&program, table, &table_data);
    runtime.write(&program, indices, &index_data);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, picked),
        &gather_reference(&table_data, &index_data, width),
        0.0,
    );
}

#[test]
fn a_gather_walks_a_table_of_every_rank() {
    let runtime = open();
    let table_dims = [2u32, 3, 2, 4];
    let rows = table_dims.iter().product::<u32>() / table_dims[3];
    let width = table_dims[3];
    let graph = Graph::new();
    let table = graph.input(Shape::of(table_dims), Element::Single);
    let indices = graph.input(Shape::matrix(3, 1), Element::Single);
    let picked = graph.gather(table, indices);
    graph.retain(picked);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let table_data = random(rows * width, 23);
    let index_data = vec![11.0, 0.0, 7.0];
    runtime.write(&program, table, &table_data);
    runtime.write(&program, indices, &index_data);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, picked),
        &gather_reference(&table_data, &index_data, width),
        0.0,
    );
}

#[test]
fn an_index_outside_its_table_stops_the_read() {
    let runtime = open();
    let graph = Graph::new();
    let indices = graph.input(Shape::matrix(3, 1), Element::Single);
    let mask = graph.one_hot(indices, 4);
    graph.retain(mask);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    for index in [4.0, -1.0, 1.5] {
        runtime.write(&program, indices, &[0.0, index, 2.0]);
        runtime.run(&program);
        assert!(
            refuses(|| {
                drop(runtime.read(&program, mask));
            }),
            "an index of {index} lights a class nothing holds",
        );
    }
}

#[test]
fn a_categorical_draw_follows_the_logits_it_was_handed() {
    let runtime = open();
    let rows = 512;
    let classes = 4;
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(rows, classes), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let draw = graph.categorical(logits, seed);
    graph.retain(draw);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let probabilities = [0.5f32, 0.25, 0.125, 0.125];
    let logit_data = (0..rows)
        .flat_map(|_| probabilities.map(f32::ln))
        .collect::<Vec<_>>();
    runtime.write(&program, logits, &logit_data);

    let mut drawn = Vec::new();
    for seed_value in [f32::from_bits(0), f32::from_bits(1), f32::from_bits(7)] {
        runtime.write(&program, seed, &[seed_value]);
        runtime.run(&program);
        let action = runtime.read(&program, draw);
        let counted = counts(&action, classes);
        for (class, count) in counted.iter().enumerate() {
            let expected = probabilities[class] * rows as f32;
            assert!(
                (*count as f32 - expected).abs() <= rows as f32 * 0.08,
                "seed {seed_value} drew {count} of class {class} where {expected} were due: {counted:?}",
            );
        }
        if let Some(previous) = drawn.last() {
            assert_ne!(&action, previous, "two seeds draw two action lists");
        }
        drawn.push(action);
    }
    runtime.write(&program, seed, &[f32::from_bits(0)]);
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, draw),
        drawn[0],
        "a seeded draw repeats on every run",
    );
}

#[test]
fn a_draw_picks_the_largest_logit_its_seeded_noise_hands_it() {
    let runtime = open();
    let rows = 512;
    let classes = 2;
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(rows, classes), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let draw = graph.categorical(logits, seed);
    graph.retain(draw);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let logit_data = (0..rows).flat_map(|_| [0.0f32, 0.4]).collect::<Vec<_>>();
    runtime.write(&program, logits, &logit_data);
    for raw in [0u32, 3, 19] {
        runtime.write(&program, seed, &[f32::from_bits(raw)]);
        runtime.run(&program);
        let expected = (0..rows)
            .map(|row| {
                let base = row * classes;
                let left = logit_data[base as usize] + gumbel_reference(raw, base);
                let right = logit_data[base as usize + 1] + gumbel_reference(raw, base + 1);
                if right > left { 1.0 } else { 0.0 }
            })
            .collect::<Vec<_>>();
        assert_eq!(
            runtime.read(&program, draw),
            expected,
            "seed {raw} draws the noise the device makes",
        );
    }
}

#[test]
fn a_seeded_draw_rides_every_profile_the_same_way() {
    let runtime = open();
    let rows = 64;
    let classes = 100;
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(rows, classes), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let draw = graph.categorical(logits, seed);
    graph.retain(draw);
    let weights = runtime.weights(&graph);
    let data = tied(rows * classes, 9);
    let mut drawn = Vec::new();
    for profile in runtime.profiles() {
        let program = runtime.compile_with(&graph, &weights, profile);
        runtime.write(&program, logits, &data);
        runtime.write(&program, seed, &[f32::from_bits(11)]);
        runtime.run(&program);
        drawn.push(runtime.read(&program, draw));
    }
    for pair in drawn.windows(2) {
        assert_eq!(
            pair[0], pair[1],
            "a seeded draw lies on the row schedule a profile hands it",
        );
    }
}

#[test]
fn a_dominated_class_leaves_every_draw_to_its_winner() {
    let runtime = open();
    let rows = 64;
    let classes = 3;
    let graph = Graph::new();
    let logits = graph.input(Shape::matrix(rows, classes), Element::Single);
    let seed = graph.input(Shape::scalar(), Element::Single);
    let draw = graph.categorical(logits, seed);
    graph.retain(draw);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let logit_data = (0..rows)
        .flat_map(|_| [40.0f32, 0.0, 0.0])
        .collect::<Vec<_>>();
    runtime.write(&program, logits, &logit_data);
    runtime.write(&program, seed, &[f32::from_bits(5)]);
    runtime.run(&program);
    assert_eq!(runtime.read(&program, draw), vec![0.0; rows as usize]);
}

#[test]
fn a_table_learns_through_the_one_hot_product_of_its_index_list() {
    let runtime = open();
    let classes = 4;
    let width = 3;
    let picks = 5;
    let graph = Graph::new();
    let table = graph.parameter(Shape::matrix(classes, width), Init::Zero, Element::Single);
    let indices = graph.input(Shape::matrix(picks, 1), Element::Single);
    let picked = graph.matmul(graph.one_hot(indices, classes), table);
    let loss = graph.sum(picked);
    let gradients = graph.backward(loss);
    let table_gradient = gradients.of(table);
    graph.retain(picked);
    graph.retain(table_gradient);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let table_data = random(classes * width, 17);
    let index_data = vec![1.0, 3.0, 1.0, 0.0, 1.0];
    runtime.write(&program, table, &table_data);
    runtime.write(&program, indices, &index_data);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, picked),
        &gather_reference(&table_data, &index_data, width),
        1e-6,
    );
    let counted = counts(&index_data, classes);
    let expected = counted
        .iter()
        .flat_map(|count| vec![*count as f32; width as usize])
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, table_gradient), &expected, 0.0);
}

#[test]
fn a_gather_walks_its_gradient_back_into_the_table_it_reads() {
    let runtime = open();
    let classes = 4;
    let width = 3;
    let picks = 5;
    let graph = Graph::new();
    let table = graph.parameter(Shape::matrix(classes, width), Init::Zero, Element::Single);
    let indices = graph.input(Shape::matrix(picks, 1), Element::Single);
    let picked = graph.gather(table, indices);
    let loss = graph.sum(picked);
    let gradients = graph.backward(loss);
    let table_gradient = gradients.of(table);
    graph.retain(picked);
    graph.retain(table_gradient);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let table_data = random(classes * width, 17);
    let index_data = vec![1.0, 3.0, 1.0, 0.0, 1.0];
    runtime.write(&program, table, &table_data);
    runtime.write(&program, indices, &index_data);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, picked),
        &gather_reference(&table_data, &index_data, width),
        1e-6,
    );
    let counted = counts(&index_data, classes);
    let expected = counted
        .iter()
        .flat_map(|count| vec![*count as f32; width as usize])
        .collect::<Vec<_>>();
    assert_close(&runtime.read(&program, table_gradient), &expected, 0.0);
}

#[test]
fn a_scatter_accumulates_updates_into_the_rows_it_names() {
    let runtime = open();
    let classes = 4;
    let width = 3;
    let picks = 5;
    let graph = Graph::new();
    let table = graph.resident(Shape::matrix(classes, width), Element::Single);
    let indices = graph.input(Shape::matrix(picks, 1), Element::Single);
    let updates = graph.input(Shape::matrix(picks, width), Element::Single);
    graph.scatter_into(table, indices, updates);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let table_data = random(classes * width, 19);
    let index_data = vec![1.0, 3.0, 1.0, 0.0, 1.0];
    let update_data = random(picks * width, 23);
    runtime.write(&program, table, &table_data);
    runtime.write(&program, indices, &index_data);
    runtime.write(&program, updates, &update_data);
    runtime.run(&program);
    let scattered = scatter_reference(&table_data, &index_data, &update_data, width);
    assert_close(&runtime.read(&program, table), &scattered, 0.0);
    runtime.run(&program);
    let doubled = scatter_reference(&scattered, &index_data, &update_data, width);
    assert_close(&runtime.read(&program, table), &doubled, 0.0);
}

#[test]
fn a_policy_picks_its_action_on_the_device() {
    let runtime = open();
    let samples = 6;
    let observations = 3;
    let actions = 4;
    let graph = Graph::new();
    let observed = graph.input(Shape::matrix(samples, observations), Element::Single);
    let weight = graph.parameter(
        Shape::matrix(observations, actions),
        Init::Zero,
        Element::Single,
    );
    let bias = graph.parameter(Shape::vector(actions), Init::Zero, Element::Single);
    let logits = graph.add(graph.matmul(observed, weight), bias);
    let action = graph.argmax(logits);
    graph.retain(action);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observation_data = random(samples * observations, 41);
    let weight_data = random(observations * actions, 43);
    let bias_data = vec![0.0, 1.0, 2.0, 3.0];
    runtime.write(&program, observed, &observation_data);
    runtime.write(&program, weight, &weight_data);
    runtime.write(&program, bias, &bias_data);
    runtime.run(&program);
    let mut expected_logits = matmul_reference(
        &observation_data,
        &weight_data,
        samples,
        observations,
        actions,
    );
    for (index, value) in expected_logits.iter_mut().enumerate() {
        *value += bias_data[index % actions as usize];
    }
    assert_eq!(
        runtime.read(&program, action),
        argmax_reference(&expected_logits, samples, actions),
    );
    assert!(program.readable(action));
    assert_eq!(program.span(action).elements, samples);
}
