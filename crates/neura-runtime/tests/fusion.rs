use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

fn samples(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    Init::Uniform {
        low: -1.5,
        high: 1.5,
    }
    .samples(count, &mut entropy)
}

#[test]
fn an_opened_sum_matches_the_sum_it_replaced() {
    let runtime = open();
    let values = samples(5000, 7);
    let mut tasks = Vec::new();
    let mut observed = Vec::new();
    for pinned in [true, false] {
        let graph = Graph::new();
        let data = graph.input(Shape::vector(5000), Element::Single);
        let activated = graph.exp(data);
        if pinned {
            graph.retain(activated);
        }
        let loss = graph.sum(activated);
        graph.retain(loss);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        tasks.push(program.task_count());
        runtime.write(&program, data, &values);
        runtime.run(&program);
        observed.push(runtime.read(&program, loss)[0]);
    }
    assert!(tasks[1] < tasks[0], "the opened sum holds {tasks:?} tasks");
    assert_close(&observed[1..], &observed[..1], 1e-3);
}

#[test]
fn an_opened_sum_folds_a_product_of_two_operands() {
    let runtime = open();
    let values = samples(5000, 13);
    let mut tasks = Vec::new();
    let mut observed = Vec::new();
    for pinned in [true, false] {
        let graph = Graph::new();
        let data = graph.input(Shape::vector(5000), Element::Single);
        let scale = graph.parameter(Shape::vector(5000), Init::Constant(0.5), Element::Single);
        let scaled = graph.mul(data, scale);
        if pinned {
            graph.retain(scaled);
        }
        let loss = graph.sum(graph.relu(scaled));
        graph.retain(loss);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        tasks.push(program.task_count());
        runtime.write(&program, data, &values);
        runtime.run(&program);
        observed.push(runtime.read(&program, loss)[0]);
    }
    assert!(tasks[1] < tasks[0], "the opened sum holds {tasks:?} tasks");
    assert_close(&observed[1..], &observed[..1], 1e-3);
}

#[test]
fn an_opened_row_fold_matches_the_rows_it_replaced() {
    let runtime = open();
    for columns in [16u32, 600] {
        let values = samples(7 * columns, 11);
        let mut tasks = Vec::new();
        let mut observed = Vec::new();
        for pinned in [true, false] {
            let graph = Graph::new();
            let data = graph.input(Shape::matrix(7, columns), Element::Single);
            let squared = graph.mul(data, data);
            if pinned {
                graph.retain(squared);
            }
            let rows = graph.sum_rows(squared);
            graph.retain(rows);
            let weights = runtime.weights(&graph);
            let program = runtime.compile(&graph, &weights);
            tasks.push(program.task_count());
            runtime.write(&program, data, &values);
            runtime.run(&program);
            observed.push(runtime.read(&program, rows));
        }
        assert!(
            tasks[1] < tasks[0],
            "the opened fold of {columns} columns holds {tasks:?} tasks",
        );
        let expected = values
            .chunks(columns as usize)
            .map(|row| row.iter().map(|value| value * value).sum::<f32>())
            .collect::<Vec<_>>();
        assert_close(&observed[0], &expected, 1e-3);
        assert_close(&observed[1], &expected, 1e-3);
    }
}

#[test]
fn a_reduction_carries_the_chain_it_opens_and_the_chain_it_closes() {
    let runtime = open();
    let values = samples(16 * 64, 5);
    let mut tasks = Vec::new();
    let mut observed = Vec::new();
    for pinned in [true, false] {
        let graph = Graph::new();
        let data = graph.input(Shape::matrix(16, 64), Element::Single);
        let squared = graph.mul(data, data);
        if pinned {
            graph.retain(squared);
        }
        let activated = graph.relu(graph.sum_rows(squared));
        graph.retain(activated);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        tasks.push(program.task_count());
        runtime.write(&program, data, &values);
        runtime.run(&program);
        observed.push(runtime.read(&program, activated));
    }
    assert!(tasks[1] < tasks[0], "the opened fold holds {tasks:?} tasks");
    let expected = values
        .chunks(64)
        .flat_map(|row| {
            let total = row.iter().map(|value| value * value).sum::<f32>();
            [total.max(0.0)]
        })
        .collect::<Vec<_>>();
    assert_close(&observed[0], &expected, 1e-3);
    assert_close(&observed[1], &expected, 1e-3);
}

#[test]
fn an_opened_choice_picks_the_row_it_picked_before() {
    let runtime = open();
    let values = samples(9 * 400, 17);
    let mut observed = Vec::new();
    for pinned in [true, false] {
        let graph = Graph::new();
        let data = graph.input(Shape::matrix(9, 400), Element::Single);
        let logits = graph.exp(data);
        if pinned {
            graph.retain(logits);
        }
        let action = graph.argmax(logits);
        graph.retain(action);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, data, &values);
        runtime.run(&program);
        observed.push(runtime.read(&program, action));
    }
    assert_eq!(observed[0], observed[1]);
    let expected = values
        .chunks(400)
        .map(|row| {
            row.iter()
                .map(|value| value.exp())
                .enumerate()
                .fold((0usize, f32::MIN), |(best, top), (index, value)| {
                    if value > top {
                        (index, value)
                    } else {
                        (best, top)
                    }
                })
                .0 as f32
        })
        .collect::<Vec<_>>();
    assert_eq!(observed[0], expected);
}

#[test]
fn an_opened_draw_takes_the_action_it_took_before() {
    let runtime = open();
    let values = samples(8 * 300, 23);
    let mut observed = Vec::new();
    for pinned in [true, false] {
        let graph = Graph::new();
        let data = graph.input(Shape::matrix(8, 300), Element::Single);
        let seed = graph.input(Shape::scalar(), Element::Single);
        let logits = graph.mul(data, data);
        if pinned {
            graph.retain(logits);
        }
        let action = graph.categorical(logits, seed);
        graph.retain(action);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, data, &values);
        runtime.write(&program, seed, &[0.375]);
        runtime.run(&program);
        observed.push(runtime.read(&program, action));
    }
    assert_eq!(observed[0], observed[1]);
    assert_eq!(observed[0].len(), 8);
}

#[test]
fn an_opened_reduction_keeps_the_result_of_a_viewed_operand() {
    let runtime = open();
    let values = samples(4 * 8, 29);
    let mut observed = Vec::new();
    for pinned in [true, false] {
        let graph = Graph::new();
        let data = graph.input(Shape::matrix(4, 8), Element::Single);
        let squared = graph.mul(graph.transpose(graph.transpose(data)), data);
        if pinned {
            graph.retain(squared);
        }
        let rows = graph.sum_rows(squared);
        graph.retain(rows);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, data, &values);
        runtime.run(&program);
        observed.push(runtime.read(&program, rows));
    }
    let expected = values
        .chunks(8)
        .map(|row| row.iter().map(|value| value * value).sum::<f32>())
        .collect::<Vec<_>>();
    assert_close(&observed[0], &expected, 1e-4);
    assert_close(&observed[1], &expected, 1e-4);
}
