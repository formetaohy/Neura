use neura_abi::Element;
use neura_graph::{Graph, Shape, Value};
use neura_nn::{DeltaInput, DeltaRule};
use neura_runtime::{Runtime, RuntimeRequest};

fn open() -> Runtime {
    Runtime::open(RuntimeRequest::default()).expect("a device runs the delta rule test")
}

fn random(count: u32, seed: u32) -> Vec<f32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
        })
        .collect()
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

#[test]
fn a_chunked_delta_rule_walks_the_rows_a_recurrent_step_walks() {
    let (heads, batch, width, rows, chunk) = (2u32, 2u32, 3u32, 6u32, 2u32);
    let mut queries = random(heads * batch * rows * width, 7);
    let mut keys = random(heads * batch * rows * width, 11);
    for tensor in [&mut queries, &mut keys] {
        for head in 0..heads {
            for plane in 0..batch {
                for row in 0..rows {
                    let at = (((head * batch + plane) * rows + row) * width) as usize;
                    let span = &mut tensor[at..at + width as usize];
                    let length = span.iter().map(|value| value * value).sum::<f32>().sqrt();
                    for value in span {
                        *value /= length;
                    }
                }
            }
        }
    }
    let values = random(heads * batch * rows * width, 13);
    let betas = random(heads * batch * rows, 17)
        .into_iter()
        .map(|value| 1.0 / (1.0 + (-value).exp()))
        .collect::<Vec<f32>>();
    let decays = random(heads * batch * rows, 19)
        .into_iter()
        .map(|value| -value.abs())
        .collect::<Vec<f32>>();
    let runtime = open();

    let graph = Graph::new();
    let query = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let key = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let value = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let beta = graph.input(Shape::of([heads, batch, rows, 1]), Element::Single);
    let decay = graph.input(Shape::of([heads, batch, rows, 1]), Element::Single);
    let keep = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
    let rule = DeltaRule::chunked(&graph, heads, width, batch, rows, chunk);
    let out = rule.forward(
        &graph,
        DeltaInput {
            query,
            key,
            value,
            beta,
            log_decay: decay,
        },
        Some(keep),
    );
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile_with(&graph, &weights, runtime.default_profile());
    runtime.write(&program, query, &queries);
    runtime.write(&program, key, &keys);
    runtime.write(&program, value, &values);
    runtime.write(&program, beta, &betas);
    runtime.write(&program, decay, &decays);
    runtime.write(&program, keep, &[1.0; 2]);
    runtime.run(&program);
    let chunked = runtime.read(&program, out);

    let graph = Graph::new();
    let query = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let key = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let value = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let beta = graph.input(Shape::of([heads, batch, rows, 1]), Element::Single);
    let decay = graph.input(Shape::of([heads, batch, rows, 1]), Element::Single);
    let rule = DeltaRule::new(&graph, heads, width, batch, 1);
    let walked = (0..rows)
        .map(|row| {
            rule.forward(
                &graph,
                DeltaInput {
                    query: graph.slice(query, 2, row, 1),
                    key: graph.slice(key, 2, row, 1),
                    value: graph.slice(value, 2, row, 1),
                    beta: graph.slice(beta, 2, row, 1),
                    log_decay: graph.slice(decay, 2, row, 1),
                },
                None,
            )
        })
        .collect::<Vec<Value>>();
    let out = graph.concat(&walked, 2);
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile_with(&graph, &weights, runtime.default_profile());
    runtime.write(&program, query, &queries);
    runtime.write(&program, key, &keys);
    runtime.write(&program, value, &values);
    runtime.write(&program, beta, &betas);
    runtime.write(&program, decay, &decays);
    runtime.run(&program);
    let recurrent = runtime.read(&program, out);
    assert_close(&chunked, &recurrent, 1e-4);
}

#[test]
fn a_fused_delta_step_advances_the_state_a_recurrent_step_advances() {
    let (heads, batch, width) = (2u32, 2u32, 128u32);
    let runtime = open();
    let initial = random(heads * batch * width * width, 23);
    let queries = random(heads * batch * width, 7);
    let keys = random(heads * batch * width, 11);
    let values = random(heads * batch * width, 13);
    let betas = random(heads * batch, 17);
    let decays = random(heads * batch, 19)
        .into_iter()
        .map(|value| -value.abs())
        .collect::<Vec<f32>>();
    let profile = runtime
        .profiles()
        .into_iter()
        .max_by_key(|profile| profile.shared_bytes())
        .expect("a device offers a profile");
    let run = |fused: bool| {
        let graph = Graph::new();
        let rule = DeltaRule::new(&graph, heads, width, batch, 1);
        let state = rule.state();
        let query = graph.input(Shape::of([heads, batch, 1, width]), Element::Single);
        let key = graph.input(Shape::of([heads, batch, 1, width]), Element::Single);
        let value = graph.input(Shape::of([heads, batch, 1, width]), Element::Single);
        let beta = graph.input(Shape::of([heads, batch, 1, 1]), Element::Single);
        let decay = graph.input(Shape::of([heads, batch, 1, 1]), Element::Single);
        let keep = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
        let input = DeltaInput {
            query,
            key,
            value,
            beta,
            log_decay: decay,
        };
        let out = match fused {
            true => rule.advance(&graph, input, Some(keep)),
            false => rule.forward(&graph, input, Some(keep)),
        };
        graph.retain(out);
        graph.retain(state);
        let weights = runtime.weights(&graph);
        let program = runtime.compile_with(&graph, &weights, profile);
        runtime.write(&program, state, &initial);
        runtime.write(&program, query, &queries);
        runtime.write(&program, key, &keys);
        runtime.write(&program, value, &values);
        runtime.write(&program, beta, &betas);
        runtime.write(&program, decay, &decays);
        runtime.write(&program, keep, &[1.0, 0.0]);
        runtime.run(&program);
        (runtime.read(&program, out), runtime.read(&program, state))
    };
    let (fused_out, fused_state) = run(true);
    let (plain_out, plain_state) = run(false);
    assert_close(&fused_out, &plain_out, 1e-3);
    assert_close(&fused_state, &plain_state, 1e-3);
}
