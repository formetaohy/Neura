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
fn a_delta_walk_of_rows_matches_a_step_a_row() {
    let (heads, batch, width, rows) = (2u32, 2u32, 3u32, 6u32);
    let queries = random(heads * batch * rows * width, 7);
    let keys = random(heads * batch * rows * width, 11);
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
    let rule = DeltaRule::new(&graph, "rule", heads, width, batch, rows);
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
    graph.retain(rule.state());
    let weights = runtime.weights(&graph);
    let program = runtime.compile_with(&graph, &weights, runtime.default_profile());
    runtime.write(&program, query, &queries);
    runtime.write(&program, key, &keys);
    runtime.write(&program, value, &values);
    runtime.write(&program, beta, &betas);
    runtime.write(&program, decay, &decays);
    runtime.write(&program, keep, &[1.0, 0.0]);
    runtime.run(&program);
    let walked = runtime.read(&program, out);
    let walked_state = runtime.read(&program, rule.state());

    let graph = Graph::new();
    let query = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let key = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let value = graph.input(Shape::of([heads, batch, rows, width]), Element::Single);
    let beta = graph.input(Shape::of([heads, batch, rows, 1]), Element::Single);
    let decay = graph.input(Shape::of([heads, batch, rows, 1]), Element::Single);
    let keep = graph.input(Shape::of([1, batch, 1, 1]), Element::Single);
    let rule = DeltaRule::new(&graph, "rule", heads, width, batch, 1);
    let stepped = (0..rows)
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
                Some(keep),
            )
        })
        .collect::<Vec<Value>>();
    let out = graph.concat(&stepped, 2);
    graph.retain(out);
    graph.retain(rule.state());
    let weights = runtime.weights(&graph);
    let program = runtime.compile_with(&graph, &weights, runtime.default_profile());
    runtime.write(&program, query, &queries);
    runtime.write(&program, key, &keys);
    runtime.write(&program, value, &values);
    runtime.write(&program, beta, &betas);
    runtime.write(&program, decay, &decays);
    runtime.write(&program, keep, &[1.0, 0.0]);
    runtime.run(&program);
    let step = runtime.read(&program, out);
    let step_state = runtime.read(&program, rule.state());
    assert_close(&walked, &step, 1e-4);
    assert_close(&walked_state, &step_state, 1e-4);
}
