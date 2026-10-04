use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};
use neura_runtime::Runtime;

#[path = "support/attention.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use reference::{Shapes, attention_backward, attention_forward};
use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn data(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..count)
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn graph_of(
    shapes: Shapes,
) -> (
    Graph<'static>,
    Value<'static>,
    Value<'static>,
    Value<'static>,
    Value<'static>,
) {
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
            scale: shapes.scale,
            causal: shapes.causal,
            origin: None,
            segments: None,
            reach: (shapes.reach > 0).then_some(shapes.reach),
        },
    );
    graph.retain(out);
    (graph, queries, keys, values, out)
}

fn run_forward(shapes: Shapes, tolerance: f32) -> (Vec<f32>, Vec<f32>) {
    let runtime = open();
    let (graph, queries, keys, values, out) = graph_of(shapes);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let queries_data = data(
        shapes.heads * shapes.batch * shapes.queries * shapes.width,
        17,
    );
    let keys_data = data(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        29,
    );
    let values_data = data(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        43,
    );
    runtime.write(&program, queries, &queries_data);
    runtime.write(&program, keys, &keys_data);
    runtime.write(&program, values, &values_data);
    runtime.run(&program);
    let produced = runtime.read(&program, out);
    let (expected, statistics) = attention_forward(shapes, &queries_data, &keys_data, &values_data);
    assert_close(&produced, &expected, tolerance);
    (produced, statistics)
}

fn run_backward(shapes: Shapes, tolerance: f32) {
    let runtime = open();
    let (graph, queries, keys, values, out) = graph_of(shapes);
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let query_grad = gradients.of(queries);
    let key_grad = gradients.of(keys);
    let value_grad = gradients.of(values);
    graph.retain(query_grad);
    graph.retain(key_grad);
    graph.retain(value_grad);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let queries_data = data(
        shapes.heads * shapes.batch * shapes.queries * shapes.width,
        17,
    );
    let keys_data = data(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        29,
    );
    let values_data = data(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        43,
    );
    runtime.write(&program, queries, &queries_data);
    runtime.write(&program, keys, &keys_data);
    runtime.write(&program, values, &values_data);
    runtime.run(&program);
    let (out_data, statistics) = attention_forward(shapes, &queries_data, &keys_data, &values_data);
    let gradient = vec![1.0f32; out_data.len()];
    let (expected_query, expected_key, expected_value) = attention_backward(
        shapes,
        &queries_data,
        &keys_data,
        &values_data,
        &out_data,
        &statistics,
        &gradient,
    );
    let produced = runtime.read_many(&program, &[query_grad, key_grad, value_grad]);
    assert_close(&produced[0], &expected_query, tolerance);
    assert_close(&produced[1], &expected_key, tolerance);
    assert_close(&produced[2], &expected_value, tolerance);
}

fn shapes(causal: bool) -> Shapes {
    Shapes {
        heads: 2,
        key_heads: 2,
        batch: 2,
        queries: 5,
        keys: 5,
        width: 4,
        causal,
        origin: 0,
        reach: 0,
        scale: 0.5,
    }
}

#[test]
fn an_attention_block_scores_every_row_it_weights() {
    run_forward(shapes(false), 1e-5);
    run_forward(shapes(true), 1e-5);
}

#[test]
fn an_attention_reads_queries_and_keys_of_different_lengths() {
    run_forward(
        Shapes {
            heads: 1,
            key_heads: 1,
            batch: 1,
            queries: 3,
            keys: 7,
            width: 3,
            causal: false,
            origin: 0,
            reach: 0,
            scale: 0.25,
        },
        1e-5,
    );
}

#[test]
fn an_attention_wider_than_one_task_walks_every_row_of_its_block() {
    let shapes = Shapes {
        heads: 1,
        key_heads: 1,
        batch: 1,
        queries: 300,
        keys: 300,
        width: 2,
        causal: true,
        origin: 0,
        reach: 0,
        scale: 0.5,
    };
    let runtime = open();
    let (graph, _, _, _, _) = graph_of(shapes);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert!(
        program.task_count() > 1,
        "a block of 300 rows rides more than one task",
    );
    run_forward(shapes, 1e-4);
}

#[test]
fn an_attention_block_walks_its_gradient_back_into_queries_keys_and_values() {
    run_backward(shapes(false), 1e-5);
    run_backward(shapes(true), 1e-5);
}

#[test]
fn an_attention_shares_one_key_head_among_a_group_of_queries() {
    run_forward(
        Shapes {
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
        },
        1e-5,
    );
    run_backward(
        Shapes {
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
        },
        1e-5,
    );
    run_forward(
        Shapes {
            heads: 8,
            key_heads: 1,
            batch: 1,
            queries: 6,
            keys: 4,
            width: 2,
            causal: false,
            origin: 0,
            reach: 0,
            scale: 0.25,
        },
        1e-5,
    );
}

#[test]
fn an_attention_refuses_values_of_another_width() {
    let runtime = open();
    assert!(
        refuses(|| {
            let graph = Graph::new();
            let queries = graph.parameter(Shape::of([1, 1, 2, 4]), Init::Zero, Element::Single);
            let keys = graph.parameter(Shape::of([1, 1, 2, 4]), Init::Zero, Element::Single);
            let values = graph.parameter(Shape::of([1, 1, 2, 6]), Init::Zero, Element::Single);
            let out = graph.attention(
                queries,
                keys,
                values,
                AttentionOptions {
                    scale: 0.5,
                    causal: false,
                    origin: None,
                    segments: None,
                    reach: None,
                },
            );
            graph.retain(out);
            let weights = runtime.weights(&graph);
            let _ = runtime.compile(&graph, &weights);
        }),
        "an attention of width four reads keys of width four and values of width six",
    );
}

#[test]
fn an_attention_refuses_a_group_of_queries_that_does_not_divide() {
    let runtime = open();
    assert!(
        refuses(|| {
            let shapes = Shapes {
                heads: 3,
                key_heads: 2,
                batch: 1,
                queries: 2,
                keys: 2,
                width: 2,
                causal: false,
                origin: 0,
                reach: 0,
                scale: 0.5,
            };
            let (graph, _, _, _, _) = graph_of(shapes);
            let weights = runtime.weights(&graph);
            let _ = runtime.compile(&graph, &weights);
        }),
        "three query heads share two key heads only by halves",
    );
}

#[test]
fn a_fused_attention_holds_no_score_matrix_of_its_own() {
    let runtime = open();
    let tokens = [64u32, 128, 256];
    let mut held = Vec::new();
    for tokens in tokens {
        let shapes = Shapes {
            heads: 4,
            key_heads: 4,
            batch: 1,
            queries: tokens,
            keys: tokens,
            width: 32,
            causal: true,
            origin: 0,
            reach: 0,
            scale: 0.176_776_69,
        };
        let (graph, queries, keys, values, out) = graph_of(shapes);
        let loss = graph.sum(out);
        let gradients = graph.backward(loss);
        graph.retain(gradients.of(queries));
        graph.retain(gradients.of(keys));
        graph.retain(gradients.of(values));
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        assert!(
            program.task_count() < 64,
            "an attention of {tokens} tokens rides {} tasks",
            program.task_count(),
        );
        held.push(program.tensor_bytes());
    }
    assert!(
        held[1] < 4 * held[0] && held[2] < 4 * held[1],
        "an attention of twice the tokens quadruples its arena: {held:?}",
    );
    assert!(
        held[2] < 1 << 20,
        "an attention of 256 tokens holds {} bytes",
        held[2],
    );
}

#[test]
fn an_attention_carries_the_log_sum_of_every_row_it_weights() {
    let runtime = open();
    let shapes = Shapes {
        heads: 1,
        key_heads: 1,
        batch: 1,
        queries: 4,
        keys: 4,
        width: 2,
        causal: true,
        origin: 0,
        reach: 0,
        scale: 0.5,
    };
    let (graph, queries, keys, values, _) = graph_of(shapes);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let queries_data = data(8, 17);
    let keys_data = data(8, 29);
    let values_data = data(8, 43);
    runtime.write(&program, queries, &queries_data);
    runtime.write(&program, keys, &keys_data);
    runtime.write(&program, values, &values_data);
    runtime.run(&program);
    let (_, statistics) = attention_forward(shapes, &queries_data, &keys_data, &values_data);
    let snapshot = graph.snapshot();
    let statistic = snapshot
        .values()
        .iter()
        .find(|value| value.shape.dims() == [1, 1, 4, 1])
        .expect("an attention declares the log sum of its rows");
    assert_eq!(statistic.shape.elements(), 4);
    assert_eq!(statistics.len(), 4);
}

#[test]
fn a_causal_attention_stops_the_graph_it_cannot_align() {
    let graph = Graph::new();
    let query = graph.parameter(Shape::of([1, 1, 4, 2]), Init::Zero, Element::Single);
    let key = graph.parameter(Shape::of([1, 1, 3, 2]), Init::Zero, Element::Single);
    let value = graph.parameter(Shape::of([1, 1, 3, 2]), Init::Zero, Element::Single);
    assert!(refuses(|| {
        let _ = graph.attention(
            query,
            key,
            value,
            AttentionOptions {
                scale: 0.5,
                causal: true,
                origin: None,
                segments: None,
                reach: None,
            },
        );
    }));
    assert!(refuses(|| {
        let _ = graph.attention(
            query,
            key,
            value,
            AttentionOptions {
                scale: 0.0,
                causal: false,
                origin: None,
                segments: None,
                reach: None,
            },
        );
    }));
    let wide = graph.parameter(Shape::of([1, 1, 3, 8]), Init::Zero, Element::Single);
    assert!(refuses(|| {
        let _ = graph.attention(
            query,
            wide,
            value,
            AttentionOptions {
                scale: 0.5,
                causal: false,
                origin: None,
                segments: None,
                reach: None,
            },
        );
    }));
}

#[test]
fn a_fused_attention_leaves_no_room_for_a_score_it_cannot_carry() {
    let runtime: Runtime = open();
    let graph = Graph::new();
    let width = 96;
    let query = graph.parameter(Shape::of([1, 1, 4, width]), Init::Zero, Element::Single);
    let key = graph.parameter(Shape::of([1, 1, 4, width]), Init::Zero, Element::Single);
    let value = graph.parameter(Shape::of([1, 1, 4, width]), Init::Zero, Element::Single);
    let out = graph.attention(
        query,
        key,
        value,
        AttentionOptions {
            scale: 0.5,
            causal: false,
            origin: None,
            segments: None,
            reach: None,
        },
    );
    graph.retain(out);
    let weights = runtime.weights(&graph);
    assert!(
        refuses(|| {
            let _ = runtime.compile(&graph, &weights);
        }),
        "a fused attention holds a query row of a width no thread carries",
    );
}

#[test]
fn a_fused_attention_holds_a_sequence_no_score_matrix_holds() {
    let runtime = open();
    let heads = 4u32;
    let tokens = 512u32;
    let width = 32u32;
    let composed = || {
        let graph = Graph::new();
        let tensor = graph.parameter(
            Shape::of([heads, 1, tokens, width]),
            Init::Zero,
            Element::Single,
        );
        let scores = graph.mul(
            graph.matmul(tensor, graph.permute(tensor, [0, 1, 3, 2])),
            graph.fill(Shape::scalar(), 1.0 / (width as f32).sqrt()),
        );
        let out = graph.matmul(graph.softmax(scores), tensor);
        let loss = graph.sum(out);
        let gradients = graph.backward(loss);
        graph.retain(gradients.of(tensor));
        let weights = runtime.weights(&graph);
        let _ = runtime.compile(&graph, &weights);
    };
    assert!(
        refuses(composed),
        "a score matrix of {tokens} tokens by {tokens} fits the device heap",
    );
    let graph = Graph::new();
    let tensor = graph.parameter(
        Shape::of([heads, 1, tokens, width]),
        Init::Zero,
        Element::Single,
    );
    let out = graph.attention(
        tensor,
        tensor,
        tensor,
        AttentionOptions {
            scale: 1.0 / (width as f32).sqrt(),
            causal: true,
            origin: None,
            segments: None,
            reach: None,
        },
    );
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let gradient = gradients.of(tensor);
    graph.retain(gradient);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert!(
        program.tensor_bytes() < 1 << 21,
        "a fused attention of {tokens} tokens holds {} bytes",
        program.tensor_bytes(),
    );
    runtime.run(&program);
    assert!(
        runtime
            .read(&program, gradient)
            .iter()
            .all(|value| value.is_finite()),
        "a fused attention of {tokens} tokens walks a gradient back into its input",
    );
}

#[test]
fn every_profile_the_device_offers_runs_the_same_attention() {
    let shapes = Shapes {
        heads: 2,
        key_heads: 2,
        batch: 1,
        queries: 7,
        keys: 7,
        width: 3,
        causal: true,
        origin: 0,
        reach: 0,
        scale: 0.5,
    };
    let runtime = open();
    let (graph, queries, keys, values, out) = graph_of(shapes);
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let query_grad = gradients.of(queries);
    let key_grad = gradients.of(keys);
    let value_grad = gradients.of(values);
    graph.retain(query_grad);
    graph.retain(key_grad);
    graph.retain(value_grad);
    let weights = runtime.weights(&graph);
    let queries_data = data(
        shapes.heads * shapes.batch * shapes.queries * shapes.width,
        17,
    );
    let keys_data = data(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        29,
    );
    let values_data = data(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        43,
    );
    let (expected_out, statistics) =
        attention_forward(shapes, &queries_data, &keys_data, &values_data);
    let gradient = vec![1.0f32; expected_out.len()];
    let (expected_query, expected_key, expected_value) = attention_backward(
        shapes,
        &queries_data,
        &keys_data,
        &values_data,
        &expected_out,
        &statistics,
        &gradient,
    );
    for profile in runtime.profiles() {
        let program = runtime.compile_with(&graph, &weights, profile);
        runtime.write(&program, queries, &queries_data);
        runtime.write(&program, keys, &keys_data);
        runtime.write(&program, values, &values_data);
        runtime.run(&program);
        let produced = runtime.read_many(&program, &[out, query_grad, key_grad, value_grad]);
        assert_close(&produced[0], &expected_out, 1e-5);
        assert_close(&produced[1], &expected_query, 1e-5);
        assert_close(&produced[2], &expected_key, 1e-5);
        assert_close(&produced[3], &expected_value, 1e-5);
    }
}

fn block<'g>(
    graph: &Graph<'g>,
    shapes: Shapes,
    recomputed: bool,
) -> (Value<'g>, [Value<'g>; 3], [Value<'g>; 3]) {
    let tensor = |heads: u32, tokens: u32| {
        graph.parameter(
            Shape::of([heads, shapes.batch, tokens, shapes.width]),
            Init::Uniform {
                low: -0.5,
                high: 0.5,
            },
            Element::Single,
        )
    };
    let queries = tensor(shapes.heads, shapes.queries);
    let keys = tensor(shapes.key_heads, shapes.keys);
    let values = tensor(shapes.key_heads, shapes.keys);
    let attend = |graph: &Graph<'g>| {
        graph.attention(
            queries,
            keys,
            values,
            AttentionOptions {
                scale: shapes.scale,
                causal: shapes.causal,
                origin: None,
                segments: None,
                reach: None,
            },
        )
    };
    let out = if recomputed {
        graph.recompute(|graph| attend(graph))
    } else {
        attend(graph)
    };
    graph.retain(out);
    let gradients = graph.backward(graph.sum(out));
    let weight_gradients = [
        gradients.of(queries),
        gradients.of(keys),
        gradients.of(values),
    ];
    for gradient in &weight_gradients {
        graph.retain(*gradient);
    }
    (out, [queries, keys, values], weight_gradients)
}

#[test]
fn a_recomputed_attention_block_matches_the_gradients_it_replaced() {
    let runtime = open();
    let shapes = shapes(true);
    let plain = Graph::new();
    let (plain_out, plain_inputs, plain_gradients) = block(&plain, shapes, false);
    let plain_weights = runtime.weights(&plain);
    let plain_program = runtime.compile(&plain, &plain_weights);

    let recomputed = Graph::new();
    let (out, inputs, gradients) = block(&recomputed, shapes, true);
    let weights = runtime.weights(&recomputed);
    let program = runtime.compile(&recomputed, &weights);

    let queries_data = data(
        shapes.heads * shapes.batch * shapes.queries * shapes.width,
        17,
    );
    let keys_data = data(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        29,
    );
    let values_data = data(
        shapes.key_heads * shapes.batch * shapes.keys * shapes.width,
        43,
    );
    for (runtime, program, inputs) in [
        (&runtime, &plain_program, plain_inputs),
        (&runtime, &program, inputs),
    ] {
        runtime.write(program, inputs[0], &queries_data);
        runtime.write(program, inputs[1], &keys_data);
        runtime.write(program, inputs[2], &values_data);
        runtime.run(program);
    }
    assert_close(
        &runtime.read(&program, out),
        &runtime.read(&plain_program, plain_out),
        1e-5,
    );
    let observed = runtime.read_many(&program, &gradients);
    let expected = runtime.read_many(&plain_program, &plain_gradients);
    for (observed, expected) in observed.iter().zip(&expected) {
        assert_close(observed, expected, 1e-5);
    }
    assert!(
        program.task_count() > plain_program.task_count(),
        "a recomputed block re-runs the plan prefix it was authored into",
    );
}

#[test]
fn a_window_weighs_only_the_keys_a_query_reaches() {
    let mut banded = shapes(true);
    banded.reach = 2;
    run_forward(banded, 1e-5);
    run_backward(banded, 1e-5);
}

#[test]
fn a_window_of_one_weighs_every_row_its_own_key() {
    let mut banded = shapes(true);
    banded.reach = 1;
    run_forward(banded, 1e-5);
    run_backward(banded, 1e-5);
}

#[test]
fn a_window_wider_than_the_keys_weighs_them_all() {
    let mut banded = shapes(true);
    banded.reach = banded.keys + 1;
    run_forward(banded, 1e-5);
    run_backward(banded, 1e-5);
}
