use neura_abi::Element;
use neura_gpu::PREFERENCE;
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};

#[path = "support/backend.rs"]
mod backend;
#[path = "support/attention.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use backend::open_with;
use reference::{Shapes, attention_backward, attention_forward};
use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

const BESIDE_ROWS: u32 = 256;

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
    graph_of_beside(shapes, None)
}

fn graph_of_beside(
    shapes: Shapes,
    beside_the_output: Option<f32>,
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
            scale: Some(graph.knob(shapes.scale)),
            causal: shapes.causal,
            origin: None,
            segments: None,
            reach: (shapes.reach > 0).then_some(shapes.reach),
            query_segments: None,
        },
    );
    graph.retain(out);
    if let Some(value) = beside_the_output {
        graph.retain(graph.fill(Shape::of([1, 1, BESIDE_ROWS, shapes.width]), value));
    }
    (graph, queries, keys, values, out)
}

fn run_forward(shapes: Shapes, tolerance: f32) -> (Vec<f32>, Vec<f32>) {
    let runtime = open();
    run_forward_on(&runtime, shapes, tolerance)
}

fn run_forward_on(
    runtime: &neura_runtime::Runtime,
    shapes: Shapes,
    tolerance: f32,
) -> (Vec<f32>, Vec<f32>) {
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
    run_backward_beside(shapes, tolerance, None);
}

fn run_backward_beside(shapes: Shapes, tolerance: f32, beside_the_output: Option<f32>) {
    let runtime = open();
    run_backward_on(&runtime, shapes, tolerance, beside_the_output)
}

fn run_backward_on(
    runtime: &neura_runtime::Runtime,
    shapes: Shapes,
    tolerance: f32,
    beside_the_output: Option<f32>,
) {
    let (graph, queries, keys, values, out) = graph_of_beside(shapes, beside_the_output);
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
fn a_masked_key_weighs_nothing_where_a_row_scores_the_identity_of_the_maximum() {
    let runtime = open();
    let shapes = Shapes {
        heads: 1,
        key_heads: 1,
        batch: 1,
        queries: 4,
        keys: 4,
        width: 4,
        causal: true,
        origin: 0,
        reach: 0,
        scale: 1.0,
    };
    let (graph, queries, keys, values, out) = graph_of(shapes);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let queries_data = [f32::MIN, 0.0, 0.0, 0.0].repeat(4);
    let keys_data = [1.0, 0.0, 0.0, 0.0].repeat(4);
    let values_data = (1..=16).map(|index| index as f32).collect::<Vec<_>>();
    runtime.write(&program, queries, &queries_data);
    runtime.write(&program, keys, &keys_data);
    runtime.write(&program, values, &values_data);
    runtime.run(&program);
    let produced = runtime.read(&program, out);
    let (expected, _) = attention_forward(shapes, &queries_data, &keys_data, &values_data);
    assert_close(&produced, &expected, 1e-5);
    assert_close(
        &produced,
        &[
            1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 6.0, 5.0, 6.0, 7.0, 8.0, 7.0, 8.0, 9.0, 10.0,
        ],
        1e-5,
    );
}

#[test]
fn a_masked_key_weighs_nothing_on_every_platform_backend() {
    let shapes = Shapes {
        heads: 1,
        key_heads: 1,
        batch: 1,
        queries: 5,
        keys: 5,
        width: 4,
        causal: true,
        origin: 0,
        reach: 0,
        scale: 0.5,
    };
    for &backend in PREFERENCE {
        let runtime = open_with(backend);
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
        let (expected, _) = attention_forward(shapes, &queries_data, &keys_data, &values_data);
        assert_close(&produced, &expected, 1e-5);
    }
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
fn an_attention_gradient_reads_no_row_of_the_storage_beside_its_output() {
    for causal in [true, false] {
        let shapes = shapes(causal);
        run_backward_beside(shapes, 1e-5, Some(f32::NAN));
        run_backward_beside(
            Shapes {
                heads: 1,
                key_heads: 1,
                batch: 1,
                queries: 260,
                keys: 260,
                width: 4,
                causal,
                origin: 0,
                reach: 0,
                scale: 0.5,
            },
            1e-4,
            Some(f32::NAN),
        );
    }
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
                    scale: Some(graph.knob(0.5)),
                    causal: false,
                    origin: None,
                    segments: None,
                    reach: None,
                    query_segments: None,
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
                scale: Some(graph.knob(0.5)),
                causal: true,
                origin: None,
                segments: None,
                reach: None,
                query_segments: None,
            },
        );
    }));
    let wide = graph.parameter(Shape::of([1, 1, 3, 8]), Init::Zero, Element::Single);
    let vector = graph.parameter(Shape::vector(2), Init::Zero, Element::Single);
    assert!(refuses(|| {
        let _ = graph.attention(
            query,
            key,
            value,
            AttentionOptions {
                scale: Some(vector),
                causal: false,
                origin: None,
                segments: None,
                reach: None,
                query_segments: None,
            },
        );
    }));
    assert!(refuses(|| {
        let _ = graph.attention(
            query,
            wide,
            value,
            AttentionOptions {
                scale: Some(graph.knob(0.5)),
                causal: false,
                origin: None,
                segments: None,
                reach: None,
                query_segments: None,
            },
        );
    }));
}

#[test]
fn a_head_wider_than_a_thread_carries_walks_every_number_of_its_row() {
    let reference_scale = |width: u32| 1.0 / (width as f32).sqrt();
    for width in [67u32, 80, 96, 128, 256] {
        for causal in [false, true] {
            let shapes = Shapes {
                heads: 1,
                key_heads: 1,
                batch: 1,
                queries: 5,
                keys: 5,
                width,
                causal,
                origin: 0,
                reach: 0,
                scale: reference_scale(width),
            };
            run_forward(shapes, 1e-4);
            run_backward(shapes, 1e-4);
        }
    }
    let grouped = Shapes {
        heads: 4,
        key_heads: 2,
        batch: 2,
        queries: 6,
        keys: 6,
        width: 128,
        causal: true,
        origin: 0,
        reach: 0,
        scale: reference_scale(128),
    };
    run_forward(grouped, 1e-4);
    run_backward(grouped, 1e-4);
    for &backend in PREFERENCE {
        let runtime = open_with(backend);
        for (width, causal) in [(67u32, true), (128, false)] {
            let shapes = Shapes {
                heads: 2,
                key_heads: 1,
                batch: 1,
                queries: 5,
                keys: 5,
                width,
                causal,
                origin: 0,
                reach: 0,
                scale: reference_scale(width),
            };
            run_forward_on(&runtime, shapes, 1e-4);
            run_backward_on(&runtime, shapes, 1e-4, None);
        }
    }
    let windowed = Shapes {
        heads: 1,
        key_heads: 1,
        batch: 1,
        queries: 7,
        keys: 7,
        width: 128,
        causal: true,
        origin: 0,
        reach: 3,
        scale: reference_scale(128),
    };
    run_forward(windowed, 1e-4);
    run_backward(windowed, 1e-4);
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
            scale: Some(graph.knob(1.0 / (width as f32).sqrt())),
            causal: true,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
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
                scale: Some(graph.knob(shapes.scale)),
                causal: shapes.causal,
                origin: None,
                segments: None,
                reach: None,
                query_segments: None,
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

const KNOB_WIDTH: u32 = 4;
const KNOB_ROWS: u32 = 3;

fn knob_graph<'g>(
    graph: &Graph<'g>,
    scale: Option<Value<'g>>,
) -> (Value<'g>, Value<'g>, Value<'g>, Value<'g>) {
    let tensor = || {
        graph.parameter(
            Shape::of([1, 1, KNOB_ROWS, KNOB_WIDTH]),
            Init::Zero,
            Element::Single,
        )
    };
    let (queries, keys, values) = (tensor(), tensor(), tensor());
    let out = graph.attention(
        queries,
        keys,
        values,
        AttentionOptions {
            scale,
            causal: true,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    (queries, keys, values, out)
}

fn knob_shapes(scale: f32) -> Shapes {
    Shapes {
        heads: 1,
        key_heads: 1,
        batch: 1,
        queries: KNOB_ROWS,
        keys: KNOB_ROWS,
        width: KNOB_WIDTH,
        causal: true,
        origin: 0,
        reach: 0,
        scale,
    }
}

fn knob_inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    (
        data(KNOB_ROWS * KNOB_WIDTH, 17),
        data(KNOB_ROWS * KNOB_WIDTH, 29),
        data(KNOB_ROWS * KNOB_WIDTH, 43),
    )
}

#[test]
fn an_attention_scale_the_host_writes_steers_every_score_it_weighs() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let knob = graph.named_knob("attention.scale", 1.0);
    let (queries, keys, values, out) = knob_graph(&graph, Some(knob));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (queries_data, keys_data, values_data) = knob_inputs();
    runtime.write(&program, queries, &queries_data);
    runtime.write(&program, keys, &keys_data);
    runtime.write(&program, values, &values_data);
    for written in [1.0f32, 0.5, 2.0, 1.0 / (KNOB_WIDTH as f32).sqrt()] {
        runtime.write(&program, knob, &[written]);
        runtime.run(&program);
        let produced = runtime.read(&program, out);
        let (expected, _) = attention_forward(
            knob_shapes(written),
            &queries_data,
            &keys_data,
            &values_data,
        );
        assert_close(&produced, &expected, 1e-5);
    }
    assert_eq!(
        runtime.built_plans(),
        1,
        "a program weighs every scale the host writes",
    );
}

#[test]
fn an_attention_without_a_scale_weighs_its_scores_by_the_width_it_spans() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let (queries, keys, values, out) = knob_graph(&graph, None);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (queries_data, keys_data, values_data) = knob_inputs();
    runtime.write(&program, queries, &queries_data);
    runtime.write(&program, keys, &keys_data);
    runtime.write(&program, values, &values_data);
    runtime.run(&program);
    let produced = runtime.read(&program, out);
    let (expected, _) = attention_forward(
        knob_shapes(1.0 / (KNOB_WIDTH as f32).sqrt()),
        &queries_data,
        &keys_data,
        &values_data,
    );
    assert_close(&produced, &expected, 1e-5);
}

#[test]
fn the_device_refuses_an_attention_scale_it_cannot_weigh_with() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let knob = graph.named_knob("attention.scale", 1.0);
    let (_, _, _, out) = knob_graph(&graph, Some(knob));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    for written in [0.0f32, f32::NAN, f32::INFINITY] {
        runtime.write(&program, knob, &[written]);
        runtime.run(&program);
        let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = runtime.read(&program, out);
        }))
        .expect_err("a scale no score survives is refused");
        let message = refused
            .downcast_ref::<String>()
            .cloned()
            .unwrap_or_else(|| "a refusal without a message".to_owned());
        assert!(
            message.contains("the device refused the attention scale of the attention task"),
            "{message}",
        );
    }
}

#[test]
fn a_device_task_steers_the_scale_of_an_attention_it_feeds() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let knob = graph.named_knob("attention.scale", 1.0);
    graph.mul_into(knob, graph.fill(Shape::scalar(), 2.0));
    let (queries, keys, values, out) = knob_graph(&graph, Some(knob));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (queries_data, keys_data, values_data) = knob_inputs();
    runtime.write(&program, queries, &queries_data);
    runtime.write(&program, keys, &keys_data);
    runtime.write(&program, values, &values_data);
    for round in 1..=2u32 {
        runtime.run(&program);
        let written = 2.0f32.powi(round as i32);
        assert_eq!(
            runtime.read(&program, knob),
            [written],
            "the device doubled the scale it weighs with",
        );
        let produced = runtime.read(&program, out);
        let (expected, _) = attention_forward(
            knob_shapes(written),
            &queries_data,
            &keys_data,
            &values_data,
        );
        assert_close(&produced, &expected, 1e-5);
    }
}
