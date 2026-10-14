use neura_abi::Element;
use neura_decoder::Cache;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::Runtime;

use crate::support;
use crate::support::attention::{Shapes, attention_forward};

use support::{assert_close, open};

const HEADS: u32 = 2;
const WIDTH: u32 = 4;
const CAPACITY: u32 = 16;
const VOCABULARY: u32 = 6;
const SCALE: f32 = 0.5;
const CHUNK: u32 = 4;

struct Model {
    token: Value,
    cursor: Value,
    cache: Cache,
    table: Value,
    output: Value,
    logits: Value,
}

fn model(graph: &Graph, rows: u32) -> Model {
    let token = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
    let cursor = graph.input(Shape::scalar(), Element::Single);
    let cache = Cache::new(graph, HEADS, WIDTH, CAPACITY, rows);
    let spread =
        |graph: &Graph, held: Value| graph.broadcast_to(held, Shape::of([HEADS, 1, rows, WIDTH]));
    let table = graph.parameter(
        Shape::of([1, 1, VOCABULARY, WIDTH]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let output = graph.parameter(
        Shape::of([1, 1, VOCABULARY, WIDTH]),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let embedded = graph.matmul(graph.one_hot(token, VOCABULARY), table);
    let written = spread(graph, embedded);
    cache.write(graph, cursor, written, written);
    let attended = cache.attend(graph, written, cursor, graph.knob(SCALE));
    let logits = graph.matmul(
        graph.slice(attended, 0, 0, 1),
        graph.permute(output, [0, 1, 3, 2]),
    );
    Model {
        token,
        cursor,
        cache,
        table,
        output,
        logits,
    }
}

fn pair(runtime: &Runtime) -> (neura_runtime::Program, Model, neura_runtime::Program, Model) {
    let wide = Graph::new();
    let wide_model = model(&wide, CHUNK);
    let weights = runtime.weights(&wide);
    let prefill = runtime.compile(&wide, &weights);
    let narrow = Graph::new();
    let narrow_model = model(&narrow, 1);
    runtime.rebind(&weights, &narrow);
    let decode = runtime.compile(&narrow, &weights);
    (prefill, wide_model, decode, narrow_model)
}

fn embedding(
    runtime: &Runtime,
    model: &Model,
    program: &neura_runtime::Program,
    tokens: &[f32],
) -> Vec<f32> {
    let table = runtime.read(program, model.table);
    let mut held = Vec::new();
    for id in tokens {
        let at = *id as usize * WIDTH as usize;
        held.extend_from_slice(&table[at..at + WIDTH as usize]);
    }
    held
}

#[test]
fn a_chunked_prefill_fills_the_cache_a_rebound_decode_reads() {
    let runtime = open();
    let (prefill, wide, decode, narrow) = pair(&runtime);
    let tokens = [1.0f32, 3.0, 0.0, 5.0];
    runtime.write(&prefill, wide.token, &tokens);
    runtime.write(&prefill, wide.cursor, &[0.0]);
    runtime.run(&prefill);

    let rows = embedding(&runtime, &wide, &prefill, &tokens);
    let mut expected = vec![0.0f32; (HEADS * CAPACITY * WIDTH) as usize];
    for head in 0..HEADS {
        for row in 0..CHUNK {
            let source = row as usize * WIDTH as usize;
            let at = (head * CAPACITY + row) as usize * WIDTH as usize;
            expected[at..at + WIDTH as usize]
                .copy_from_slice(&rows[source..source + WIDTH as usize]);
        }
    }
    let [keys, values] = narrow.cache.states();
    assert_close(&runtime.read(&decode, keys), &expected, 0.0);
    assert_close(&runtime.read(&decode, values), &expected, 0.0);

    let asked = [2.0f32];
    runtime.write(&decode, narrow.token, &asked);
    runtime.write(&decode, narrow.cursor, &[CHUNK as f32]);
    runtime.run(&decode);
    let produced = runtime.read(&decode, narrow.logits);
    let written = embedding(&runtime, &narrow, &decode, &asked);
    let mut cached = vec![0.0f32; (CAPACITY * WIDTH) as usize];
    cached[..(CHUNK * WIDTH) as usize].copy_from_slice(&rows);
    cached[(CHUNK * WIDTH) as usize..((CHUNK + 1) * WIDTH) as usize].copy_from_slice(&written);
    let (attended, _) = attention_forward(
        Shapes {
            heads: 1,
            key_heads: 1,
            batch: 1,
            queries: 1,
            keys: CAPACITY,
            width: WIDTH,
            causal: true,
            origin: CHUNK,
            reach: 0,
            scale: SCALE,
        },
        &written,
        &cached,
        &cached,
    );
    let output = runtime.read(&decode, narrow.output);
    let mut reference = vec![0.0f32; VOCABULARY as usize];
    for class in 0..VOCABULARY as usize {
        reference[class] = (0..WIDTH as usize)
            .map(|depth| attended[depth] * output[class * WIDTH as usize + depth])
            .sum();
    }
    assert_close(&produced, &reference, 1e-4);
}

#[test]
fn a_chunk_walks_a_prompt_into_the_logits_a_row_at_a_time_walks() {
    let runtime = open();
    let (prefill, wide, decode, narrow) = pair(&runtime);
    let prompt = [1.0f32, 4.0, 2.0, 0.0, 3.0, 5.0];
    runtime.write(&prefill, wide.token, &prompt[..CHUNK as usize]);
    runtime.write(&prefill, wide.cursor, &[0.0]);
    runtime.run(&prefill);
    for (at, id) in prompt.iter().enumerate().skip(CHUNK as usize) {
        runtime.write(&decode, narrow.token, &[*id]);
        runtime.write(&decode, narrow.cursor, &[at as f32]);
        runtime.run(&decode);
    }
    let chunked = runtime.read(&decode, narrow.logits);

    let cleared = vec![0.0f32; (HEADS * CAPACITY * WIDTH) as usize];
    for held in narrow.cache.states() {
        runtime.write(&decode, held, &cleared);
    }
    for (at, id) in prompt.iter().enumerate() {
        runtime.write(&decode, narrow.token, &[*id]);
        runtime.write(&decode, narrow.cursor, &[at as f32]);
        runtime.run(&decode);
    }
    assert_close(&chunked, &runtime.read(&decode, narrow.logits), 1e-4);
}
