use crate::measure;
use neura::decoder::Cache;
use neura::{AttentionOptions, Element, Graph, Init, Runtime, Shape, Value};

const HEADS: u32 = 8;
const KEY_HEADS: u32 = 2;
const WIDTH: u32 = 128;
const CACHE: u32 = 1024;
const ROWS: u32 = 64;
const STARVED: u32 = 2048;
const SCALE: f32 = 0.25;

pub(crate) fn section(runtime: &Runtime) {
    let rows = [
        decode(runtime, false),
        decode(runtime, true),
        starved(runtime),
        assembled(runtime),
        prefill(runtime),
        stepped(runtime),
    ];
    measure::report("attention", &rows);
}

fn decode(runtime: &Runtime, biased: bool) -> measure::Row {
    let graph = Graph::new();
    let query = plane(&graph, Shape::of([HEADS, 1, 1, WIDTH]));
    let keys = plane(&graph, Shape::of([KEY_HEADS, 1, CACHE, WIDTH]));
    let values = plane(&graph, Shape::of([KEY_HEADS, 1, CACHE, WIDTH]));
    let cursor = graph.fill(Shape::scalar(), (CACHE - 1) as f32);
    let bias = biased.then(|| graph.fill(Shape::of([HEADS, 1, 1, CACHE]), 0.25));
    let options = AttentionOptions {
        scale: None,
        causal: true,
        origin: Some(cursor),
        segments: None,
        reach: None,
        query_segments: None,
        bias,
    };
    graph.retain(graph.attention(query, keys, values, options));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let bytes = u64::from(2 * KEY_HEADS * CACHE * WIDTH) * 4
        + u64::from(biased) * u64::from(HEADS * CACHE) * 4;
    let rate = measure::traffic(timing.step, bytes, "cache");
    let label = if biased {
        format!(
            "{HEADS} heads answering one row against {CACHE} keys of {WIDTH} through a plane of bias"
        )
    } else {
        format!("{HEADS} heads answering one row against {CACHE} keys of {WIDTH}")
    };
    measure::row(label, &program, timing, rate)
}

fn starved(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let query = plane(&graph, Shape::of([1, 1, 1, WIDTH]));
    let keys = plane(&graph, Shape::of([1, 1, STARVED, WIDTH]));
    let values = plane(&graph, Shape::of([1, 1, STARVED, WIDTH]));
    let cursor = graph.fill(Shape::scalar(), (STARVED - 1) as f32);
    graph.retain(graph.attention(query, keys, values, options(Some(cursor))));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(2 * STARVED * WIDTH) * 4, "cache");
    measure::row(
        format!("one head answering one row against {STARVED} keys of {WIDTH}"),
        &program,
        timing,
        rate,
    )
}

fn assembled(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let query = plane(&graph, Shape::of([1, 1, 1, WIDTH]));
    let keys = plane(&graph, Shape::of([1, 1, STARVED, WIDTH]));
    let values = plane(&graph, Shape::of([1, 1, STARVED, WIDTH]));
    let scores = graph.matmul(query, graph.permute(keys, [0, 1, 3, 2]));
    let weighed = graph.softmax(scores);
    graph.retain(graph.matmul(weighed, values));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(timing.step, u64::from(2 * STARVED * WIDTH) * 4, "cache");
    measure::row(
        "the same walk assembled from products and a softmax",
        &program,
        timing,
        rate,
    )
}

fn prefill(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let query = plane(&graph, Shape::of([HEADS, 1, ROWS, WIDTH]));
    let keys = plane(&graph, Shape::of([KEY_HEADS, 1, CACHE, WIDTH]));
    let values = plane(&graph, Shape::of([KEY_HEADS, 1, CACHE, WIDTH]));
    let start = graph.fill(Shape::scalar(), 0.0);
    graph.retain(graph.attention(query, keys, values, options(Some(start))));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let positions = ROWS * (ROWS + 1) / 2;
    let rate = measure::traffic(
        timing.step,
        u64::from(2 * KEY_HEADS * positions * WIDTH) * 4,
        "cache",
    );
    measure::row(
        format!("{HEADS} heads prefilling {ROWS} rows against {CACHE} keys of {WIDTH}"),
        &program,
        timing,
        rate,
    )
}

fn stepped(runtime: &Runtime) -> measure::Row {
    let graph = Graph::new();
    let cache = Cache::new(&graph, KEY_HEADS, WIDTH, CACHE, 1, 1);
    let query = plane(&graph, Shape::of([HEADS, 1, 1, WIDTH]));
    let keys = plane(&graph, Shape::of([KEY_HEADS, 1, 1, WIDTH]));
    let values = plane(&graph, Shape::of([KEY_HEADS, 1, 1, WIDTH]));
    let cursor = graph.fill(Shape::scalar(), (CACHE - 1) as f32);
    let scale = graph.fill(Shape::scalar(), SCALE);
    cache.write(&graph, cursor, keys, values);
    graph.retain(cache.attend(&graph, query, cursor, scale));
    let program = measure::program(runtime, &graph, &[], &[]);
    let timing = measure::timed(runtime, &program, measure::CHEAP);
    let rate = measure::traffic(
        timing.step,
        u64::from(2 * KEY_HEADS * CACHE * WIDTH + 2 * KEY_HEADS * WIDTH) * 4,
        "cache",
    );
    measure::row(
        format!("one decode step against a cache of {CACHE} keys of {WIDTH}"),
        &program,
        timing,
        rate,
    )
}

fn options(origin: Option<Value>) -> AttentionOptions {
    AttentionOptions {
        scale: None,
        causal: true,
        origin,
        segments: None,
        reach: None,
        query_segments: None,
        bias: None,
    }
}

fn plane(graph: &Graph, shape: Shape) -> Value {
    graph.parameter(
        shape,
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    )
}
