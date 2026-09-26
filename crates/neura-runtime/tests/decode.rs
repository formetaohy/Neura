use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/attention.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use reference::{Shapes, attention_backward, attention_forward};
use support::{assert_close, open};

const TOKENS: u32 = 5;
const CAPACITY: u32 = 8;
const WIDTH: u32 = 4;
const SCALE: f32 = 0.5;

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

fn options(cursor: Value<'static>) -> AttentionOptions<'static> {
    AttentionOptions {
        scale: SCALE,
        causal: true,
        origin: Some(cursor),
    }
}

fn cached_shapes(queries: u32, origin: u32) -> Shapes {
    Shapes {
        heads: 1,
        batch: 1,
        queries,
        keys: CAPACITY,
        width: WIDTH,
        causal: true,
        origin,
        scale: SCALE,
    }
}

struct Decoder {
    runtime: Runtime,
    graph: Graph<'static>,
    cursor: Value<'static>,
    slot: Value<'static>,
    row: Value<'static>,
    query: Value<'static>,
    out: Value<'static>,
}

impl Decoder {
    fn open(agents: u32) -> Self {
        let graph: Graph<'static> = Graph::new();
        let keys = graph.resident(Shape::of([agents, 1, CAPACITY, WIDTH]), Element::Single);
        let values = graph.resident(Shape::of([agents, 1, CAPACITY, WIDTH]), Element::Single);
        let cursor = graph.input(Shape::of([agents, 1, 1, 1]), Element::Single);
        let slot = graph.input(Shape::of([agents, 1, 1, 1]), Element::Single);
        let row = graph.input(Shape::of([agents, 1, 1, WIDTH]), Element::Single);
        let query = graph.input(Shape::of([agents, 1, 1, WIDTH]), Element::Single);
        let doubled = graph.mul(row, graph.fill(Shape::scalar(), 2.0));
        graph.write_into(keys, slot, doubled);
        graph.write_into(values, slot, doubled);
        let out = graph.attention(query, keys, values, options(cursor));
        graph.retain(out);
        Self {
            runtime: open(),
            graph,
            cursor,
            slot,
            row,
            query,
            out,
        }
    }

    fn compile(&self) -> Program<'_> {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }

    fn step(
        &self,
        program: &Program<'_>,
        cursors: &[f32],
        rows: &[f32],
        queries: &[f32],
    ) -> Vec<f32> {
        self.runtime.write(program, self.cursor, cursors);
        self.runtime.write(program, self.slot, &slots(cursors));
        self.runtime.write(program, self.row, rows);
        self.runtime.write(program, self.query, queries);
        self.runtime.run(program);
        self.runtime.read(program, self.out)
    }
}

fn slots(cursors: &[f32]) -> Vec<f32> {
    cursors
        .iter()
        .enumerate()
        .map(|(agent, cursor)| agent as f32 * CAPACITY as f32 + cursor)
        .collect()
}

#[test]
fn a_decode_step_reads_the_keys_its_cursor_reaches() {
    let decoder = Decoder::open(1);
    let program = decoder.compile();
    let tokens = data(TOKENS * WIDTH, 17);
    let queries = data(TOKENS * WIDTH, 41);
    let cached = tokens.iter().map(|value| value * 2.0).collect::<Vec<_>>();
    let (expected, _) = attention_forward(
        Shapes {
            heads: 1,
            batch: 1,
            queries: TOKENS,
            keys: TOKENS,
            width: WIDTH,
            causal: true,
            origin: 0,
            scale: SCALE,
        },
        &queries,
        &cached,
        &cached,
    );
    let width = WIDTH as usize;
    let pass = |program: &Program<'_>| {
        for step in 0..TOKENS {
            let at = step as usize * width;
            let produced = decoder.step(
                program,
                &[step as f32],
                &tokens[at..at + width],
                &queries[at..at + width],
            );
            assert_close(&produced, &expected[at..at + width], 1e-4);
        }
    };
    pass(&program);
    pass(&program);
}

#[test]
fn a_decode_gradient_reaches_the_keys_a_cursor_exposes() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let keys = graph.parameter(
        Shape::of([1, 1, CAPACITY, WIDTH]),
        Init::Zero,
        Element::Single,
    );
    let values = graph.parameter(
        Shape::of([1, 1, CAPACITY, WIDTH]),
        Init::Zero,
        Element::Single,
    );
    let cursor = graph.input(Shape::scalar(), Element::Single);
    let query = graph.parameter(Shape::of([1, 1, 1, WIDTH]), Init::Zero, Element::Single);
    let out = graph.attention(query, keys, values, options(cursor));
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    let query_grad = gradients.of(query);
    let key_grad = gradients.of(keys);
    let value_grad = gradients.of(values);
    graph.retain(query_grad);
    graph.retain(key_grad);
    graph.retain(value_grad);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let query_row = data(WIDTH, 41);
    let cache = data(CAPACITY * WIDTH, 17);
    runtime.write(&program, keys, &cache);
    runtime.write(&program, values, &cache);
    runtime.write(&program, query, &query_row);
    for step in 0..TOKENS {
        let shapes = cached_shapes(1, step);
        let (produced_row, statistics) = attention_forward(shapes, &query_row, &cache, &cache);
        let (expected_query, expected_key, expected_value) = attention_backward(
            shapes,
            &query_row,
            &cache,
            &cache,
            &produced_row,
            &statistics,
            &[1.0; WIDTH as usize],
        );
        runtime.write(&program, cursor, &[step as f32]);
        runtime.run(&program);
        let produced = runtime.read_many(&program, &[query_grad, key_grad, value_grad]);
        assert_close(&produced[0], &expected_query, 1e-4);
        assert_close(&produced[1], &expected_key, 1e-4);
        assert_close(&produced[2], &expected_value, 1e-4);
    }
}

#[test]
fn a_cursor_past_the_keys_leaves_no_room_for_the_block() {
    let decoder = Decoder::open(1);
    let program = decoder.compile();
    let row = data(WIDTH, 17);
    let query = data(WIDTH, 41);
    let produced = decoder.step(&program, &[0.0], &row, &query);
    assert_eq!(produced.len(), WIDTH as usize);
    for cursor in [CAPACITY as f32, 0.5, -1.0] {
        assert!(refuses(|| {
            decoder.runtime.write(&program, decoder.cursor, &[cursor]);
            decoder.runtime.run(&program);
            let _ = decoder.runtime.read(&program, decoder.out);
        }));
    }
}

#[test]
fn a_cursor_frees_each_plane_at_its_own_position() {
    let agents = 2;
    let rows = TOKENS + agents;
    let decoder = Decoder::open(agents);
    let program = decoder.compile();
    let tokens = data(agents * rows * WIDTH, 17);
    let queries = data(agents * rows * WIDTH, 41);
    let width = WIDTH as usize;
    let above = |values: &[f32], agent: u32, position: u32| {
        let at = (agent * rows + position) as usize * width;
        values[at..at + width].to_vec()
    };
    for step in 0..TOKENS {
        let cursors = (0..agents)
            .map(|agent| (step + agent) as f32)
            .collect::<Vec<_>>();
        let written = (0..agents)
            .flat_map(|agent| above(&tokens, agent, step + agent))
            .collect::<Vec<_>>();
        let asked = (0..agents)
            .flat_map(|agent| above(&queries, agent, step + agent))
            .collect::<Vec<_>>();
        let produced = decoder.step(&program, &cursors, &written, &asked);
        for agent in 0..agents {
            let cursor = step + agent;
            let mut cached = vec![0.0f32; CAPACITY as usize * width];
            for position in agent..=cursor {
                let offset = position as usize * width;
                let row = above(&tokens, agent, position);
                for (slot, value) in cached[offset..offset + width].iter_mut().zip(row) {
                    *slot = value * 2.0;
                }
            }
            let (expected, _) = attention_forward(
                cached_shapes(1, cursor),
                &above(&queries, agent, cursor),
                &cached,
                &cached,
            );
            let at = agent as usize * width;
            assert_close(&produced[at..at + width], &expected, 1e-4);
        }
    }
}
