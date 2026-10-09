use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Program, Runtime};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

const BOUND: u32 = 8;
const WIDTH: u32 = 4;

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

fn refusal_message(action: impl FnOnce()) -> String {
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action))
        .expect_err("the refusal a readback carries is reported");
    refused
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "a refusal without a message".to_owned())
}

struct Counted {
    runtime: Runtime,
    graph: Graph<'static>,
    count: Value<'static>,
    tokens: Value<'static>,
    total: Value<'static>,
}

impl Counted {
    fn open(bound: u32, width: u32) -> Self {
        let graph = Graph::new();
        let count = graph.input(Shape::scalar(), Element::Single);
        let tokens = graph.input(Shape::of([1, 1, bound, width]), Element::Single);
        let live = graph.trim(tokens, 2, count);
        let total = graph.sum(live);
        graph.retain(total);
        Self {
            runtime: open(),
            graph,
            count,
            tokens,
            total,
        }
    }

    fn compile(&self) -> Program {
        let weights = self.runtime.weights(&self.graph);
        self.runtime.compile(&self.graph, &weights)
    }
}

#[test]
fn a_refusal_survives_the_runs_the_host_never_reads() {
    let model = Counted::open(BOUND, WIDTH);
    let program = model.compile();
    model
        .runtime
        .write(&program, model.tokens, &data(BOUND * WIDTH, 7));
    model
        .runtime
        .write(&program, model.count, &[(BOUND + 1) as f32]);
    model.runtime.run(&program);
    model.runtime.write(&program, model.count, &[BOUND as f32]);
    model.runtime.run(&program);
    let message = refusal_message(|| {
        let _ = model.runtime.read(&program, model.total);
    });
    assert!(message.contains("refused extent"), "{message}");
}

#[test]
fn a_refusal_survives_the_binding_the_host_never_reads() {
    let runtime = open();
    let graph: Graph<'static> = Graph::new();
    let bound = graph.free(BOUND);
    let tokens = graph.input(
        Shape::of([BOUND, 1, WIDTH, WIDTH]).freed(&[(0, bound)]),
        Element::Single,
    );
    let table = graph.parameter(Shape::matrix(WIDTH, WIDTH), Init::Zero, Element::Single);
    let indices = graph.input(Shape::of([1, WIDTH, 1, 1]), Element::Single);
    let gathered = graph.gather(table, indices);
    let total = graph.sum(tokens);
    graph.retain(gathered);
    graph.retain(total);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.bind(&program, &[BOUND]);
    runtime.write(&program, tokens, &data(BOUND * WIDTH * WIDTH, 7));
    runtime.write(&program, indices, &[100.0, 1.0, 2.0, 3.0]);
    runtime.run(&program);
    runtime.bind(&program, &[WIDTH]);
    runtime.write(&program, indices, &[0.0, 1.0, 2.0, 3.0]);
    runtime.run(&program);
    let message = refusal_message(|| {
        let _ = runtime.read(&program, gathered);
    });
    assert!(message.contains("an index"), "{message}");
}

#[test]
fn a_readback_acknowledges_the_refusal_it_reports() {
    let model = Counted::open(BOUND, WIDTH);
    let program = model.compile();
    let tokens = data(BOUND * WIDTH, 7);
    model.runtime.write(&program, model.tokens, &tokens);
    model
        .runtime
        .write(&program, model.count, &[(BOUND + 1) as f32]);
    model.runtime.run(&program);
    let message = refusal_message(|| {
        let _ = model.runtime.read(&program, model.total);
    });
    assert!(message.contains("refused extent"), "{message}");
    model.runtime.write(&program, model.count, &[BOUND as f32]);
    model.runtime.run(&program);
    let actual = model.runtime.read(&program, model.total);
    let expected = tokens.iter().sum::<f32>();
    assert_close(&actual, &[expected], 1e-3);
}

#[test]
fn the_first_refusal_a_program_raises_is_the_one_it_reports() {
    let runtime = open();
    let graph = Graph::new();
    let count = graph.input(Shape::scalar(), Element::Single);
    let tokens = graph.input(Shape::of([1, 1, BOUND, WIDTH]), Element::Single);
    let mask = graph.input(Shape::of([1, 1, BOUND, 1]), Element::Single);
    let live = graph.trim(tokens, 2, count);
    let total = graph.sum(live);
    let selected = graph.compact(mask);
    graph.retain(total);
    graph.retain(selected.indices);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);

    runtime.write(&program, tokens, &data(BOUND * WIDTH, 7));
    runtime.write(&program, count, &[BOUND as f32]);
    runtime.write(&program, mask, &[1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    runtime.run(&program);

    runtime.write(&program, mask, &[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
    runtime.write(&program, count, &[(BOUND + 1) as f32]);
    runtime.run(&program);

    let message = refusal_message(|| {
        let _ = runtime.read(&program, selected.indices);
    });
    assert!(message.contains("a mask"), "{message}");
    assert!(!message.contains("refused extent"), "{message}");
}
