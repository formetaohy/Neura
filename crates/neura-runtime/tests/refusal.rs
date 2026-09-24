use neura_program::{Graph, Init, Shape, Value};
use neura_runtime::Precision;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

fn fault<T>(action: impl FnOnce() -> T) -> String {
    let error = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)) {
        Ok(_) => panic!("the device accepted the run"),
        Err(error) => error,
    };
    error
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| {
            error
                .downcast_ref::<&str>()
                .map(|message| (*message).to_owned())
        })
        .expect("a refusal panic carries a message")
}

fn pick_graph<'g>(graph: &Graph<'g>) -> (Value<'g>, Value<'g>) {
    let table = graph.parameter(
        Shape::matrix(2, 3),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    );
    let indices = graph.input(Shape::of([4, 1]));
    let picked = graph.gather(table, indices);
    graph.retain(picked);
    (indices, picked)
}

#[test]
fn an_index_outside_the_table_refuses_the_read() {
    let runtime = open();
    let graph = Graph::new();
    let (indices, picked) = pick_graph(&graph);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, indices, &[0.0, 1.0, 2.0, 1.0]);
    runtime.run(&program);
    let message = fault(|| runtime.read(&program, picked));
    assert!(
        message.contains("outside the rows of the gather task"),
        "the read surfaced {message}",
    );
}

#[test]
fn a_faulted_state_refuses_the_checkpoint() {
    let runtime = open();
    let graph = Graph::new();
    let (indices, _picked) = pick_graph(&graph);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, indices, &[0.0, 1.0, 2.0, 1.0]);
    runtime.run(&program);
    let message = fault(|| runtime.checkpoint(&program));
    assert!(
        message.contains("outside the rows of the gather task"),
        "the checkpoint surfaced {message}",
    );
}

#[test]
fn a_refused_task_halts_the_gather_that_reads_it() {
    let runtime = open();
    let graph = Graph::new();
    let chosen = graph.input(Shape::of([2, 1]));
    let flags = graph.one_hot(chosen, 4);
    let table = graph.parameter(
        Shape::matrix(1, 3),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    );
    let listed = graph.slice(flags, 3, 0, 1);
    let picked = graph.gather(table, listed);
    graph.retain(picked);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, chosen, &[0.0, 9.0]);
    runtime.run(&program);
    let message = fault(|| runtime.read(&program, picked));
    assert!(
        message.contains("outside the rows of the one_hot task"),
        "the tape reported {message} where the first fault on it names the one_hot task",
    );
}

#[test]
fn a_nan_index_refuses_the_gather() {
    let runtime = open();
    let graph = Graph::new();
    let table = graph.parameter(
        Shape::matrix(1, 3),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
    );
    let indices = graph.input(Shape::of([4, 1]));
    let picked = graph.gather(table, indices);
    graph.retain(picked);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, indices, &[f32::NAN; 4]);
    runtime.run(&program);
    let message = fault(|| runtime.read(&program, picked));
    assert!(
        message.contains("outside the rows of the gather task"),
        "the read surfaced {message}",
    );
}

#[test]
fn a_store_without_a_fault_checkpoints_and_reads() {
    let runtime = open();
    let graph = Graph::new();
    let (indices, picked) = pick_graph(&graph);
    let weights = runtime.weights(&graph, Precision::Single);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, indices, &[0.0, 1.0, 0.0, 1.0]);
    runtime.run(&program);
    let values = runtime.read(&program, picked);
    assert_eq!(values.len(), 12);
    let checkpoint = runtime.checkpoint(&program);
    let reloaded = runtime.load(&graph, &checkpoint, Precision::Single);
    let again = runtime.compile(&graph, &reloaded);
    runtime.write(&again, indices, &[0.0, 1.0, 0.0, 1.0]);
    runtime.run(&again);
    assert_close(&runtime.read(&again, picked), &values, 0.0);
}
