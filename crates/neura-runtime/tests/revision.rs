use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_program_serves_the_revision_of_the_graph_it_was_compiled_from() {
    let runtime = open();
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(4, 2), Element::Single);
    let weight = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    let out = graph.matmul(data, weight);
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    assert!(program.is_current());
    assert_eq!(program.stamp(), graph.stamp());
    runtime.write(&program, data, &[1.0; 8]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &[0.0; 12], 1e-6);
    let _ = graph.snapshot();
    assert!(program.is_current());
    runtime.run(&program);
    assert_close(&runtime.read(&program, out), &[0.0; 12], 1e-6);
}

#[test]
fn a_program_refuses_a_graph_that_moved_on() {
    let runtime = open();
    let graph = Graph::new();
    let data = graph.input(Shape::matrix(4, 2), Element::Single);
    let weight = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    let out = graph.matmul(data, weight);
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &[1.0; 8]);
    runtime.run(&program);
    let rectified = graph.relu(out);
    graph.retain(rectified);
    assert!(!program.is_current());
    assert!(program.stamp() != graph.stamp());
    assert!(refuses(|| runtime.run(&program)));
    assert!(refuses(|| runtime.write(&program, data, &[1.0; 8])));
    assert!(refuses(|| {
        let _ = runtime.read(&program, out);
    }));
    let grown = runtime.compile(&graph, &weights);
    assert!(grown.is_current());
    assert_eq!(grown.stamp(), graph.stamp());
    runtime.write(&grown, data, &[1.0; 8]);
    runtime.run(&grown);
    assert_close(&runtime.read(&grown, rectified), &[0.0; 12], 1e-6);
}

#[test]
fn a_program_of_another_graph_runs_beside_a_graph_that_moved_on() {
    let runtime = open();
    let first = Graph::new();
    let second = Graph::new();
    let first_data = first.input(Shape::vector(4), Element::Single);
    let first_out = first.relu(first_data);
    first.retain(first_out);
    let first_weights = runtime.weights(&first);
    let first_program = runtime.compile(&first, &first_weights);

    let second_data = second.input(Shape::vector(4), Element::Single);
    let second_out = second.relu(second_data);
    second.retain(second_out);
    let second_weights = runtime.weights(&second);
    let second_program = runtime.compile(&second, &second_weights);

    first.fill(Shape::vector(4), 1.0);
    assert!(!first_program.is_current());
    assert!(second_program.is_current());
    runtime.write(&second_program, second_data, &[-1.0, 2.0, -3.0, 4.0]);
    runtime.run(&second_program);
    assert_eq!(
        runtime.read(&second_program, second_out),
        vec![0.0, 2.0, 0.0, 4.0],
    );
}
