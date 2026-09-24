use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Precision, Runtime};

#[path = "support/reference.rs"]
mod reference;
#[path = "support/mod.rs"]
mod support;

use reference::{matmul_reference, random};
use support::{assert_close, open};

struct Sensor<'g> {
    program: neura_runtime::Program<'g>,
    input: Value<'g>,
    out: Value<'g>,
    weight: Value<'g>,
}

fn sensor<'g>(runtime: &'g Runtime, graph: &Graph<'g>, samples: u32) -> Sensor<'g> {
    let input = graph.input(Shape::matrix(samples, 4));
    let weight = graph.parameter(
        Shape::matrix(4, 2),
        Init::Uniform {
            low: 0.25,
            high: 0.75,
        },
    );
    let out = graph.matmul(input, weight);
    graph.retain(out);
    let weights = runtime.weights(graph, Precision::Single);
    let program = runtime.compile(graph, &weights);
    Sensor {
        program,
        input,
        out,
        weight,
    }
}

#[test]
fn a_recompile_of_one_shape_shares_the_device_tape_but_not_the_tensors() {
    let runtime = open();
    let before = runtime.device_tapes();
    let first_graph = Graph::new();
    let second_graph = Graph::new();
    let first = sensor(&runtime, &first_graph, 16);
    let second = sensor(&runtime, &second_graph, 16);
    assert_eq!(
        runtime.device_tapes(),
        before + 1,
        "two compilations of one shape run on one device tape",
    );
    let first_input = random(16 * 4, 3);
    let second_input = random(16 * 4, 5);
    runtime.write(&first.program, first.input, &first_input);
    runtime.write(&second.program, second.input, &second_input);
    runtime.run(&first.program);
    runtime.run(&second.program);
    let first_weight = runtime.read(&first.program, first.weight);
    let second_weight = runtime.read(&second.program, second.weight);
    assert_close(
        &runtime.read(&first.program, first.out),
        &matmul_reference(&first_input, &first_weight, 16, 4, 2),
        1e-5,
    );
    assert_close(
        &runtime.read(&second.program, second.out),
        &matmul_reference(&second_input, &second_weight, 16, 4, 2),
        1e-5,
    );
}

#[test]
fn a_session_of_varying_batches_keeps_one_tape_per_shape() {
    let runtime = open();
    let before = runtime.device_tapes();
    let graphs: Vec<Graph> = [8u32, 16, 8, 24, 16].iter().map(|_| Graph::new()).collect();
    let mut programs = Vec::new();
    for (index, (graph, samples)) in graphs.iter().zip([8u32, 16, 8, 24, 16]).enumerate() {
        let sensor = sensor(&runtime, graph, samples);
        let input = random(samples * 4, 61 + index as u32);
        runtime.write(&sensor.program, sensor.input, &input);
        runtime.run(&sensor.program);
        let weight = runtime.read(&sensor.program, sensor.weight);
        assert_close(
            &runtime.read(&sensor.program, sensor.out),
            &matmul_reference(&input, &weight, samples, 4, 2),
            1e-5,
        );
        programs.push(sensor.program);
    }
    assert_eq!(
        runtime.device_tapes(),
        before + 3,
        "a batch of three shapes keeps three device tapes",
    );
}

#[test]
fn a_dropped_program_recycles_its_device_tape() {
    let runtime = open();
    let before = runtime.device_tapes();
    let graph = Graph::new();
    {
        let sensor = sensor(&runtime, &graph, 12);
        let input = random(12 * 4, 9);
        runtime.write(&sensor.program, sensor.input, &input);
        runtime.run(&sensor.program);
        let weight = runtime.read(&sensor.program, sensor.weight);
        assert_close(
            &runtime.read(&sensor.program, sensor.out),
            &matmul_reference(&input, &weight, 12, 4, 2),
            1e-5,
        );
    }
    assert_eq!(
        runtime.device_tapes(),
        before,
        "a program nobody holds leaves no device tape behind"
    );
    let again = sensor(&runtime, &graph, 12);
    let input = random(12 * 4, 13);
    runtime.write(&again.program, again.input, &input);
    runtime.run(&again.program);
    let weight = runtime.read(&again.program, again.weight);
    assert_close(
        &runtime.read(&again.program, again.out),
        &matmul_reference(&input, &weight, 12, 4, 2),
        1e-5,
    );
}
