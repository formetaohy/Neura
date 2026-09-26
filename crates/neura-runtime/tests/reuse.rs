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

#[test]
fn a_program_is_compiled_by_the_call_that_compiles_it() {
    let runtime = open();
    let graph = Graph::new();
    let sensor = sensor(&runtime, &graph, 16);
    assert!(sensor.program.is_compiled());
    let assembled = runtime.assembled_kernels();
    for _ in 0..4 {
        runtime.run(&sensor.program);
        assert_eq!(
            runtime.assembled_kernels(),
            assembled,
            "a run only runs what a compile has compiled",
        );
    }
}

#[test]
fn a_shape_the_carried_geometry_covers_assembles_no_device_program() {
    let runtime = open();
    let both = Graph::new();
    let shared = both.parameter(
        Shape::matrix(4, 2),
        Init::Uniform {
            low: 0.25,
            high: 0.75,
        },
    );
    let small = both.input(Shape::matrix(8, 4));
    let large = both.input(Shape::matrix(64, 4));
    both.retain(both.matmul(small, shared));
    both.retain(both.matmul(large, shared));
    let weights = runtime.weights(&both, Precision::Single);
    runtime.compile(&both, &weights);
    let assembled = runtime.assembled_kernels();
    let programs = runtime.declared_kernels();
    assert_eq!(
        programs, 1,
        "one device program carries every tile of a model"
    );
    for samples in [8u32, 64] {
        let graph = Graph::new();
        let weight = graph.parameter(Shape::matrix(4, 2), Init::Zero);
        let input = graph.input(Shape::matrix(samples, 4));
        let out = graph.matmul(input, weight);
        graph.retain(out);
        let weights = runtime.weights(&graph, Precision::Single);
        let program = runtime.compile(&graph, &weights);
        assert_eq!(
            runtime.assembled_kernels(),
            assembled,
            "a shape whose tiles a carried geometry holds assembles no device program",
        );
        assert_eq!(
            runtime.declared_kernels(),
            programs,
            "a shape whose tiles a carried geometry holds compiles no device program",
        );
        let input_data = random(samples * 4, samples);
        let weight_data = random(8, 3);
        runtime.write(&program, input, &input_data);
        runtime.write(&program, weight, &weight_data);
        runtime.run(&program);
        assert_close(
            &runtime.read(&program, out),
            &matmul_reference(&input_data, &weight_data, samples, 4, 2),
            1e-5,
        );
    }
}

#[test]
fn many_runs_in_flight_keep_the_data_of_every_program() {
    let runtime = open();
    let shapes = [8u32, 16, 32];
    let mut sensors = Vec::new();
    let mut expected = Vec::new();
    for (index, samples) in shapes.iter().enumerate() {
        let graph = Graph::new();
        let sensor = sensor(&runtime, &graph, *samples);
        let input = random(*samples * 4, index as u32 + 1);
        let weight = random(8, index as u32 + 7);
        runtime.write(&sensor.program, sensor.input, &input);
        runtime.write(&sensor.program, sensor.weight, &weight);
        expected.push(matmul_reference(&input, &weight, *samples, 4, 2));
        sensors.push((sensor, *samples));
    }
    for _ in 0..24 {
        for (sensor, _) in &sensors {
            runtime.run(&sensor.program);
        }
    }
    for (index, (sensor, _)) in sensors.iter().enumerate() {
        assert_close(
            &runtime.read(&sensor.program, sensor.out),
            &expected[index],
            1e-4,
        );
    }
    let (sensor, samples) = &sensors[0];
    let input = random(samples * 4, 11);
    let weight = random(8, 13);
    runtime.write(&sensor.program, sensor.input, &input);
    runtime.write(&sensor.program, sensor.weight, &weight);
    runtime.run(&sensor.program);
    assert_close(
        &runtime.read(&sensor.program, sensor.out),
        &matmul_reference(&input, &weight, *samples, 4, 2),
        1e-4,
    );
}
