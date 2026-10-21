use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_nn::{Adapter, AdapterShape};
use neura_runtime::{MemoryRequest, Runtime, RuntimeRequest};

const PERSONAS: u32 = 2;
const INPUTS: u32 = 3;
const OUTPUTS: u32 = 2;
const RANK: u32 = 2;
const ROWS: u32 = 2;

fn expected(input: &[f32], base: &[f32], down: &[f32], up: &[f32], scale: f32) -> Vec<f32> {
    let hidden = (0..RANK as usize)
        .map(|rank| {
            (0..INPUTS as usize)
                .map(|step| input[step] * down[step * RANK as usize + rank])
                .sum::<f32>()
        })
        .collect::<Vec<f32>>();
    (0..OUTPUTS as usize)
        .map(|column| {
            base[column]
                + scale
                    * (0..RANK as usize)
                        .map(|rank| hidden[rank] * up[rank * OUTPUTS as usize + column])
                        .sum::<f32>()
        })
        .collect()
}

#[test]
fn a_batch_walks_the_adapter_of_each_row() {
    let runtime = Runtime::open(RuntimeRequest {
        memory: MemoryRequest {
            heap_bytes: 1 << 24,
            readback_bytes: 1 << 20,
            ..Default::default()
        },
        ..Default::default()
    })
    .expect("a device that walks adapters");
    let graph = Graph::new();
    let input = graph.input(Shape::of([1, 1, ROWS, INPUTS]), Element::Single);
    let base = graph.input(Shape::of([1, 1, ROWS, OUTPUTS]), Element::Single);
    let selection = graph.input(Shape::of([1, ROWS, 1, 1]), Element::Single);
    let adapter = Adapter::new(
        &graph,
        "persona",
        AdapterShape::new(PERSONAS, INPUTS, OUTPUTS, RANK),
        Init::Zero,
        Element::Single,
        1.0,
    );
    let out = adapter.forward(&graph, input, base, selection);
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(
        &program,
        adapter.down(),
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, -1.0],
    );
    runtime.write(
        &program,
        adapter.up(),
        &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 3.0],
    );
    runtime.write(&program, adapter.scale(), &[0.5, 2.0]);
    let inputs = [1.0, 2.0, 3.0, -1.0, 0.0, 2.0];
    let bases = [0.25, -0.5, 1.0, 2.0];
    runtime.write(&program, input, &inputs);
    runtime.write(&program, base, &bases);
    runtime.write(&program, selection, &[1.0, 0.0]);
    runtime.run(&program);
    let mut walked = runtime.read(&program, out);
    let first = expected(
        &inputs[..3],
        &bases[..2],
        &[0.0, 1.0, 1.0, 0.0, 1.0, -1.0],
        &[2.0, 0.0, 0.0, 3.0],
        2.0,
    );
    let second = expected(
        &inputs[3..],
        &bases[2..],
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        &[1.0, 0.0, 0.0, 1.0],
        0.5,
    );
    let mut expected_values = first;
    expected_values.extend(second);
    for (value, wanted) in walked.iter().zip(&expected_values) {
        assert!(
            (*value - *wanted).abs() <= 1e-5,
            "the adapter returned {value} where {wanted} was expected",
        );
    }
    runtime.write(&program, selection, &[0.0, 1.0]);
    runtime.run(&program);
    walked = runtime.read(&program, out);
    let first = expected(
        &inputs[..3],
        &bases[..2],
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        &[1.0, 0.0, 0.0, 1.0],
        0.5,
    );
    let second = expected(
        &inputs[3..],
        &bases[2..],
        &[0.0, 1.0, 1.0, 0.0, 1.0, -1.0],
        &[2.0, 0.0, 0.0, 3.0],
        2.0,
    );
    let mut expected_values = first;
    expected_values.extend(second);
    for (value, wanted) in walked.iter().zip(&expected_values) {
        assert!(
            (*value - *wanted).abs() <= 1e-5,
            "the adapter returned {value} where {wanted} was expected after the personas swapped",
        );
    }
    let loss = graph.sum(out);
    let gradients = graph.backward(loss);
    for parameter in adapter.parameters() {
        assert!(
            graph.trains(parameter),
            "every persona table trains through the gather that selects it",
        );
        assert_eq!(
            graph.shape(gradients.of(parameter)).dims(),
            graph.shape(parameter).dims(),
            "a persona table collects a gradient of its own shape",
        );
    }
}
