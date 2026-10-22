use neura_abi::Element;
use neura_graph::{Graph, Shape};
use neura_runtime::Runtime;

const PLANES: u32 = 2;
const BOUND: u32 = 1024;
const COLUMNS: u32 = 5;
const LENGTHS: [f32; 2] = [600.0, 424.0];

#[test]
fn a_segment_sum_adds_the_rows_of_every_chunk_of_every_plane() {
    let runtime = Runtime::open(Default::default()).expect("a device");
    let graph = Graph::new();
    let lengths = graph.input(Shape::vector(PLANES), Element::Single);
    let ragged = graph.ragged(BOUND, lengths);
    let values = graph.input(
        Shape::of([1, 1, BOUND, COLUMNS]).freed(&[(2, ragged.extent)]),
        Element::Single,
    );
    let summed = graph.segment_sum(values, ragged);
    graph.retain(summed);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let held = (0..BOUND * COLUMNS)
        .map(|index| index as f32 * 0.25 - 1.0)
        .collect::<Vec<f32>>();
    runtime.write(&program, values, &held);
    runtime.write(&program, lengths, &LENGTHS);
    runtime.run(&program);
    let walked = runtime.read(&program, summed);
    let mut expected = Vec::new();
    let mut start = 0usize;
    for length in LENGTHS {
        for column in 0..COLUMNS as usize {
            let mut total = 0.0f32;
            for row in 0..length as usize {
                total += held[(start + row) * COLUMNS as usize + column];
            }
            expected.push(total);
        }
        start += length as usize;
    }
    assert_eq!(walked.len(), expected.len());
    for (index, (actual, expected)) in walked.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 0.05,
            "element {index} came back as {actual} where {expected} was expected",
        );
    }
}
