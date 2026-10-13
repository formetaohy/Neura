use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_precision::{pack, unpack};

use crate::support;
use crate::support::input;

use input::random;
use support::{assert_close, open};

#[test]
fn a_gather_reads_the_rows_of_a_block_table() {
    let runtime = open();
    let (rows, width) = (64u32, 32u32);
    let table = pack(Element::Q8_0, 1.0, &random(rows * width, 11));
    let graph = Graph::new();
    let weights_table = graph.named_block_quantized_parameter(
        "token_embd.weight",
        Shape::matrix(rows, width),
        Init::Zero,
        Element::Q8_0,
    );
    let indices = graph.input(Shape::matrix(3, 1), Element::Single);
    let picked = graph.gather(weights_table, indices);
    graph.retain(picked);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let index_data = vec![5.0, 0.0, 63.0];
    runtime.write(&program, weights_table, &random(rows * width, 11));
    runtime.write(&program, indices, &index_data);
    runtime.run(&program);
    let table_values = unpack(Element::Q8_0, (rows * width) as usize, &table);
    let mut want = Vec::new();
    for index in &index_data {
        let start = *index as usize * width as usize;
        want.extend_from_slice(&table_values[start..start + width as usize]);
    }
    assert_close(&runtime.read(&program, picked), &want, 0.0);
}

#[test]
fn a_product_reads_a_block_table_through_the_view_it_transposes() {
    let runtime = open();
    let (rows, width) = (64u32, 32u32);
    let values = random(rows * width, 13);
    let table = pack(Element::Q8_0, 1.0, &values);
    let graph = Graph::new();
    let weights_table = graph.named_block_quantized_parameter(
        "token_embd.weight",
        Shape::matrix(rows, width),
        Init::Zero,
        Element::Q8_0,
    );
    let hidden = graph.input(Shape::matrix(1, width), Element::Single);
    let logits = graph.matmul(hidden, graph.permute(weights_table, [0, 1, 3, 2]));
    graph.retain(logits);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let hidden_data = random(width, 17);
    runtime.write(&program, weights_table, &values);
    runtime.write(&program, hidden, &hidden_data);
    runtime.run(&program);
    let table_values = unpack(Element::Q8_0, (rows * width) as usize, &table);
    let want = (0..rows as usize)
        .map(|row| {
            let start = row * width as usize;
            hidden_data
                .iter()
                .zip(&table_values[start..start + width as usize])
                .map(|(left, right)| left * right)
                .sum::<f32>()
        })
        .collect::<Vec<f32>>();
    assert_close(&runtime.read(&program, logits), &want, 1e-4);
}
