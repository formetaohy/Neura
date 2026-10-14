use neura_abi::Element;
use neura_graph::{Graph, Shape};
use neura_precision::{pack, unpack};

use crate::support;
use support::input::random;
use support::{assert_close, open};

fn rounded(element: Element, values: &[f32]) -> Vec<f32> {
    unpack(element, values.len(), &pack(element, 1.0, values))
}

#[test]
fn a_quantized_tensor_reads_back_the_blocks_a_device_packs() {
    for shape in [Shape::matrix(5, 96), Shape::matrix(3, 33), Shape::vector(7)] {
        let runtime = open();
        let graph = Graph::new();
        let data = graph.input(shape, Element::Single);
        let blocks = graph.block_quantize(data, Element::Q8_0);
        graph.retain(blocks);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let values = random(shape.elements(), 7);
        runtime.write(&program, data, &values);
        runtime.run(&program);
        assert_eq!(
            runtime.read(&program, blocks),
            rounded(Element::Q8_0, &values),
            "a device packs {shape:?} into the blocks its host packs",
        );
    }
}

fn writing(element: Element, rows: u32, width: u32) {
    let runtime = open();
    let graph = Graph::new();
    let state = graph.resident(Shape::matrix(rows, width), element);
    let indices = graph.input(Shape::matrix(rows, 1), Element::Single);
    let updates = graph.input(Shape::matrix(rows, width), Element::Single);
    let packed = if element == Element::Single {
        updates
    } else if element.block().is_some() {
        graph.block_quantize(updates, element)
    } else {
        graph.cast(updates, element)
    };
    graph.write_into(state, indices, packed);
    graph.retain(state);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let values = random(rows * width, 11);
    let seats: Vec<f32> = (0..rows).map(|row| (row * 2 % rows) as f32).collect();
    runtime.write(&program, indices, &seats);
    runtime.write(&program, updates, &values);
    runtime.run(&program);
    let held = runtime.read(&program, state);
    let mut expected = vec![0.0f32; values.len()];
    for (row, seat) in seats.iter().enumerate() {
        let at = *seat as usize * width as usize;
        let from = row * width as usize;
        expected[at..at + width as usize]
            .copy_from_slice(&rounded(element, &values[from..from + width as usize]));
    }
    assert_eq!(
        held,
        expected,
        "a {rows}x{width} {} state holds the rows a scatter wrote",
        element.name()
    );
}

#[test]
fn a_state_takes_the_rows_a_device_writes() {
    writing(Element::Q8_0, 8, 128);
    writing(Element::Q8_0, 7, 64);
    writing(Element::Half, 8, 128);
    writing(Element::Half, 5, 34);
    writing(Element::Q8_0, 4, 192);
    writing(Element::Bfloat16, 4, 96);
    writing(Element::Single, 4, 96);
}

#[test]
fn a_narrow_cache_weighs_the_keys_and_values_a_device_packs() {
    for element in [Element::Q8_0, Element::Half] {
        let runtime = open();
        let graph = Graph::new();
        let shapes = crate::support::attention::Shapes {
            heads: 2,
            key_heads: 2,
            batch: 1,
            queries: 1,
            keys: 4,
            width: 128,
            causal: false,
            origin: 0,
            reach: 0,
            scale: 1.0,
        };
        let rows = shapes.heads * shapes.keys;
        let query = graph.input(
            Shape::of([shapes.heads, 1, 1, shapes.width]),
            Element::Single,
        );
        let indices = graph.input(Shape::matrix(rows, 1), Element::Single);
        let keyed = graph.input(Shape::matrix(rows, shapes.width), Element::Single);
        let valued = graph.input(Shape::matrix(rows, shapes.width), Element::Single);
        let held = |value| {
            if element == Element::Single {
                value
            } else if element.block().is_some() {
                graph.block_quantize(value, element)
            } else {
                graph.cast(value, element)
            }
        };
        let cache = [keyed, valued].map(|updates| {
            let stored = graph.resident(
                Shape::of([shapes.heads, 1, shapes.keys, shapes.width]),
                element,
            );
            graph.write_into(stored, indices, held(updates));
            stored
        });
        let out = graph.attention(
            query,
            cache[0],
            cache[1],
            neura_graph::AttentionOptions {
                scale: Some(graph.knob(1.0)),
                causal: false,
                origin: None,
                segments: None,
                reach: None,
                query_segments: None,
                bias: None,
            },
        );
        graph.retain(out);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        let queries = random(shapes.heads * shapes.width, 3);
        let keys = random(rows * shapes.width, 5);
        let values = random(rows * shapes.width, 13);
        runtime.write(&program, query, &queries);
        runtime.write(
            &program,
            indices,
            &(0..rows).map(|row| row as f32).collect::<Vec<_>>(),
        );
        runtime.write(&program, keyed, &keys);
        runtime.write(&program, valued, &values);
        runtime.run(&program);
        let (expected, _) = crate::support::attention::attention_forward(
            shapes,
            &queries,
            &rounded(element, &keys),
            &rounded(element, &values),
        );
        assert_close(&runtime.read(&program, out), &expected, 1e-3);
    }
}
