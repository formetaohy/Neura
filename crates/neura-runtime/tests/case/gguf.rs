use neura_abi::Element;
use neura_gguf::Gguf;
use neura_graph::{Graph, Init, Shape};
use neura_precision::{pack, unpack};

use crate::support;
use crate::support::gguf::build;
use crate::support::input;
use crate::support::reference;

use input::random;
use reference::matmul_reference;
use support::{assert_close, open};

#[test]
fn a_gguf_weight_feeds_a_product_the_blocks_its_file_packs() {
    let runtime = open();
    let values = random(256, 23);
    let payload = pack(Element::Q4K, 1.0, &values);
    let gguf = Gguf::decode(&build(32, &[(Element::Q4K, &[128, 2], &payload)]));
    let graph = Graph::new();
    let weight = graph.named_block_quantized_parameter(
        "q4_k",
        Shape::matrix(128, 2),
        Init::Zero,
        Element::Q4K,
    );
    let input = graph.input(Shape::matrix(3, 128), Element::Single);
    let product = graph.matmul(input, weight);
    graph.retain(product);
    let weights = runtime.load(&graph, &gguf);
    let program = runtime.compile(&graph, &weights);
    let inputs = random(384, 17);
    runtime.write(&program, input, &inputs);
    runtime.run(&program);

    let quantized = unpack(Element::Q4K, values.len(), &payload);
    assert_eq!(
        runtime.read(&program, weight),
        quantized,
        "a weight out of a gguf reads back the numbers its blocks place",
    );
    assert_close(
        &runtime.read(&program, product),
        &matmul_reference(&inputs, &quantized, 3, 128, 2),
        1e-4,
    );
}

#[test]
fn a_gguf_holds_every_element_a_store_reads() {
    let runtime = open();
    let held = [
        (Element::Half, 256u64),
        (Element::Q8_0, 64),
        (Element::Q5K, 256),
        (Element::Q6K, 512),
    ];
    let values = random(held.iter().map(|(_, elements)| *elements as u32).sum(), 41);
    let mut payloads = Vec::new();
    let mut cursor = 0usize;
    for (element, elements) in &held {
        let count = *elements as usize;
        payloads.push(pack(*element, 1.0, &values[cursor..cursor + count]));
        cursor += count;
    }
    let shapes = held
        .iter()
        .map(|(_, elements)| vec![*elements])
        .collect::<Vec<Vec<u64>>>();
    let tensors = held
        .iter()
        .enumerate()
        .map(|(at, (element, _))| (*element, &shapes[at][..], &payloads[at][..]))
        .collect::<Vec<_>>();
    let gguf = Gguf::decode(&build(32, &tensors));
    let graph = Graph::new();
    let mut declared = Vec::new();
    for (element, elements) in &held {
        let shape = Shape::matrix(*elements as u32, 1);
        let value = match element.block() {
            Some(_) => {
                graph.named_block_quantized_parameter(element.name(), shape, Init::Zero, *element)
            }
            None => graph.named_parameter(element.name(), shape, Init::Zero, *element),
        };
        let data = graph.input(Shape::matrix(2, *elements as u32), Element::Single);
        graph.retain(graph.matmul(data, value));
        declared.push(value);
    }
    let weights = runtime.load(&graph, &gguf);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    let mut cursor = 0usize;
    for (at, (element, elements)) in held.iter().enumerate() {
        let count = *elements as usize;
        assert_eq!(
            runtime.read(&program, declared[at]),
            unpack(*element, count, &payloads[at]),
            "a {} weight out of a gguf reads back the numbers its blocks place",
            element.name(),
        );
        cursor += count;
    }
    assert_eq!(cursor, values.len());
}

#[test]
fn a_gguf_that_lacks_a_declared_tensor_is_refused() {
    let runtime = open();
    let values = random(32, 29);
    let payload = pack(Element::Q8_0, 1.0, &values);
    let gguf = Gguf::decode(&build(32, &[(Element::Q8_0, &[32], &payload)]));
    let graph = Graph::new();
    graph.named_block_quantized_parameter("absent", Shape::vector(32), Init::Zero, Element::Q8_0);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            runtime.load(&graph, &gguf);
        }))
        .is_err(),
        "a graph that declares a tensor the gguf does not hold is refused at load",
    );
}
