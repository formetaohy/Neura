use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_runtime::Checkpoint;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn assemble(header: &str, data: &[u8]) -> Vec<u8> {
    let mut header = header.as_bytes().to_vec();
    let padded = header.len().next_multiple_of(8);
    header.resize(padded, b' ');
    let mut bytes = Vec::with_capacity(8 + padded + data.len());
    bytes.extend_from_slice(&(padded as u64).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(data);
    bytes
}

fn header(bytes: &[u8]) -> &str {
    let length = u64::from_le_bytes(bytes[..8].try_into().expect("a length word")) as usize;
    assert!(bytes.len() >= 8 + length, "a container holds its header");
    std::str::from_utf8(&bytes[8..8 + length]).expect("a container header holds UTF-8")
}

fn data(bytes: &[u8]) -> &[u8] {
    let length = u64::from_le_bytes(bytes[..8].try_into().expect("a length word")) as usize;
    &bytes[8 + length..]
}

fn half(value: f32) -> [u8; 2] {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = (((bits >> 23) & 0xff) as i32 - 127 + 15) as u16;
    let mantissa = ((bits >> 13) & 0x03ff) as u16;
    (sign | (exponent << 10) | mantissa).to_le_bytes()
}

#[test]
fn a_container_follows_the_safetensors_layout() {
    let runtime = open();
    let graph = Graph::new();
    graph.named_parameter("beta", Shape::matrix(4, 5), Init::Zero, Element::Half);
    graph.named_parameter("alpha", Shape::vector(4), Init::Zero, Element::Single);
    graph.named_block_quantized_parameter("packed", Shape::matrix(4, 8), Init::Zero, Element::Int4);
    let checkpoint = runtime.checkpoint(&runtime.weights(&graph));
    let bytes = checkpoint.bytes();
    let length = u64::from_le_bytes(bytes[..8].try_into().expect("a length word")) as usize;
    assert_eq!(length % 8, 0, "a container pads its header to eight bytes");
    let text = header(bytes);
    assert!(text.starts_with('{'), "a container header holds JSON");
    assert!(
        text.trim_end().ends_with('}'),
        "a container header holds one JSON object",
    );
    assert!(
        text.contains(r#""alpha":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}"#),
        "a container lists its tensors by dtype, shape and offsets: {text}",
    );
    assert!(
        text.contains(r#""beta":{"dtype":"F16","shape":[4,5],"data_offsets":[20,60]}"#),
        "a container follows its widest tensors with the narrower: {text}",
    );
    assert!(
        text.contains(r#""neura.element.packed":"int4""#),
        "a container names the element of a packed tensor: {text}",
    );
    assert!(
        text.contains(r#""neura.elements.packed":"32""#),
        "a container counts the numbers of a packed tensor: {text}",
    );
    assert!(
        text.contains(r#""packed":{"dtype":"U8","shape":[16],"data_offsets":[60,76]}"#),
        "a container packs four bit numbers into their bytes: {text}",
    );
    assert!(
        text.contains(r#""packed.quanta":{"dtype":"F32","shape":[1],"data_offsets":[16,20]}"#),
        "a container follows a packed tensor with its table: {text}",
    );
    assert_eq!(
        data(bytes).len(),
        76,
        "a container indexes every byte of its buffer",
    );
    assert_eq!(checkpoint.tensors(), 4);
}

#[test]
fn a_container_written_by_another_framework_loads_by_name() {
    let values: Vec<f32> = (0..6).map(|value| value as f32 * 0.5).collect();
    let mut payload = Vec::new();
    for value in &values {
        payload.extend_from_slice(&value.to_le_bytes());
    }
    for value in [0.5f32, -1.0] {
        payload.extend_from_slice(&half(value));
    }
    let bytes = assemble(
        r#"{"weight":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},"half":{"dtype":"F16","shape":[2],"data_offsets":[24,28]},"unused":{"dtype":"F64","shape":[1],"data_offsets":[28,36]}}"#,
        &[payload.clone(), vec![0u8; 8]].concat(),
    );
    let checkpoint = Checkpoint::decode(&bytes);
    assert_eq!(checkpoint.tensors(), 3);
    let weight = checkpoint.tensor("weight").expect("a named tensor");
    assert_eq!(weight.element, Some(Element::Single));
    assert_eq!(weight.elements, 6);
    assert_eq!(weight.scale, 1.0);
    assert_eq!(weight.quanta, None);
    assert!(checkpoint.tensor("unused").is_some());

    let runtime = open();
    let graph = Graph::new();
    let input = graph.input(Shape::matrix(1, 2), Element::Single);
    let weight = graph.named_parameter("weight", Shape::matrix(2, 3), Init::Zero, Element::Single);
    let half = graph.named_parameter("half", Shape::vector(2), Init::Zero, Element::Half);
    let product = graph.matmul(input, weight);
    graph.retain(product);
    graph.retain(half);
    let weights = runtime.load(&graph, &checkpoint);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, input, &[1.0, 1.0]);
    runtime.run(&program);
    let expected = [
        values[0] + values[3],
        values[1] + values[4],
        values[2] + values[5],
    ];
    assert_close(&runtime.read(&program, product), &expected, 1e-6);
    assert_close(&runtime.read(&program, half), &[0.5, -1.0], 1e-6);
}

#[test]
fn a_container_the_reference_implementation_wrote_loads_by_name() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/reference.safetensors"
    );
    let bytes = std::fs::read(path).expect("the reference container beside the test");
    let checkpoint = Checkpoint::decode(&bytes);
    assert_eq!(checkpoint.tensors(), 2);
    let weight = checkpoint.tensor("weight").expect("a named tensor");
    assert_eq!(weight.element, Some(Element::Single));
    assert_eq!(weight.elements, 4);
    assert_eq!(weight.payload.len(), 16);
    let half = checkpoint.tensor("half").expect("a named tensor");
    assert_eq!(half.element, Some(Element::Half));
    assert_eq!(half.elements, 2);

    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_parameter("weight", Shape::matrix(2, 2), Init::Zero, Element::Single);
    let read = graph.mul(weight, graph.fill(Shape::scalar(), 1.0));
    graph.retain(read);
    let weights = runtime.load(&graph, &checkpoint);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    assert_close(&runtime.read(&program, read), &[0.25, 0.5, 0.75, 1.0], 1e-6);
}

#[test]
fn a_container_keeps_names_that_need_escaping() {
    let runtime = open();
    let name = "block \"0\" \\ \u{3b2}";
    let graph = Graph::new();
    let weight = graph.named_parameter(name, Shape::vector(3), Init::Zero, Element::Single);
    let read = graph.mul(weight, graph.fill(Shape::scalar(), 1.0));
    graph.retain(read);
    let weights = runtime.weights(&graph);
    let checkpoint = runtime.checkpoint(&weights);
    assert_eq!(checkpoint.names().collect::<Vec<&str>>(), vec![name]);

    let decoded = Checkpoint::decode(checkpoint.bytes());
    assert_eq!(decoded.names().collect::<Vec<&str>>(), vec![name]);
    let weights = runtime.load(&graph, &decoded);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, weight, &[1.0, 2.0, 3.0]);
    runtime.run(&program);
    assert_close(&runtime.read(&program, read), &[1.0, 2.0, 3.0], 1e-6);
}

#[test]
fn a_packed_tensor_keeps_an_odd_number_of_nibbles() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_block_quantized_parameter(
        "weight",
        Shape::vector(3),
        Init::Constant(1.5),
        Element::Int4,
    );
    let read = graph.mul(weight, graph.fill(Shape::scalar(), 1.0));
    graph.retain(read);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.run(&program);
    let before = runtime.read(&program, read);
    let checkpoint = runtime.checkpoint(&weights);
    let tensor = checkpoint.tensor("weight").expect("a named tensor");
    assert_eq!(tensor.element, Some(Element::Int4));
    assert_eq!(tensor.elements, 3);
    assert_eq!(tensor.payload.len(), 2);
    assert_eq!(tensor.quanta.expect("a table").len(), 4);

    let reloaded = runtime.load(&graph, &checkpoint);
    let program = runtime.compile(&graph, &reloaded);
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, read),
        before,
        "a store of three four bit numbers packs their last nibble",
    );
}

#[test]
fn a_container_refuses_bytes_its_header_does_not_index() {
    let trailing = assemble(
        r#"{"weight":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#,
        &[0u8; 12],
    );
    assert!(
        refuses(|| {
            Checkpoint::decode(&trailing);
        }),
        "a container indexes every byte it holds",
    );
    let overlapping = assemble(
        r#"{"one":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"two":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
        &[0u8; 4],
    );
    assert!(
        refuses(|| {
            Checkpoint::decode(&overlapping);
        }),
        "two tensors of one container hold disjoint bytes",
    );
    let short = assemble(
        r#"{"weight":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
        &[0u8; 8],
    );
    assert!(
        refuses(|| {
            Checkpoint::decode(&short);
        }),
        "a container holds as many bytes as its shape declares numbers",
    );
    let duplicated = assemble(
        r#"{"weight":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"weight":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}"#,
        &[0u8; 8],
    );
    assert!(
        refuses(|| {
            Checkpoint::decode(&duplicated);
        }),
        "a name identifies one tensor of a container",
    );
}

#[test]
fn a_container_without_a_quantum_loads_as_one() {
    let bytes = assemble(
        r#"{"weight":{"dtype":"I8","shape":[4],"data_offsets":[0,4]}}"#,
        &[2u8, 4, 6, 8],
    );
    let checkpoint = Checkpoint::decode(&bytes);
    let weight = checkpoint.tensor("weight").expect("a named tensor");
    assert_eq!(weight.element, Some(Element::Int8));
    assert_eq!(weight.scale, 1.0);
    assert_eq!(weight.payload, &[2u8, 4, 6, 8]);

    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_quantized_parameter("weight", Shape::vector(4), Init::Zero, 2.0);
    let read = graph.mul(weight, graph.fill(Shape::scalar(), 1.0));
    graph.retain(read);
    assert!(
        refuses(|| {
            runtime.load(&graph, &checkpoint);
        }),
        "a container of no quantum loads only a graph that reconstructs by one",
    );
    let ones = Graph::new();
    let weight = ones.named_quantized_parameter("weight", Shape::vector(4), Init::Zero, 1.0);
    let doubled = ones.mul(weight, ones.fill(Shape::scalar(), 2.0));
    ones.retain(doubled);
    let weights = runtime.load(&ones, &checkpoint);
    let program = runtime.compile(&ones, &weights);
    runtime.run(&program);
    assert_close(
        &runtime.read(&program, doubled),
        &[4.0, 8.0, 12.0, 16.0],
        1e-6,
    );
}

#[test]
fn a_container_refuses_a_reserved_name() {
    let runtime = open();
    let graph = Graph::new();
    graph.named_parameter(
        "__metadata__",
        Shape::vector(2),
        Init::Zero,
        Element::Single,
    );
    let weights = runtime.weights(&graph);
    assert!(
        refuses(|| {
            runtime.checkpoint(&weights);
        }),
        "a container reserves __metadata__ for its own keys",
    );
}

#[test]
fn a_container_of_an_empty_store_decodes_empty() {
    let bytes = assemble("{}", &[]);
    let checkpoint = Checkpoint::decode(&bytes);
    assert_eq!(checkpoint.tensors(), 0);
    assert_eq!(checkpoint.names().count(), 0);
}

#[test]
fn a_container_round_trips_through_its_own_bytes() {
    let runtime = open();
    let graph = Graph::new();
    graph.named_parameter("weight", Shape::matrix(4, 4), Init::Zero, Element::Single);
    graph.named_quantized_parameter("quantized", Shape::vector(4), Init::Zero, 0.25);
    let weights = runtime.weights(&graph);
    let checkpoint = runtime.checkpoint(&weights);
    let decoded = Checkpoint::decode(checkpoint.bytes());
    assert_eq!(decoded.tensors(), checkpoint.tensors());
    assert_eq!(
        decoded.names().collect::<Vec<&str>>(),
        checkpoint.names().collect::<Vec<&str>>(),
    );
    for name in checkpoint.names() {
        let left = checkpoint.tensor(name).expect("a named tensor");
        let right = decoded.tensor(name).expect("a named tensor");
        assert_eq!(left.element, right.element);
        assert_eq!(left.elements, right.elements);
        assert_eq!(left.scale, right.scale);
        assert_eq!(left.payload, right.payload);
        assert_eq!(left.quanta, right.quanta);
    }
}
