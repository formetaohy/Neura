use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_quantized_tensor_carries_its_scale_through_the_ops_it_feeds() {
    let graph = Graph::new();
    let weight = graph.quantized_parameter(Shape::matrix(4, 4), Init::Zero, 0.125);
    let observations = graph.input(Shape::matrix(2, 4), Element::Single);
    assert_eq!(graph.element(weight), Element::Int8);
    assert_eq!(graph.scale(weight), 0.125);
    let rectified = graph.relu(weight);
    assert_eq!(graph.element(rectified), Element::Int8);
    assert_eq!(
        graph.scale(rectified),
        0.125,
        "a unary op over a quantized tensor lands in the storage it reads",
    );
    let product = graph.matmul(observations, weight);
    assert_eq!(graph.element(product), Element::Single);
    assert_eq!(
        graph.scale(product),
        1.0,
        "a single precision tensor reconstructs nothing",
    );
    let halves = graph.matmul(graph.cast(observations, Element::Half), weight);
    assert_eq!(graph.element(halves), Element::Half);
    assert_eq!(
        graph.scale(halves),
        1.0,
        "a product takes the scale of the operand that carries its storage",
    );
    let turned = graph.permute(weight, [0, 1, 3, 2]);
    assert_eq!(graph.element(turned), Element::Int8);
    assert_eq!(
        graph.scale(turned),
        0.125,
        "a view reconstructs the numbers of the storage it walks",
    );
    assert_eq!(graph.element(graph.mul(weight, weight)), Element::Single);
}

#[test]
fn a_quantized_tensor_weighs_no_gradient() {
    let graph = Graph::new();
    let quantized = graph.quantized_parameter(Shape::matrix(4, 4), Init::Zero, 0.125);
    let weight = graph.parameter(Shape::matrix(4, 4), Init::Zero, Element::Single);
    let observations = graph.input(Shape::matrix(2, 4), Element::Single);
    let loss = graph.sum(graph.matmul(observations, weight));
    let gradients = graph.backward(loss);
    assert!(
        refuses(|| {
            gradients.of(quantized);
        }),
        "a quantized tensor carries the storage no gradient reaches",
    );
    assert_eq!(gradients.of(weight).shape(), Shape::matrix(4, 4));

    let alone = Graph::new();
    let quantized = alone.quantized_parameter(Shape::vector(4), Init::Zero, 0.125);
    let loss = alone.sum(alone.relu(quantized));
    assert!(
        refuses(|| {
            alone.backward(loss);
        }),
        "a loss that derives from a quantized tensor alone derives from no parameter",
    );
}

#[test]
fn a_cast_refuses_a_quantized_storage() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    assert!(
        refuses(|| {
            graph.cast(data, Element::Int8);
        }),
        "a cast into int8 storage quantizes by a scale it does not declare",
    );
    assert!(
        refuses(|| {
            graph.parameter(Shape::vector(4), Init::Zero, Element::Int8);
        }),
        "a parameter of int8 storage is declared with the scale it reconstructs by",
    );
    assert!(
        refuses(|| {
            graph.input(Shape::vector(4), Element::Int8);
        }),
        "an input of int8 storage is declared with the scale it reconstructs by",
    );
    assert!(
        refuses(|| {
            graph.quantize(data, 0.0);
        }),
        "an int8 tensor of scale zero reconstructs nothing",
    );
    assert_eq!(graph.element(graph.quantize(data, 0.25)), Element::Int8);
}

#[test]
fn a_block_quantized_parameter_carries_the_quantum_of_every_block_it_packs() {
    let graph = Graph::new();
    let weight = graph.block_quantized_parameter(Shape::matrix(300, 4), Init::Zero, Element::Int4);
    assert_eq!(graph.element(weight), Element::Int4);
    assert_eq!(
        graph.element(graph.permute(weight, [0, 1, 3, 2])),
        Element::Int4,
        "a view walks the storage a block quantized tensor packs",
    );
    assert_eq!(
        graph.element(graph.matmul(graph.input(Shape::matrix(2, 300), Element::Single), weight,)),
        Element::Single,
    );
    assert!(
        refuses(|| {
            graph.scale(weight);
        }),
        "a block quantized tensor reconstructs through the quantum its storage holds",
    );
    assert!(
        refuses(|| {
            graph.scale(graph.permute(weight, [0, 1, 3, 2]));
        }),
        "a view of a block quantized tensor carries the storage it walks",
    );
}

#[test]
fn a_four_bit_float_parameter_carries_the_block_its_storage_packs() {
    let graph = Graph::new();
    let weight =
        graph.block_quantized_parameter(Shape::matrix(64, 4), Init::Zero, Element::Fp4E2M1);
    assert_eq!(graph.element(weight), Element::Fp4E2M1);
    assert_eq!(Element::Fp4E2M1.block(), 32);
    assert!(
        refuses(|| {
            graph.block_quantized_parameter(Shape::matrix(64, 4), Init::Zero, Element::Int8);
        }),
        "a parameter that declares a block quantized storage carries one",
    );
}

#[test]
fn a_block_quantized_tensor_is_a_weight_and_nothing_else() {
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    assert!(
        refuses(|| {
            graph.parameter(Shape::vector(4), Init::Zero, Element::Int4);
        }),
        "a block quantized weight packs the quantum of every block out of the numbers it holds",
    );
    assert!(
        refuses(|| {
            graph.input(Shape::vector(4), Element::Int4);
        }),
        "an input arrives quantized by no host",
    );
    assert!(
        refuses(|| {
            graph.resident(Shape::vector(4), Element::Int4);
        }),
        "a resident tensor holds the numbers a host wrote, and only a weight packs its blocks",
    );
    assert!(
        refuses(|| {
            graph.cast(data, Element::Int4);
        }),
        "a cast into int4 storage packs blocks the cast does not weigh",
    );
    let weight = graph.block_quantized_parameter(Shape::vector(4), Init::Zero, Element::Int4);
    assert!(
        refuses(|| {
            graph.relu(weight);
        }),
        "a pointwise op over a block quantized weight writes a storage whose blocks it does not weigh",
    );
    assert_eq!(
        graph.element(graph.cast(weight, Element::Single)),
        Element::Single,
        "a weight materializes into exact storage before the numbers it holds feed an op",
    );
}
