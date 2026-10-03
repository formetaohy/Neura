use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::Checkpoint;

#[path = "support/mod.rs"]
mod support;

use support::{assert_close, open};

struct Model<'g> {
    observations: Value<'g>,
    prediction: Value<'g>,
}

fn spread() -> Init {
    Init::Uniform {
        low: -0.25,
        high: 0.25,
    }
}

fn network<'g>(graph: &Graph<'g>, hidden: u32, element: Element) -> Model<'g> {
    let observations = graph.input(Shape::matrix(8, 4), Element::Single);
    let first = graph.named_parameter("first", Shape::matrix(4, hidden), spread(), element);
    let second = graph.named_parameter("second", Shape::matrix(hidden, 2), spread(), element);
    let prediction = graph.matmul(graph.relu(graph.matmul(observations, first)), second);
    graph.retain(prediction);
    Model {
        observations,
        prediction,
    }
}

fn batch(seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    let mut next = move || {
        entropy = entropy.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (entropy >> 8) as f32 / 16_777_216.0 - 0.5
    };
    (0..8 * 4).map(|_| next()).collect()
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_checkpoint_outlives_its_runtime() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, 5, Element::Single);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations = batch(3);
    runtime.write(&program, model.observations, &observations);
    for _ in 0..40 {
        runtime.run(&program);
    }
    let prediction = runtime.read(&program, model.prediction);
    let checkpoint = runtime.checkpoint(&weights);

    let another = open();
    let rebuilt = Graph::new();
    let model = network(&rebuilt, 5, Element::Single);
    let weights = another.load(&rebuilt, &checkpoint);
    let program = another.compile(&rebuilt, &weights);
    another.write(&program, model.observations, &observations);
    another.run(&program);
    assert_close(&another.read(&program, model.prediction), &prediction, 1e-6);
}

#[test]
fn a_store_restores_its_trained_parameters() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, 5, Element::Single);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations = batch(7);
    runtime.write(&program, model.observations, &observations);
    runtime.run(&program);
    let fresh = runtime.read(&program, model.prediction);
    let initial = runtime.checkpoint(&weights);
    for _ in 0..40 {
        runtime.run(&program);
    }
    let prediction = runtime.read(&program, model.prediction);
    let checkpoint = runtime.checkpoint(&weights);

    runtime.restore(&weights, &initial);
    runtime.run(&program);
    assert_close(&runtime.read(&program, model.prediction), &fresh, 1e-6);
    runtime.restore(&weights, &checkpoint);
    runtime.run(&program);
    assert_close(&runtime.read(&program, model.prediction), &prediction, 1e-6);
}

#[test]
fn a_store_names_every_tensor_it_carries() {
    let runtime = open();
    let graph = Graph::new();
    let _model = network(&graph, 5, Element::Single);
    let checkpoint = runtime.checkpoint(&runtime.weights(&graph));
    assert_eq!(checkpoint.tensors(), 2);
    assert_eq!(
        checkpoint.names().collect::<Vec<&str>>(),
        vec!["first", "second"],
    );
    let first = checkpoint.tensor("first").expect("a named tensor");
    assert_eq!(first.element, Some(Element::Single));
    assert_eq!(first.elements, 4 * 5);
    assert_eq!(first.payload.len(), 4 * 5 * 4);
    assert_eq!(first.quanta, None);
    assert!(checkpoint.tensor("third").is_none());
}

#[test]
fn a_store_loads_only_the_tensors_its_graph_names() {
    let runtime = open();
    let graph = Graph::new();
    let _model = network(&graph, 5, Element::Single);
    let checkpoint = runtime.checkpoint(&runtime.weights(&graph));

    let renamed = Graph::new();
    let _renamed = named_network(&renamed, 5, Element::Single, "other");
    assert!(
        refuses(|| {
            runtime.load(&renamed, &checkpoint);
        }),
        "a store loads only a graph that names the very tensors the store holds",
    );

    let wider = Graph::new();
    let _wider = named_network(&wider, 6, Element::Single, "first");
    assert!(
        refuses(|| {
            runtime.load(&wider, &checkpoint);
        }),
        "a store loads only a graph whose tensors hold the very number of numbers the store holds",
    );

    let other = Graph::new();
    let _other = named_network(&other, 5, Element::Half, "first");
    assert!(
        refuses(|| {
            runtime.load(&other, &checkpoint);
        }),
        "a store loads only a graph of the elements the store holds",
    );
}

fn named_network<'g>(graph: &Graph<'g>, hidden: u32, element: Element, first: &str) -> Model<'g> {
    let observations = graph.input(Shape::matrix(8, 4), Element::Single);
    let head = graph.named_parameter(first, Shape::matrix(4, hidden), spread(), element);
    let tail = graph.named_parameter("second", Shape::matrix(hidden, 2), spread(), element);
    let prediction = graph.matmul(graph.relu(graph.matmul(observations, head)), tail);
    graph.retain(prediction);
    Model {
        observations,
        prediction,
    }
}

#[test]
fn an_unnamed_parameter_refuses_a_checkpoint() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(4, 4), spread(), Element::Single);
    graph.retain(weight);
    let weights = runtime.weights(&graph);
    assert!(
        refuses(|| {
            runtime.checkpoint(&weights);
        }),
        "a checkpoint names every tensor it holds, and an unnamed parameter carries no name",
    );
}

#[test]
fn a_half_store_round_trips_bit_for_bit() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, 5, Element::Half);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, model.observations, &batch(11));
    runtime.run(&program);
    let before = runtime.read(&program, model.prediction);
    let checkpoint = runtime.checkpoint(&weights);
    let first = checkpoint.tensor("first").expect("a named tensor");
    assert_eq!(first.element, Some(Element::Half));
    assert_eq!(first.elements, 4 * 5);
    assert_eq!(first.payload.len(), 4 * 5 * 2);

    let reloaded = runtime.load(&graph, &checkpoint);
    runtime.restore(&reloaded, &checkpoint);
    let program = runtime.compile(&graph, &reloaded);
    runtime.write(&program, model.observations, &batch(11));
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, model.prediction),
        before,
        "a half store pours back the very words it was saved from",
    );
}

#[test]
fn a_quantized_store_round_trips_bit_for_bit() {
    let runtime = open();
    let scale = 0.03125;
    let graph = Graph::new();
    let weight = graph.named_quantized_parameter("weight", Shape::matrix(4, 4), spread(), scale);
    let observations = graph.input(Shape::matrix(2, 4), Element::Single);
    let prediction = graph.matmul(observations, weight);
    graph.retain(prediction);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, observations, &batch(7)[..8]);
    runtime.run(&program);
    let before = runtime.read(&program, prediction);
    let checkpoint = runtime.checkpoint(&weights);
    let tensor = checkpoint.tensor("weight").expect("a named tensor");
    assert_eq!(tensor.element, Some(Element::Int8));
    assert_eq!(tensor.elements, 16);
    assert_eq!(tensor.scale, scale);
    assert_eq!(tensor.payload.len(), 16);

    let another = open();
    let rebuilt = Graph::new();
    let weight = rebuilt.named_quantized_parameter("weight", Shape::matrix(4, 4), spread(), scale);
    let observations = rebuilt.input(Shape::matrix(2, 4), Element::Single);
    let prediction = rebuilt.matmul(observations, weight);
    rebuilt.retain(prediction);
    let weights = another.load(&rebuilt, &checkpoint);
    let program = another.compile(&rebuilt, &weights);
    another.write(&program, observations, &batch(7)[..8]);
    another.run(&program);
    assert_eq!(
        another.read(&program, prediction),
        before,
        "a quantized store pours back the very words it was saved from",
    );

    let rescaled = Graph::new();
    rescaled.named_quantized_parameter("weight", Shape::matrix(4, 4), Init::Zero, scale / 2.0);
    assert!(
        refuses(|| {
            another.load(&rescaled, &checkpoint);
        }),
        "a store loads only a graph of the quantum it was saved by",
    );
}

#[test]
fn a_block_quantized_store_round_trips_bit_for_bit() {
    for storage in [Element::Int4, Element::Fp4E2M1] {
        let runtime = open();
        let graph = Graph::new();
        let weight =
            graph.named_block_quantized_parameter("weight", Shape::matrix(4, 8), spread(), storage);
        let observations = graph.input(Shape::matrix(2, 4), Element::Single);
        let prediction = graph.matmul(observations, weight);
        graph.retain(prediction);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.write(&program, observations, &batch(13)[..8]);
        runtime.run(&program);
        let before = runtime.read(&program, prediction);
        let checkpoint = runtime.checkpoint(&weights);
        let tensor = checkpoint.tensor("weight").expect("a named tensor");
        assert_eq!(tensor.element, Some(storage));
        assert_eq!(tensor.elements, 32);
        assert_eq!(tensor.payload.len(), 32 / 2);
        assert_eq!(
            tensor.quanta.expect("a block quantized table").len(),
            storage.quanta(32) as usize * 4,
        );

        let another = open();
        let rebuilt = Graph::new();
        let weight = rebuilt.named_block_quantized_parameter(
            "weight",
            Shape::matrix(4, 8),
            spread(),
            storage,
        );
        let observations = rebuilt.input(Shape::matrix(2, 4), Element::Single);
        let prediction = rebuilt.matmul(observations, weight);
        rebuilt.retain(prediction);
        let weights = another.load(&rebuilt, &checkpoint);
        let program = another.compile(&rebuilt, &weights);
        another.write(&program, observations, &batch(13)[..8]);
        another.run(&program);
        assert_eq!(
            another.read(&program, prediction),
            before,
            "a {} store pours back the very words it was saved from",
            storage.name(),
        );
    }
}

#[test]
fn a_checkpoint_carries_every_element_of_its_store() {
    let runtime = open();
    let graph = Graph::new();
    let weight = graph.named_parameter("weight", Shape::matrix(4, 4), spread(), Element::Bfloat16);
    let bias = graph.named_parameter("bias", Shape::vector(4), Init::Zero, Element::Single);
    let observations = graph.input(Shape::matrix(2, 4), Element::Single);
    let prediction = graph.add(graph.matmul(observations, weight), bias);
    graph.retain(prediction);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, observations, &batch(5)[..8]);
    runtime.run(&program);
    let before = runtime.read(&program, prediction);
    let checkpoint = runtime.checkpoint(&weights);
    assert_eq!(checkpoint.tensors(), 2);
    assert_eq!(
        checkpoint.tensor("weight").expect("a named tensor").element,
        Some(Element::Bfloat16),
    );
    assert_eq!(
        checkpoint.tensor("bias").expect("a named tensor").element,
        Some(Element::Single),
    );

    let reloaded = runtime.load(&graph, &checkpoint);
    let program = runtime.compile(&graph, &reloaded);
    runtime.write(&program, observations, &batch(5)[..8]);
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, prediction),
        before,
        "a store pours every tensor back in the element it was saved with",
    );
}

#[test]
fn a_truncated_checkpoint_refuses_to_decode() {
    let runtime = open();
    let graph = Graph::new();
    let _model = network(&graph, 5, Element::Single);
    let checkpoint = runtime.checkpoint(&runtime.weights(&graph));
    let bytes = checkpoint.bytes();
    assert!(
        refuses(|| {
            Checkpoint::decode(&bytes[..bytes.len() - 4]);
        }),
        "a checkpoint decodes only bytes its header accounts for",
    );
    assert!(
        refuses(|| {
            Checkpoint::decode(&[]);
        }),
        "a checkpoint decodes only bytes its header opens with",
    );
}
