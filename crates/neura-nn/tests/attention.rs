use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Residency, Shape, Value};
use neura_nn::{Adam, MultiHeadAttention, mse_loss};
use neura_runtime::{Runtime, RuntimeRequest};

fn open() -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn sampled(elements: usize) -> impl Iterator<Item = usize> {
    let stride = (elements / 4).max(1);
    (0..elements).step_by(stride).take(4)
}

fn assert_slope(element: usize, analytic: f32, numeric: f32, elements: usize) {
    let slack = 1e-3 + 1e-2 * analytic.abs().max(numeric.abs());
    assert!(
        (numeric - analytic).abs() <= slack,
        "element {element} of a tensor of {elements} numbers: the tape gives {analytic} where the slope is {numeric}",
    );
}

struct Block {
    runtime: Runtime,
    graph: Graph<'static>,
    model: MultiHeadAttention<'static>,
    input: Value<'static>,
    target: Value<'static>,
    loss: Value<'static>,
}

const HEADS: u32 = 2;
const BATCH: u32 = 2;
const TOKENS: u32 = 4;
const WIDTH: u32 = 3;

fn block(causal: bool) -> Block {
    let graph = Graph::new();
    let model = MultiHeadAttention::new(
        &graph,
        HEADS,
        WIDTH,
        Init::Uniform {
            low: -0.3,
            high: 0.3,
        },
        Element::Single,
        AttentionOptions {
            scale: 1.0 / (WIDTH as f32).sqrt(),
            causal,
        },
    );
    let input = graph.input(Shape::of([HEADS, BATCH, TOKENS, WIDTH]), Element::Single);
    let target = graph.input(Shape::of([HEADS, BATCH, TOKENS, WIDTH]), Element::Single);
    let loss = mse_loss(&graph, model.forward(&graph, input), target);
    graph.retain(loss);
    Block {
        runtime: open(),
        graph,
        model,
        input,
        target,
        loss,
    }
}

fn elements() -> u32 {
    HEADS * BATCH * TOKENS * WIDTH
}

fn inputs(seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..elements())
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn gradient_of_a_multi_head_attention_matches_finite_differences(causal: bool) {
    let block = block(causal);
    let gradients = block.graph.backward(block.loss);
    let parameters = block.model.parameters();
    for parameter in parameters {
        block.graph.retain(gradients.of(parameter));
    }
    let weights = block.runtime.weights(&block.graph);
    let program = block.runtime.compile(&block.graph, &weights);
    block.runtime.write(&program, block.input, &inputs(23));
    block.runtime.write(&program, block.target, &inputs(71));
    block.runtime.run(&program);
    for parameter in parameters {
        let values = block.runtime.read(&program, parameter);
        let analytic = block.runtime.read(&program, gradients.of(parameter));
        for element in sampled(values.len()) {
            let step = 0.01 * values[element].abs().max(0.1);
            let mut probe = values.clone();
            probe[element] += step;
            block.runtime.write(&program, parameter, &probe);
            block.runtime.run(&program);
            let high = block.runtime.read(&program, block.loss)[0];
            probe[element] -= 2.0 * step;
            block.runtime.write(&program, parameter, &probe);
            block.runtime.run(&program);
            let low = block.runtime.read(&program, block.loss)[0];
            assert_slope(
                element,
                analytic[element],
                (high - low) / (2.0 * step),
                values.len(),
            );
        }
        block.runtime.write(&program, parameter, &values);
    }
}

#[test]
fn an_attention_gradient_matches_finite_differences() {
    gradient_of_a_multi_head_attention_matches_finite_differences(false);
    gradient_of_a_multi_head_attention_matches_finite_differences(true);
}

#[test]
fn a_multi_head_attention_lowers_the_loss_it_was_shown() {
    let block = block(true);
    let gradients = block.graph.backward(block.loss);
    let parameters = block.model.parameters();
    let mut optimizer = Adam::new(&block.graph, 0.02, 0.9, 0.999, 1e-8);
    optimizer.track_all(&block.graph, &parameters);
    optimizer.step(&block.graph, &gradients);
    let weights = block.runtime.weights(&block.graph);
    let program = block.runtime.compile(&block.graph, &weights);
    block.runtime.write(&program, block.input, &inputs(23));
    block.runtime.write(&program, block.target, &inputs(71));
    let mut first = None;
    let mut last = 0.0;
    for step in 0..400 {
        block.runtime.run(&program);
        if step % 100 == 0 || step == 399 {
            last = block.runtime.read(&program, block.loss)[0];
            first.get_or_insert(last);
        }
    }
    let first = first.expect("a run reads a loss");
    assert!(
        last < 0.5 * first,
        "a step of {first} became {last} over 400 steps",
    );
}

#[test]
fn a_multi_head_attention_keeps_its_parameters_beside_its_tape() {
    let block = block(false);
    let gradients = block.graph.backward(block.loss);
    for parameter in block.model.parameters() {
        block.graph.retain(gradients.of(parameter));
    }
    let weights = block.runtime.weights(&block.graph);
    let program = block.runtime.compile(&block.graph, &weights);
    assert_eq!(
        program.weights().tensors(),
        8,
        "an attention block carries four projections and their shifts",
    );
    assert!(
        program.tensor_bytes() < 1 << 16,
        "a fused attention block holds {} bytes of tensors",
        program.tensor_bytes(),
    );
    assert!(
        !block
            .graph
            .snapshot()
            .values()
            .iter()
            .any(|value| value.shape.dims()[2] == TOKENS * TOKENS
                && value.residency == Residency::Derived),
        "a fused attention materializes no score tensor",
    );
}
