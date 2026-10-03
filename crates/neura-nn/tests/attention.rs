use neura_abi::Element;
use neura_graph::{AttentionOptions, Graph, Init, Residency, Shape, Value};
use neura_nn::{AdamW, HeadShape, MultiHeadAttention, mse_loss};
use neura_runtime::{Runtime, RuntimeRequest};

fn open() -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 20,
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
        "element {element} of a tensor of {elements} numbers: the plan gives {analytic} where the slope is {numeric}",
    );
}

struct Block {
    runtime: Runtime,
    graph: Graph<'static>,
    model: MultiHeadAttention<'static>,
    input: Value<'static>,
    target: Value<'static>,
    loss: Value<'static>,
    input_elements: u32,
    target_elements: u32,
}

const HEADS: u32 = 2;
const KEY_HEADS: u32 = 2;
const GROUPED_HEADS: u32 = 4;
const GROUPED_KEY_HEADS: u32 = 2;
const STREAM_HEADS: u32 = 1;
const BATCH: u32 = 2;
const TOKENS: u32 = 4;
const WIDTH: u32 = 3;
const WIDE_WIDTH: u32 = 64;

fn block(width: u32, heads: u32, key_heads: u32, input_heads: u32, causal: bool) -> Block {
    let graph = Graph::new();
    let model = MultiHeadAttention::new(
        &graph,
        HeadShape::new(heads, key_heads, width),
        Init::Uniform {
            low: -0.3,
            high: 0.3,
        },
        Element::Single,
        AttentionOptions {
            scale: 1.0 / (width as f32).sqrt(),
            causal,
            origin: None,
        },
    );
    let input = graph.input(
        Shape::of([input_heads, BATCH, TOKENS, width]),
        Element::Single,
    );
    let target = graph.input(Shape::of([heads, BATCH, TOKENS, width]), Element::Single);
    let loss = mse_loss(&graph, model.forward(&graph, input), target);
    graph.retain(loss);
    Block {
        runtime: open(),
        graph,
        model,
        input,
        target,
        loss,
        input_elements: input_heads * BATCH * TOKENS * width,
        target_elements: heads * BATCH * TOKENS * width,
    }
}

fn inputs(count: u32, seed: u32) -> Vec<f32> {
    let mut entropy = seed | 1;
    (0..count)
        .map(|_| {
            entropy ^= entropy << 13;
            entropy ^= entropy >> 17;
            entropy ^= entropy << 5;
            (entropy >> 8) as f32 / 16_777_216.0 - 0.5
        })
        .collect()
}

fn gradient_of_a_multi_head_attention_matches_finite_differences(
    width: u32,
    heads: u32,
    key_heads: u32,
    input_heads: u32,
    causal: bool,
) {
    let block = block(width, heads, key_heads, input_heads, causal);
    let gradients = block.graph.backward(block.loss);
    let parameters = block.model.parameters();
    for parameter in parameters {
        block.graph.retain(gradients.of(parameter));
    }
    let weights = block.runtime.weights(&block.graph);
    let program = block.runtime.compile(&block.graph, &weights);
    block
        .runtime
        .write(&program, block.input, &inputs(block.input_elements, 23));
    block
        .runtime
        .write(&program, block.target, &inputs(block.target_elements, 71));
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
    gradient_of_a_multi_head_attention_matches_finite_differences(
        WIDTH, HEADS, KEY_HEADS, HEADS, false,
    );
    gradient_of_a_multi_head_attention_matches_finite_differences(
        WIDTH, HEADS, KEY_HEADS, HEADS, true,
    );
    gradient_of_a_multi_head_attention_matches_finite_differences(
        WIDTH,
        GROUPED_HEADS,
        GROUPED_KEY_HEADS,
        STREAM_HEADS,
        true,
    );
    gradient_of_a_multi_head_attention_matches_finite_differences(
        WIDTH,
        GROUPED_HEADS,
        1,
        STREAM_HEADS,
        false,
    );
}

#[test]
fn a_wide_head_attention_matches_finite_differences() {
    gradient_of_a_multi_head_attention_matches_finite_differences(
        WIDE_WIDTH, HEADS, KEY_HEADS, HEADS, true,
    );
    gradient_of_a_multi_head_attention_matches_finite_differences(
        WIDE_WIDTH,
        GROUPED_HEADS,
        GROUPED_KEY_HEADS,
        STREAM_HEADS,
        false,
    );
}

#[test]
fn a_multi_head_attention_lowers_the_loss_it_was_shown() {
    let block = block(WIDTH, HEADS, KEY_HEADS, HEADS, true);
    let gradients = block.graph.backward(block.loss);
    let parameters = block.model.parameters();
    let mut optimizer = AdamW::new(&block.graph, 0.02, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&block.graph, &parameters);
    optimizer.step(&block.graph, &gradients);
    let weights = block.runtime.weights(&block.graph);
    let program = block.runtime.compile(&block.graph, &weights);
    block
        .runtime
        .write(&program, block.input, &inputs(block.input_elements, 23));
    block
        .runtime
        .write(&program, block.target, &inputs(block.target_elements, 71));
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

fn rotary_block() -> Block {
    let graph = Graph::new();
    let width = WIDTH + 1;
    let model = MultiHeadAttention::rotary(
        &graph,
        HeadShape::new(HEADS, KEY_HEADS, width),
        Init::Uniform {
            low: -0.3,
            high: 0.3,
        },
        Element::Single,
        AttentionOptions {
            scale: 1.0 / (width as f32).sqrt(),
            causal: true,
            origin: None,
        },
        10000.0,
    );
    let input = graph.input(Shape::of([HEADS, BATCH, TOKENS, width]), Element::Single);
    let target = graph.input(Shape::of([HEADS, BATCH, TOKENS, width]), Element::Single);
    let loss = mse_loss(&graph, model.forward(&graph, input), target);
    graph.retain(loss);
    Block {
        runtime: open(),
        graph,
        model,
        input,
        target,
        loss,
        input_elements: HEADS * BATCH * TOKENS * width,
        target_elements: HEADS * BATCH * TOKENS * width,
    }
}

#[test]
fn a_rotary_attention_gradient_matches_finite_differences() {
    let block = rotary_block();
    let gradients = block.graph.backward(block.loss);
    let parameters = block.model.parameters();
    for parameter in parameters {
        block.graph.retain(gradients.of(parameter));
    }
    let weights = block.runtime.weights(&block.graph);
    let program = block.runtime.compile(&block.graph, &weights);
    let count = HEADS * BATCH * TOKENS * (WIDTH + 1);
    block
        .runtime
        .write(&program, block.input, &inputs(count, 23));
    block
        .runtime
        .write(&program, block.target, &inputs(count, 71));
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
fn a_multi_head_attention_keeps_its_parameters_beside_its_tape() {
    let block = block(WIDTH, HEADS, KEY_HEADS, HEADS, false);
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
