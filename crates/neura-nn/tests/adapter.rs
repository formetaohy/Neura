use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_nn::{AdamW, Adapter, Linear, mse_loss};
use neura_runtime::{Program, Runtime, RuntimeRequest};

fn open() -> Runtime {
    pollster::block_on(Runtime::open(RuntimeRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn plane(samples: u32) -> (Vec<f32>, Vec<f32>) {
    let mut inputs = Vec::with_capacity(samples as usize * 2);
    let mut targets = Vec::with_capacity(samples as usize);
    for sample in 0..samples {
        let first = (sample as f32 / samples as f32) * 2.0 - 1.0;
        let second = ((sample * 7) % samples) as f32 / samples as f32 * 2.0 - 1.0;
        inputs.push(first);
        inputs.push(second);
        targets.push(3.0 * first - 2.0 * second);
    }
    (inputs, targets)
}

fn step<'r>(runtime: &'r Runtime, frozen: bool) -> Program<'r> {
    let graph = Graph::new();
    let base = Linear::new(&graph, 8, 8, Init::Constant(0.5), Element::Single);
    if frozen {
        graph.freeze(&base.parameters());
    }
    let adapter = Adapter::new(
        &graph,
        8,
        8,
        2,
        Init::Uniform {
            low: -0.2,
            high: 0.2,
        },
        Element::Single,
        0.5,
    );
    let inputs = graph.input(Shape::matrix(16, 8), Element::Single);
    let prediction = adapter.forward(&graph, inputs, base.forward(&graph, inputs));
    let loss = mse_loss(&graph, prediction, inputs);
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, 0.01, 0.9, 0.999, 1e-8, 0.0);
    if !frozen {
        optimizer.track_all(&graph, &base.parameters());
    }
    optimizer.track_all(&graph, &adapter.parameters());
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    runtime.compile(&graph, &weights)
}

#[test]
fn a_frozen_base_shortens_the_tape_that_carries_it() {
    let runtime = open();
    let frozen = {
        let program = step(&runtime, true);
        (program.task_count(), program.work())
    };
    let learning = {
        let program = step(&runtime, false);
        (program.task_count(), program.work())
    };
    assert!(
        frozen.0 < learning.0,
        "a frozen base keeps {} tasks where the same base learning asks for {}",
        frozen.0,
        learning.0,
    );
    assert!(
        frozen.1 < learning.1,
        "a frozen base asks the device for {} element ops where the same base learning asks for {}",
        frozen.1,
        learning.1,
    );
}

#[test]
fn an_adapter_begins_as_a_bypass_of_zero() {
    let runtime = open();
    let graph = Graph::new();
    let adapter = Adapter::new(
        &graph,
        2,
        1,
        2,
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Single,
        0.5,
    );
    let inputs = graph.input(Shape::matrix(64, 2), Element::Single);
    let prediction = adapter.forward(&graph, inputs, graph.fill(Shape::matrix(64, 1), 0.0));
    graph.retain(prediction);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (inputs_data, _) = plane(64);
    runtime.write(&program, inputs, &inputs_data);
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, prediction),
        vec![0.0; 64],
        "an adapter whose up projection is zero adds nothing to the base it rides",
    );
}

#[test]
fn a_frozen_base_trains_only_the_bypass_it_carries() {
    let runtime = open();
    let graph = Graph::new();
    let base = Linear::new(&graph, 2, 1, Init::Constant(1.0), Element::Single);
    graph.freeze(&base.parameters());
    let adapter = Adapter::new(
        &graph,
        2,
        1,
        2,
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Single,
        0.5,
    );
    let inputs = graph.input(Shape::matrix(64, 2), Element::Single);
    let targets = graph.input(Shape::matrix(64, 1), Element::Single);
    let prediction = adapter.forward(&graph, inputs, base.forward(&graph, inputs));
    let loss = mse_loss(&graph, prediction, targets);
    let gradients = graph.backward(loss);
    assert!(graph.trains(adapter.down()) && graph.trains(adapter.up()));
    assert!(!graph.trains(base.weight()) && !graph.trains(base.bias()));
    assert!(refuses(|| {
        let _ = gradients.of(base.weight());
    }));
    let mut optimizer = AdamW::new(&graph, 0.02, 0.9, 0.999, 1e-8, 0.0);
    assert!(refuses(|| {
        let mut refused = AdamW::new(&graph, 0.02, 0.9, 0.999, 1e-8, 0.0);
        refused.track(&graph, base.weight());
    }));
    optimizer.track_all(&graph, &adapter.parameters());
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (inputs_data, targets_data) = plane(64);
    runtime.write(&program, inputs, &inputs_data);
    runtime.write(&program, targets, &targets_data);
    let frozen = runtime.read(&program, base.weight());
    assert_eq!(
        frozen,
        vec![1.0; 2],
        "a frozen base still draws the numbers its seed declares",
    );
    let bypass = runtime.read(&program, adapter.down());
    let mut start = None;
    let mut end = 0.0;
    for step in 0..400 {
        runtime.run(&program);
        if step % 40 == 0 || step == 399 {
            end = runtime.read(&program, loss)[0];
            start.get_or_insert(end);
        }
    }
    let start = start.expect("a first loss");
    assert!(
        end < start * 0.5,
        "the loss fell from {start} to {end} while the base stood still",
    );
    assert_eq!(
        runtime.read(&program, base.weight()),
        frozen,
        "a frozen base hands every run the very same weight",
    );
    assert_eq!(
        runtime.read(&program, base.bias()),
        vec![0.0; 1],
        "a frozen bias stays where its seed left it",
    );
    assert_ne!(
        runtime.read(&program, adapter.down()),
        bypass,
        "the bypass a frozen base carries is the tensor that learns",
    );
}

#[test]
fn a_four_bit_base_trains_the_bypass_it_carries() {
    let runtime = open();
    let graph = Graph::new();
    let base = Linear::block_quantized(
        &graph,
        2,
        1,
        Init::Constant(1.0),
        Element::Single,
        Element::Int4,
    );
    let adapter = Adapter::new(
        &graph,
        2,
        1,
        2,
        Init::Uniform {
            low: -0.4,
            high: 0.4,
        },
        Element::Single,
        0.5,
    );
    let inputs = graph.input(Shape::matrix(64, 2), Element::Single);
    let targets = graph.input(Shape::matrix(64, 1), Element::Single);
    let prediction = adapter.forward(&graph, inputs, base.forward(&graph, inputs));
    let loss = mse_loss(&graph, prediction, targets);
    let gradients = graph.backward(loss);
    assert!(graph.trains(adapter.down()) && graph.trains(adapter.up()));
    assert!(
        !graph.trains(base.weight()),
        "a four bit weight packs the quantum of its blocks out of numbers no gradient reaches",
    );
    assert!(
        graph.trains(base.bias()),
        "the bias of a four bit base stays a parameter of its own",
    );
    assert!(refuses(|| {
        let _ = gradients.of(base.weight());
    }));
    let mut optimizer = AdamW::new(&graph, 0.02, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&graph, &adapter.parameters());
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (inputs_data, targets_data) = plane(64);
    runtime.write(&program, inputs, &inputs_data);
    runtime.write(&program, targets, &targets_data);
    assert_eq!(
        runtime.read(&program, base.weight()),
        vec![1.0; 2],
        "a four bit block of ones reconstructs the number its quantum places",
    );
    let base_weight = runtime.read(&program, base.weight());
    let bypass = runtime.read(&program, adapter.down());
    let mut start = None;
    let mut end = 0.0;
    for step in 0..400 {
        runtime.run(&program);
        if step % 40 == 0 || step == 399 {
            end = runtime.read(&program, loss)[0];
            start.get_or_insert(end);
        }
    }
    let start = start.expect("a first loss");
    assert!(
        end < start * 0.5,
        "the loss fell from {start} to {end} while the four bit base stood still",
    );
    assert_eq!(
        runtime.read(&program, base.weight()),
        base_weight,
        "a four bit base hands every run the very same weight",
    );
    assert_eq!(
        runtime.read(&program, base.bias()),
        vec![0.0; 1],
        "a base bias the optimizer does not track stays where its seed left it",
    );
    assert_ne!(
        runtime.read(&program, adapter.down()),
        bypass,
        "the bypass a four bit base carries is the tensor that learns",
    );
}
