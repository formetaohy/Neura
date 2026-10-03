use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_nn::{AdamW, Linear, mse_loss};
use neura_runtime::{Runtime, RuntimeRequest};

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

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{} numbers came back where {} were expected",
        actual.len(),
        expected.len(),
    );
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index} came back as {actual} where {expected} was expected",
        );
    }
}

const SAMPLES: u32 = 16;
const WIDTH: u32 = 4;
const HIDDEN: u32 = 8;
const OUTPUT: u32 = 2;
const STEPS: u32 = 60;

struct Model<'g> {
    observations: Value<'g>,
    targets: Value<'g>,
    prediction: Value<'g>,
    loss: Value<'g>,
    parameters: Vec<Value<'g>>,
}

fn network<'g>(graph: &Graph<'g>, hidden: u32) -> Model<'g> {
    let observations = graph.input(Shape::matrix(SAMPLES, WIDTH), Element::Single);
    let targets = graph.input(Shape::matrix(SAMPLES, OUTPUT), Element::Single);
    let spread = Init::Uniform {
        low: -0.3,
        high: 0.3,
    };
    let first = Linear::new(graph, WIDTH, hidden, spread, Element::Single);
    let second = Linear::new(graph, hidden, OUTPUT, spread, Element::Single);
    let prediction = second.forward(graph, graph.relu(first.forward(graph, observations)));
    graph.retain(prediction);
    let loss = mse_loss(graph, prediction, targets);
    let parameters = first
        .parameters()
        .iter()
        .chain(second.parameters().iter())
        .copied()
        .collect();
    Model {
        observations,
        targets,
        prediction,
        loss,
        parameters,
    }
}

fn descend<'g>(graph: &Graph<'g>, model: &Model<'g>) {
    let gradients = graph.backward(model.loss);
    let mut optimizer = AdamW::new(graph, 0.02, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(graph, &model.parameters);
    optimizer.step(graph, &gradients);
}

fn batch(seed: u32) -> (Vec<f32>, Vec<f32>) {
    let mut entropy = seed | 1;
    let mut next = move || {
        entropy = entropy.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (entropy >> 8) as f32 / 16_777_216.0 - 0.5
    };
    let observations = (0..SAMPLES * WIDTH).map(|_| next()).collect();
    let targets = (0..SAMPLES * OUTPUT).map(|_| next()).collect();
    (observations, targets)
}

fn host_forward(observations: &[f32], parameters: &[Vec<f32>], hidden: u32) -> Vec<f32> {
    let [weight, bias, out_weight, out_bias] = parameters else {
        panic!("a network of two layers carries four tensors");
    };
    let mut middle = vec![0.0f32; (SAMPLES * hidden) as usize];
    for row in 0..SAMPLES {
        for column in 0..hidden {
            let mut total = bias[column as usize];
            for depth in 0..WIDTH {
                total += observations[(row * WIDTH + depth) as usize]
                    * weight[(depth * hidden + column) as usize];
            }
            middle[(row * hidden + column) as usize] = total.max(0.0);
        }
    }
    let mut prediction = vec![0.0f32; (SAMPLES * OUTPUT) as usize];
    for row in 0..SAMPLES {
        for column in 0..OUTPUT {
            let mut total = out_bias[column as usize];
            for depth in 0..hidden {
                total += middle[(row * hidden + depth) as usize]
                    * out_weight[(depth * OUTPUT + column) as usize];
            }
            prediction[(row * OUTPUT + column) as usize] = total;
        }
    }
    prediction
}

#[test]
fn a_training_graph_keeps_its_state_beside_the_model_it_learns() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, HIDDEN);
    descend(&graph, &model);
    let weights = runtime.weights(&graph);
    assert_eq!(
        weights.tensors(),
        4,
        "the model carries four tensors while the moments of the descent ride beside them",
    );
    let checkpoint = runtime.checkpoint(&weights);
    assert_eq!(checkpoint.tensors(), 4);
    assert_eq!(
        checkpoint.state_tensors(),
        9,
        "four pairs of moments beside the clock the descent counts its steps with",
    );
}

#[test]
fn a_deployment_graph_serves_the_store_a_training_graph_learned() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, HIDDEN);
    descend(&graph, &model);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (observations, targets) = batch(7);
    runtime.write(&program, model.observations, &observations);
    runtime.write(&program, model.targets, &targets);
    for _ in 0..STEPS {
        runtime.run(&program);
    }
    let learned = runtime.read_many(&program, &model.parameters);
    let expected = host_forward(&observations, &learned, HIDDEN);

    let deployment = Graph::new();
    let deployed = network(&deployment, HIDDEN);
    let serving = runtime.compile(&deployment, &weights);
    assert!(
        !serving.updates_weights(),
        "a deployment plan carries no step of a descent",
    );
    assert!(
        serving.task_count() < program.task_count(),
        "a deployment plan carries {} tasks where the plan that trained it carries {}",
        serving.task_count(),
        program.task_count(),
    );
    runtime.write(&serving, deployed.observations, &observations);
    runtime.run(&serving);
    assert_close(
        &runtime.read(&serving, deployed.prediction),
        &expected,
        1e-5,
    );
}

#[test]
fn a_checkpoint_of_training_state_pours_into_a_deployment_graph() {
    let runtime = open();
    let graph = Graph::new();
    let model = network(&graph, HIDDEN);
    descend(&graph, &model);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let (observations, targets) = batch(11);
    runtime.write(&program, model.observations, &observations);
    runtime.write(&program, model.targets, &targets);
    for _ in 0..STEPS {
        runtime.run(&program);
    }
    let learned = runtime.read_many(&program, &model.parameters);
    let expected = host_forward(&observations, &learned, HIDDEN);
    let checkpoint = runtime.checkpoint(&weights);

    let another = open();
    let deployment = Graph::new();
    let deployed = network(&deployment, HIDDEN);
    let weights = another.load(&deployment, &checkpoint);
    let serving = another.compile(&deployment, &weights);
    another.write(&serving, deployed.observations, &observations);
    another.run(&serving);
    assert_close(
        &another.read(&serving, deployed.prediction),
        &expected,
        1e-5,
    );
}

#[test]
fn a_training_graph_refuses_a_store_that_carries_no_state() {
    let runtime = open();
    let deployment = Graph::new();
    let _ = network(&deployment, HIDDEN);
    let weights = runtime.weights(&deployment);
    let graph = Graph::new();
    let model = network(&graph, HIDDEN);
    descend(&graph, &model);
    assert!(
        refuses(|| {
            let _ = runtime.compile(&graph, &weights);
        }),
        "a descent resumes only from the state it wrote",
    );
}
