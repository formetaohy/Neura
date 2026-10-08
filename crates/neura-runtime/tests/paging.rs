use neura_abi::Element;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{Backends, MemoryRequest, Runtime, RuntimeRequest};

const ROWS: u32 = 16;
const WIDTH: u32 = 128;
const STEPS: u32 = 3;
const RATE: f32 = 0.05;

struct Model<'g> {
    first_weight: Value<'g>,
    first_bias: Value<'g>,
    second_weight: Value<'g>,
    second_bias: Value<'g>,
}

impl<'g> Model<'g> {
    fn new(graph: &Graph<'g>) -> Self {
        let init = Init::Uniform {
            low: -0.1,
            high: 0.1,
        };
        Self {
            first_weight: graph.parameter(Shape::matrix(WIDTH, WIDTH), init, Element::Single),
            first_bias: graph.parameter(Shape::matrix(1, WIDTH), Init::Zero, Element::Single),
            second_weight: graph.parameter(Shape::matrix(WIDTH, WIDTH), init, Element::Single),
            second_bias: graph.parameter(Shape::matrix(1, WIDTH), Init::Zero, Element::Single),
        }
    }

    fn forward(&self, graph: &Graph<'g>, data: Value<'g>) -> Value<'g> {
        let hidden = graph.relu(graph.add(graph.matmul(data, self.first_weight), self.first_bias));
        graph.add(graph.matmul(hidden, self.second_weight), self.second_bias)
    }

    fn parameters(&self) -> [Value<'g>; 4] {
        [
            self.first_weight,
            self.first_bias,
            self.second_weight,
            self.second_bias,
        ]
    }
}

fn open(backends: Backends, memory: MemoryRequest) -> Runtime {
    Runtime::open(RuntimeRequest {
        gpu: neura_runtime::GpuRequest {
            backends,
            ..Default::default()
        },
        memory,
    })
    .unwrap_or_else(|error| panic!("no device runs the tests over {backends:?}: {error}"))
}

const STREAMED_BYTES: u64 = 5 * (1 << 14);

fn train(backends: Backends, memory: MemoryRequest) -> (Vec<Vec<f32>>, Vec<f32>) {
    let runtime = open(backends, memory);
    let graph = Graph::new();
    let model = Model::new(&graph);
    let data = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    let target = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    let difference = graph.sub(model.forward(&graph, data), target);
    let loss = graph.sum(graph.mul(difference, difference));
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let descent = graph.fill(Shape::scalar(), -RATE);
    for parameter in model.parameters() {
        graph.add_into(parameter, graph.mul(gradients.of(parameter), descent));
    }

    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations = (0..(ROWS * WIDTH))
        .map(|index| ((index * 37) % 101) as f32 / 101.0 - 0.5)
        .collect::<Vec<_>>();
    let targets = (0..(ROWS * WIDTH))
        .map(|index| ((index * 53) % 97) as f32 / 97.0 - 0.5)
        .collect::<Vec<_>>();
    let mut observed = 0.0;
    for _ in 0..STEPS {
        runtime.write(&program, data, &observations);
        runtime.write(&program, target, &targets);
        runtime.run(&program);
        observed = runtime.read(&program, loss)[0];
    }
    let parameters = model
        .parameters()
        .iter()
        .map(|parameter| runtime.read(&program, *parameter))
        .collect();
    (parameters, vec![observed])
}

fn train_streamed_like_the_resident_store(backends: Backends) {
    let resident = train(backends, MemoryRequest::default());
    let streamed = train(
        backends,
        MemoryRequest {
            resident_weight_bytes: Some(STREAMED_BYTES),
            ..Default::default()
        },
    );
    assert_eq!(
        resident.0.len(),
        streamed.0.len(),
        "a streamed store carries the parameters of its model",
    );
    for (expected, observed) in resident.0.iter().zip(&streamed.0) {
        assert_eq!(expected.len(), observed.len());
        for (at, (expected, observed)) in expected.iter().zip(observed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a streamed store reached {observed} where the resident store reached {expected} at {at}",
            );
        }
    }
    assert!(
        (resident.1[0] - streamed.1[0]).abs() <= 1e-5,
        "a streamed store held a loss of {} where the resident store held {}",
        streamed.1[0],
        resident.1[0],
    );
}

fn resident_words_stay_within_the_budget(backends: Backends) {
    let runtime = open(
        backends,
        MemoryRequest {
            resident_weight_bytes: Some(STREAMED_BYTES),
            ..Default::default()
        },
    );
    let graph = Graph::new();
    let model = Model::new(&graph);
    let data = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    let target = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    let difference = graph.sub(model.forward(&graph, data), target);
    let loss = graph.sum(graph.mul(difference, difference));
    graph.retain(loss);
    let weights = runtime.weights(&graph);
    assert!(
        weights.pages() > weights.resident_pages(),
        "a store of {} pages keeps {} of them beside a budget of {STREAMED_BYTES} bytes",
        weights.pages(),
        weights.resident_pages(),
    );
    assert!(
        weights.bytes() > STREAMED_BYTES,
        "a store of {} bytes fits the budget of {STREAMED_BYTES} bytes and never streams",
        weights.bytes(),
    );
    assert!(
        weights.resident_bytes() <= STREAMED_BYTES,
        "the device holds {} resident weight bytes beside a budget of {STREAMED_BYTES}",
        weights.resident_bytes(),
    );
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &vec![0.25; (ROWS * WIDTH) as usize]);
    runtime.write(&program, target, &vec![0.5; (ROWS * WIDTH) as usize]);
    runtime.run(&program);
    let observed = runtime.read(&program, loss)[0];
    assert!(
        observed.is_finite(),
        "a streamed store ran a step whose loss is {observed}",
    );
}

fn a_budget_below_one_task_is_refused(backends: Backends) {
    let runtime = open(
        backends,
        MemoryRequest {
            resident_weight_bytes: Some(1 << 14),
            ..Default::default()
        },
    );
    let graph = Graph::new();
    let model = Model::new(&graph);
    let data = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    let target = graph.input(Shape::matrix(ROWS, WIDTH), Element::Single);
    let difference = graph.sub(model.forward(&graph, data), target);
    let loss = graph.sum(graph.mul(difference, difference));
    graph.retain(loss);
    let weights = runtime.weights(&graph);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        runtime.compile(&graph, &weights);
    }));
    assert!(
        refused.is_err(),
        "a store of {} pages beside a task that walks more of them was accepted",
        weights.resident_pages(),
    );
}

fn host_writes_come_back(backends: Backends) {
    let width = 64;
    let runtime = open(
        backends,
        MemoryRequest {
            resident_weight_bytes: Some(1 << 14),
            ..Default::default()
        },
    );
    let graph = Graph::new();
    let weight = graph.parameter(Shape::matrix(width, width), Init::Zero, Element::Single);
    let other = graph.parameter(
        Shape::matrix(width, width),
        Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    let data = graph.input(Shape::matrix(width, width), Element::Single);
    graph.retain(graph.add(graph.matmul(data, weight), other));
    let weights = runtime.weights(&graph);
    assert_eq!(weights.pages(), 2);
    assert_eq!(weights.resident_pages(), 1);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &vec![0.125; (width * width) as usize]);
    runtime.run(&program);
    let written = (0..(width * width))
        .map(|index| (index as f32 * 0.001) - 0.5)
        .collect::<Vec<_>>();
    runtime.write(&program, weight, &written);
    runtime.run(&program);
    assert_eq!(
        runtime.read(&program, weight),
        written,
        "a streamed store gives back the weights the host wrote",
    );
}

fn checkpoints_carry_every_word(backends: Backends) {
    let width = 64;
    let runtime = open(
        backends,
        MemoryRequest {
            resident_weight_bytes: Some(1 << 14),
            ..Default::default()
        },
    );
    let graph = Graph::new();
    let weight = graph.named_parameter(
        "weight",
        Shape::matrix(width, width),
        Init::Uniform {
            low: -1.0,
            high: 1.0,
        },
        Element::Single,
    );
    let other = graph.named_parameter(
        "other",
        Shape::matrix(width, width),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let data = graph.input(Shape::matrix(width, width), Element::Single);
    graph.retain(graph.add(graph.matmul(data, weight), other));
    let weights = runtime.weights(&graph);
    assert_eq!(weights.pages(), 2);
    assert_eq!(weights.resident_pages(), 1);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &vec![0.125; (width * width) as usize]);
    runtime.run(&program);
    let written = (0..(width * width))
        .map(|index| (index as f32 * 0.001) - 0.5)
        .collect::<Vec<_>>();
    runtime.write(&program, weight, &written);
    let checkpoint = runtime.checkpoint(&weights);
    let restored = runtime.load(&graph, &checkpoint);
    assert_eq!(
        runtime.checkpoint(&restored).bytes(),
        checkpoint.bytes(),
        "a streamed store checkpoints and restores every word it holds",
    );
    assert_eq!(
        runtime.read(&program, weight),
        written,
        "a streamed store keeps the weights the host wrote beside a checkpoint of them",
    );
}

fn quantized_weights_cross_pages(backends: Backends) {
    let rows = 256;
    let width = 128;
    let runtime = open(
        backends,
        MemoryRequest {
            resident_weight_bytes: Some(3 * (1 << 14)),
            ..Default::default()
        },
    );
    let graph = Graph::new();
    let first = graph.quantized_parameter(
        Shape::matrix(rows, width),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        0.05,
    );
    let second = graph.quantized_parameter(
        Shape::matrix(rows, width),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        0.05,
    );
    let data = graph.input(Shape::matrix(8, rows), Element::Single);
    let out = graph.add(graph.matmul(data, first), graph.matmul(data, second));
    graph.retain(out);
    let weights = runtime.weights(&graph);
    assert!(
        weights.pages() > weights.resident_pages(),
        "two int8 weights of {} pages keep {} of them beside the budget",
        weights.pages(),
        weights.resident_pages(),
    );
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, data, &vec![0.25; 8 * rows as usize]);
    runtime.run(&program);
    let written = (0..(rows * width))
        .map(|index| ((index % 17) as f32 - 8.0) * 0.05)
        .collect::<Vec<_>>();
    runtime.write(&program, first, &written);
    let read = runtime.read(&program, first);
    assert_eq!(read.len(), written.len());
    for (at, (written, read)) in written.iter().zip(&read).enumerate() {
        assert!(
            (written - read).abs() <= 0.03,
            "a quantized weight of a streamed store came back as {read} where {written} was stored at {at}",
        );
    }
    let observed = runtime.read(&program, out);
    assert!(
        observed.iter().all(|value| value.is_finite()),
        "a streamed store ran a product over int8 weights",
    );
}

#[test]
fn a_streamed_weight_store_trains_the_model_a_resident_store_trains() {
    for backends in Backends::PLATFORM {
        train_streamed_like_the_resident_store(backends);
    }
}

#[test]
fn a_streamed_store_keeps_its_resident_words_within_its_budget() {
    for backends in Backends::PLATFORM {
        resident_words_stay_within_the_budget(backends);
    }
}

#[test]
fn a_weight_page_budget_that_cannot_hold_one_task_is_refused() {
    for backends in Backends::PLATFORM {
        a_budget_below_one_task_is_refused(backends);
    }
}

#[test]
fn a_streamed_store_reads_back_the_weights_the_host_writes() {
    for backends in Backends::PLATFORM {
        host_writes_come_back(backends);
    }
}

#[test]
fn a_streamed_store_carries_every_word_of_a_checkpoint() {
    for backends in Backends::PLATFORM {
        checkpoints_carry_every_word(backends);
    }
}

#[test]
fn a_streamed_store_reads_and_writes_quantized_weights_across_pages() {
    for backends in Backends::PLATFORM {
        quantized_weights_cross_pages(backends);
    }
}

const DEEP_WIDTH: u32 = 256;
const DEEP_LAYERS: u32 = 6;
const DEEP_BATCH: u32 = 32;
const DEEP_STEPS: u32 = 4;
const DEEP_BYTES: u64 = 96 * (1 << 14);

fn deep_run(backends: Backends, memory: MemoryRequest) -> (Vec<f32>, f32) {
    let runtime = open(
        backends,
        MemoryRequest {
            readback_bytes: 4 << 20,
            ..memory
        },
    );
    let graph = Graph::new();
    let init = Init::Uniform {
        low: -0.02,
        high: 0.02,
    };
    let mut layers = Vec::new();
    for _ in 0..DEEP_LAYERS {
        layers.push((
            graph.parameter(Shape::matrix(DEEP_WIDTH, DEEP_WIDTH), init, Element::Single),
            graph.parameter(Shape::matrix(1, DEEP_WIDTH), Init::Zero, Element::Single),
        ));
    }
    let data = graph.input(Shape::matrix(DEEP_BATCH, DEEP_WIDTH), Element::Single);
    let mut value = data;
    for (weight, bias) in &layers {
        value = graph.relu(graph.add(graph.matmul(value, *weight), *bias));
    }
    let loss = graph.sum(value);
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let descent = graph.fill(Shape::scalar(), -0.001);
    for (weight, bias) in &layers {
        graph.add_into(*weight, graph.mul(gradients.of(*weight), descent));
        graph.add_into(*bias, graph.mul(gradients.of(*bias), descent));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let numbers = (0..(DEEP_BATCH * DEEP_WIDTH))
        .map(|index| ((index % 13) as f32) * 0.01 - 0.06)
        .collect::<Vec<_>>();
    let mut observed = 0.0;
    for _ in 0..DEEP_STEPS {
        runtime.write(&program, data, &numbers);
        runtime.run(&program).seconds();
        observed = runtime.read(&program, loss)[0];
    }
    let parameters = layers
        .iter()
        .flat_map(|(weight, bias)| {
            let mut values = runtime.read(&program, *weight);
            values.extend(runtime.read(&program, *bias));
            values
        })
        .collect();
    (parameters, observed)
}

#[test]
fn a_streamed_store_trains_a_model_that_no_budget_can_hold() {
    for backends in Backends::PLATFORM {
        let (resident, resident_loss) = deep_run(backends, MemoryRequest::default());
        let (streamed, streamed_loss) = deep_run(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(DEEP_BYTES),
                ..Default::default()
            },
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a streamed store reached {observed} where the resident store reached {expected} at {at}",
            );
        }
        assert!(
            (resident_loss - streamed_loss).abs() <= 1e-5,
            "a streamed store held a loss of {streamed_loss} where the resident store held {resident_loss}",
        );
    }
}

const CHURN_WIDTH: u32 = 256;
const CHURN_LAYERS: u32 = 12;
const CHURN_BATCH: u32 = 16;
const CHURN_STEPS: u32 = 4;
const CHURN_BYTES: u64 = 128 * (1 << 14);

fn churn_run(backends: Backends, memory: MemoryRequest) -> (Vec<f32>, f32, u64, u64) {
    let runtime = open(
        backends,
        MemoryRequest {
            readback_bytes: 4 << 20,
            ..memory
        },
    );
    let graph = Graph::new();
    let init = Init::Uniform {
        low: -0.02,
        high: 0.02,
    };
    let mut layers = Vec::new();
    for _ in 0..CHURN_LAYERS {
        layers.push(graph.parameter(
            Shape::matrix(CHURN_WIDTH, CHURN_WIDTH),
            init,
            Element::Single,
        ));
    }
    let data = graph.input(Shape::matrix(CHURN_BATCH, CHURN_WIDTH), Element::Single);
    let target = graph.input(Shape::matrix(CHURN_BATCH, CHURN_WIDTH), Element::Single);
    let mut value = data;
    for weight in &layers {
        value = graph.tanh(graph.matmul(value, *weight));
    }
    let difference = graph.sub(value, target);
    let loss = graph.sum(graph.mul(difference, difference));
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let descent = graph.fill(Shape::scalar(), -0.001);
    for weight in &layers {
        graph.add_into(*weight, graph.mul(gradients.of(*weight), descent));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations = (0..(CHURN_BATCH * CHURN_WIDTH))
        .map(|index| ((index * 37) % 101) as f32 / 101.0 - 0.5)
        .collect::<Vec<_>>();
    let targets = (0..(CHURN_BATCH * CHURN_WIDTH))
        .map(|index| ((index * 53) % 97) as f32 / 97.0 - 0.5)
        .collect::<Vec<_>>();
    let mut observed = 0.0;
    for _ in 0..CHURN_STEPS {
        runtime.write(&program, data, &observations);
        runtime.write(&program, target, &targets);
        runtime.run(&program);
        observed = runtime.read(&program, loss)[0];
    }
    let parameters = layers
        .iter()
        .flat_map(|weight| runtime.read(&program, *weight))
        .collect();
    (
        parameters,
        observed,
        weights.readback_pages(),
        weights.readback_transfers(),
    )
}

#[test]
fn a_streamed_store_reads_the_pages_it_churns_back_in_windows() {
    for backends in Backends::PLATFORM {
        let (resident, resident_loss, _, _) = churn_run(backends, MemoryRequest::default());
        let (streamed, streamed_loss, pages, transfers) = churn_run(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(CHURN_BYTES),
                ..Default::default()
            },
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a churning store reached {observed} where the resident store reached {expected} at {at}",
            );
        }
        assert!(
            (resident_loss - streamed_loss).abs() <= 1e-5,
            "a churning store held a loss of {streamed_loss} where the resident store held {resident_loss}",
        );
        assert!(
            pages > 16 * CHURN_LAYERS as u64,
            "a store of {CHURN_BYTES} bytes read {pages} pages back over {CHURN_STEPS} steps and never left its deferral window",
        );
        assert!(
            transfers * 16 <= pages,
            "a store read {pages} pages back in {transfers} transfers, and one transfer carries a window of pages",
        );
    }
}
