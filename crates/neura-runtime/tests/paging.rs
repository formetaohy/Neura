use neura_abi::Element;
use neura_graph::{Free, Graph, Init, Shape, Value};
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

fn churn_run(backends: Backends, memory: MemoryRequest) -> (Vec<f32>, f32, u64, u64, u32) {
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
        program.weight_windows(),
    )
}

#[test]
fn a_streamed_store_reads_the_pages_it_churns_back_in_windows() {
    for backends in Backends::PLATFORM {
        let (resident, resident_loss, _, _, _) = churn_run(backends, MemoryRequest::default());
        let (streamed, streamed_loss, pages, transfers, windows) = churn_run(
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
        assert!(
            windows <= CHURN_LAYERS,
            "a step of {CHURN_LAYERS} layers dispatched {windows} weight windows, and a window carries the tiles of many layers",
        );
    }
}

const FREE_LAYERS: u32 = 6;
const FREE_WIDTH: u32 = 256;
const FREE_BOUND: u32 = 32;
const FREE_STEPS: u32 = 4;
const FREE_BYTES: u64 = 96 * (1 << 14);

fn free_run(backends: Backends, memory: MemoryRequest) -> (Vec<f32>, Vec<f32>, u32, usize) {
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
    for _ in 0..FREE_LAYERS {
        layers.push((
            graph.parameter(Shape::matrix(FREE_WIDTH, FREE_WIDTH), init, Element::Single),
            graph.parameter(Shape::matrix(1, FREE_WIDTH), Init::Zero, Element::Single),
        ));
    }
    let batch = graph.free(FREE_BOUND);
    let data = graph.input(
        Shape::matrix(FREE_BOUND, FREE_WIDTH).freed(&[(2, batch)]),
        Element::Single,
    );
    let target = graph.input(
        Shape::matrix(FREE_BOUND, FREE_WIDTH).freed(&[(2, batch)]),
        Element::Single,
    );
    let mut value = data;
    for (weight, bias) in &layers {
        value = graph.relu(graph.add(graph.matmul(value, *weight), *bias));
    }
    let difference = graph.sub(value, target);
    let loss = graph.sum(graph.mul(difference, difference));
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let descent = graph.fill(Shape::scalar(), -0.001);
    for (weight, bias) in &layers {
        graph.add_into(*weight, graph.mul(gradients.of(*weight), descent));
        graph.add_into(*bias, graph.mul(gradients.of(*bias), descent));
    }
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations = (0..(FREE_BOUND * FREE_WIDTH))
        .map(|index| ((index * 37) % 101) as f32 / 101.0 - 0.5)
        .collect::<Vec<_>>();
    let targets = (0..(FREE_BOUND * FREE_WIDTH))
        .map(|index| ((index * 53) % 97) as f32 / 97.0 - 0.5)
        .collect::<Vec<_>>();
    let mut losses = Vec::new();
    for step in 0..FREE_STEPS {
        let live = [FREE_BOUND, FREE_BOUND / 3, 1, FREE_BOUND][step as usize];
        runtime.bind(&program, &[live]);
        runtime.write(
            &program,
            data,
            &observations[..(live * FREE_WIDTH) as usize],
        );
        runtime.write(&program, target, &targets[..(live * FREE_WIDTH) as usize]);
        runtime.run(&program);
        losses.push(runtime.read(&program, loss)[0]);
    }
    let parameters = layers
        .iter()
        .flat_map(|(weight, bias)| {
            let mut values = runtime.read(&program, *weight);
            values.extend(runtime.read(&program, *bias));
            values
        })
        .collect();
    (
        parameters,
        losses,
        program.weight_windows(),
        program.planned_windows(),
    )
}

#[test]
fn a_streamed_store_pages_the_batch_a_binding_holds() {
    for backends in Backends::PLATFORM {
        let (resident, resident_losses, _, _) = free_run(backends, MemoryRequest::default());
        let (streamed, streamed_losses, windows, planned) = free_run(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(FREE_BYTES),
                ..Default::default()
            },
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a store of a free batch reached {observed} where the resident store reached {expected} at {at}",
            );
        }
        assert_eq!(resident_losses.len(), streamed_losses.len());
        for (at, (expected, observed)) in resident_losses.iter().zip(&streamed_losses).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "step {at} of a free batch held a loss of {observed} where the resident store held {expected}",
            );
        }
        assert!(
            windows <= FREE_LAYERS,
            "a step of {FREE_LAYERS} layers over a free batch dispatched {windows} weight windows, and a window carries the tiles of many layers",
        );
        assert_eq!(
            planned, 3,
            "a program that walked {FREE_STEPS} bindings of 3 shapes planned the pages of {planned} of them, and a shape it has walked keeps the window its pages belong to",
        );
    }
}

fn wide_run(backends: Backends, memory: MemoryRequest) -> (f32, u32, u32) {
    let runtime = open(
        backends,
        MemoryRequest {
            readback_bytes: 4 << 20,
            ..memory
        },
    );
    let graph = Graph::new();
    let batch = graph.free(64);
    let weight = graph.parameter(
        Shape::matrix(512, 512),
        Init::Uniform {
            low: -0.01,
            high: 0.01,
        },
        Element::Single,
    );
    let data = graph.input(Shape::matrix(64, 512).freed(&[(2, batch)]), Element::Single);
    let loss = graph.sum(graph.relu(graph.matmul(data, weight)));
    graph.retain(loss);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.bind(&program, &[64]);
    runtime.write(&program, data, &vec![0.25; 64 * 512]);
    runtime.run(&program);
    let observed = runtime.read(&program, loss)[0];
    (observed, weights.pages(), weights.resident_pages())
}

#[test]
fn a_streamed_store_holds_a_weight_no_budget_holds_whole() {
    for backends in Backends::PLATFORM {
        let (expected, pages, resident) = wide_run(backends, MemoryRequest::default());
        assert_eq!(
            resident, pages,
            "a store without a weight budget holds every page its model carries",
        );
        let (observed, pages, held) = wide_run(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(16 * (1 << 14)),
                ..Default::default()
            },
        );
        assert_eq!(pages, resident);
        assert!(
            held < pages,
            "a budget of 16 pages holds {held} of the {pages} pages the weight of this model carries",
        );
        assert!(
            (expected - observed).abs() <= 1e-4,
            "a store that pages a weight no budget holds whole walked to a loss of {observed} where the resident store walked to {expected}",
        );
    }
}

#[test]
fn a_streamed_store_pages_a_model_a_device_count_rules() {
    for backends in Backends::PLATFORM {
        let counted = |memory: MemoryRequest| {
            let runtime = open(backends, memory);
            let graph = Graph::new();
            let probe = graph.input(Shape::of([1, 1, 8, 1]), Element::Single);
            let tokens = graph.input(Shape::of([1, 1, 8, 128]), Element::Single);
            let first = graph.parameter(
                Shape::matrix(128, 128),
                Init::Uniform {
                    low: -0.05,
                    high: 0.05,
                },
                Element::Single,
            );
            let second = graph.parameter(
                Shape::matrix(128, 128),
                Init::Uniform {
                    low: -0.05,
                    high: 0.05,
                },
                Element::Single,
            );
            let count = graph.sum_axis(probe, 2);
            let live = graph.trim(tokens, 2, count);
            let out = graph.matmul(graph.matmul(live, first), second);
            graph.retain(out);
            let weights = runtime.weights(&graph);
            let program = runtime.compile(&graph, &weights);
            runtime.write(&program, probe, &[1.0; 8]);
            let tokens_data = (0..(8 * 128))
                .map(|index| ((index * 37) % 101) as f32 / 101.0 - 0.5)
                .collect::<Vec<_>>();
            let first_data = (0..(128 * 128))
                .map(|index| ((index * 53) % 97) as f32 / 97.0 - 0.5)
                .collect::<Vec<_>>();
            let second_data = (0..(128 * 128))
                .map(|index| ((index * 71) % 89) as f32 / 89.0 - 0.5)
                .collect::<Vec<_>>();
            runtime.write(&program, tokens, &tokens_data);
            runtime.write(&program, first, &first_data);
            runtime.write(&program, second, &second_data);
            runtime.run(&program);
            (runtime.read(&program, out), program.weight_windows())
        };
        let (resident, windows) = counted(MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        });
        assert_eq!(
            windows, 0,
            "a store that holds every weight resident walks no window",
        );
        let (streamed, windows) = counted(MemoryRequest {
            readback_bytes: 4 << 20,
            resident_weight_bytes: Some(4 * (1 << 14)),
            ..Default::default()
        });
        assert!(
            windows >= 1,
            "a store of {} pages holds {} of them, and the weights of a model a device count rules walk in a window of their own",
            8,
            4,
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a store of a model a device count rules reached {observed} where the resident store reached {expected} at {at}",
            );
        }
    }
}

const TABLE_ROWS: u32 = 4096;
const TABLE_WIDTH: u32 = 8;
const TABLE_BYTES: u64 = 2 * (1 << 14);
const _: () = assert!(TABLE_ROWS as u64 * TABLE_WIDTH as u64 * 4 > TABLE_BYTES);

fn table_graph(
    graph: &Graph<'static>,
    extent: Free,
    bound: u32,
) -> (Value<'static>, Value<'static>, Value<'static>) {
    let indices = graph.input(
        Shape::of([1, bound, 1, 1]).freed(&[(1, extent)]),
        Element::Single,
    );
    let table = graph.named_parameter(
        "table",
        Shape::matrix(TABLE_ROWS, TABLE_WIDTH),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let gathered = graph.gather(table, indices);
    graph.retain(gathered);
    (indices, table, gathered)
}

fn table_walk(
    backends: Backends,
    memory: MemoryRequest,
    steps: &[&[u32]],
) -> (Vec<Vec<f32>>, Vec<u32>) {
    let runtime = open(backends, memory);
    let bound = steps.iter().map(|rows| rows.len()).max().expect("a step") as u32;
    let graph: Graph<'static> = Graph::new();
    let extent = graph.free(bound);
    let (indices, table, gathered) = table_graph(&graph, extent, bound);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let mut walked = Vec::new();
    let mut windows = Vec::new();
    for rows in steps {
        runtime.bind(&program, &[rows.len() as u32]);
        runtime.declare_rows(&program, table, rows);
        runtime.write(
            &program,
            indices,
            &rows.iter().map(|row| *row as f32).collect::<Vec<f32>>(),
        );
        runtime.run(&program);
        walked.push(runtime.read(&program, gathered));
        windows.push(program.weight_windows());
    }
    (walked, windows)
}

#[test]
fn a_streamed_table_reads_the_rows_a_host_names() {
    for backends in Backends::PLATFORM {
        let first: Vec<u32> = (0..8).collect();
        let second: Vec<u32> = (0..8).map(|row| 512 + row).collect();
        let third: Vec<u32> = vec![0, 512, 1, 513];
        let steps: Vec<&[u32]> = vec![&first, &second, &third];
        let (resident, _) = table_walk(backends, MemoryRequest::default(), &steps);
        let (streamed, windows) = table_walk(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(TABLE_BYTES),
                ..Default::default()
            },
            &steps,
        );
        assert_eq!(resident.len(), streamed.len());
        for (step, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert_eq!(expected.len(), observed.len());
            for (at, (expected, observed)) in expected.iter().zip(observed).enumerate() {
                assert!(
                    (expected - observed).abs() <= 1e-5,
                    "step {step} of a table walk reached {observed} where the resident store reached {expected} at {at}",
                );
            }
        }
        assert_eq!(
            windows,
            vec![1; steps.len()],
            "a table walk of rows a host names dispatches one weight window by the pages those rows lie on",
        );
    }
}

#[test]
fn a_table_walk_a_host_leaves_unnamed_refuses() {
    for backends in Backends::PLATFORM {
        let runtime = open(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(TABLE_BYTES),
                ..Default::default()
            },
        );
        let graph: Graph<'static> = Graph::new();
        let extent = graph.free(4);
        let (indices, table, gathered) = table_graph(&graph, extent, 4);
        let weights = runtime.weights(&graph);
        let program = runtime.compile(&graph, &weights);
        runtime.bind(&program, &[4]);
        runtime.write(&program, indices, &[0.0, 1.0, 2.0, 3.0]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.run(&program);
            }))
            .is_err(),
            "a table of {TABLE_ROWS} rows walks more pages than the resident budget holds while no host names the rows it reads",
        );

        let elsewhere = [512u32, 513, 514, 515];
        runtime.declare_rows(&program, table, &elsewhere);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.run(&program);
                runtime.read(&program, gathered);
            }))
            .is_err(),
            "a host that names rows {elsewhere:?} of a table whose indices read page 0 of refuses",
        );

        let named = [0u32, 1, 2, 3];
        runtime.declare_rows(&program, table, &named);
        runtime.run(&program);
        assert_eq!(
            runtime.read(&program, gathered).len(),
            4 * TABLE_WIDTH as usize
        );

        runtime.write(&program, indices, &[0.0, 512.0, 1024.0, 1536.0]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.run(&program);
                runtime.read(&program, gathered);
            }))
            .is_err(),
            "a host that names rows {named:?} of a table cannot read rows on four pages of it, and the device refuses the pages it does not hold",
        );

        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.declare_rows(&program, table, &[TABLE_ROWS]);
            }))
            .is_err(),
            "a host names no row beyond the {TABLE_ROWS} a table holds",
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.declare_rows(&program, indices, &named);
            }))
            .is_err(),
            "a host names the rows of a table a task gathers, and value {indices:?} is no table",
        );
    }
}

fn table_training(backends: Backends, memory: MemoryRequest) -> Vec<f32> {
    let runtime = open(backends, memory);
    let graph: Graph<'static> = Graph::new();
    let extent = graph.free(8);
    let (indices, table, gathered) = table_graph(&graph, extent, 8);
    let target = graph.input(
        Shape::of([1, 8, 1, TABLE_WIDTH]).freed(&[(1, extent)]),
        Element::Single,
    );
    let difference = graph.sub(gathered, target);
    let loss = graph.sum(graph.mul(difference, difference));
    graph.retain(loss);
    let gradients = graph.backward(loss);
    graph.add_into(
        table,
        graph.mul(gradients.of(table), graph.fill(Shape::scalar(), -0.01)),
    );
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let rows: Vec<u32> = (0..8).collect();
    let targets = (0..(8 * TABLE_WIDTH))
        .map(|index| ((index * 37) % 101) as f32 / 101.0 - 0.5)
        .collect::<Vec<_>>();
    for _ in 0..4 {
        runtime.bind(&program, &[8]);
        runtime.declare_rows(&program, table, &rows);
        runtime.write(
            &program,
            indices,
            &rows.iter().map(|row| *row as f32).collect::<Vec<f32>>(),
        );
        runtime.write(&program, target, &targets);
        runtime.run(&program);
    }
    runtime.read(&program, table)
}

#[test]
fn a_streamed_table_trains_the_rows_a_host_names() {
    for backends in Backends::PLATFORM {
        let resident = table_training(backends, MemoryRequest::default());
        let streamed = table_training(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(TABLE_BYTES),
                ..Default::default()
            },
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a streamed table trained row {at} to {observed} where the resident table reached {expected}",
            );
        }
        let moved = (0..8)
            .flat_map(|row| {
                (0..TABLE_WIDTH)
                    .map(move |column| (row as usize) * TABLE_WIDTH as usize + column as usize)
            })
            .any(|at| resident[at] != 0.0);
        assert!(
            moved,
            "a step over a streamed table moves the rows it weighed",
        );
    }
}

fn static_table(backends: Backends, memory: MemoryRequest, rows: &[u32]) -> (Vec<f32>, u32) {
    let runtime = open(backends, memory);
    let graph: Graph<'static> = Graph::new();
    let count = rows.len() as u32;
    let indices = graph.input(Shape::of([1, count, 1, 1]), Element::Single);
    let table = graph.named_parameter(
        "table",
        Shape::matrix(TABLE_ROWS, TABLE_WIDTH),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let gathered = graph.gather(table, indices);
    graph.retain(gathered);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.declare_rows(&program, table, rows);
    runtime.write(
        &program,
        indices,
        &rows.iter().map(|row| *row as f32).collect::<Vec<f32>>(),
    );
    runtime.run(&program);
    (runtime.read(&program, gathered), program.weight_windows())
}

#[test]
fn a_streamed_table_reads_the_rows_a_host_names_at_a_fixed_shape() {
    for backends in Backends::PLATFORM {
        let rows: Vec<u32> = (0..8).collect();
        let (resident, _) = static_table(backends, MemoryRequest::default(), &rows);
        let (streamed, windows) = static_table(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(TABLE_BYTES),
                ..Default::default()
            },
            &rows,
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a fixed shape table walk reached {observed} where the resident store reached {expected} at {at}",
            );
        }
        assert_eq!(
            windows, 1,
            "a table walk of rows a host names dispatches one weight window",
        );
    }
}

const CONV_CHANNELS: u32 = 8192;
const CONV_BYTES: u64 = 6 * (1 << 14);
const CONV_PAGES: u32 = (CONV_CHANNELS * 9) / 4096;

fn convolution_run(backends: Backends, memory: MemoryRequest) -> (Vec<f32>, f32, u32) {
    let runtime = open(
        backends,
        MemoryRequest {
            readback_bytes: 4 << 20,
            ..memory
        },
    );
    let graph = Graph::new();
    let images = graph.input(Shape::of([1, 1, 4, 4]), Element::Single);
    let filter = graph.parameter(
        Shape::of([CONV_CHANNELS, 1, 3, 3]),
        Init::Uniform {
            low: -0.02,
            high: 0.02,
        },
        Element::Single,
    );
    let convolved = graph.conv2d(images, filter, neura_graph::Window::sliding([3, 3]));
    let loss = graph.sum(graph.mul(convolved, convolved));
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let descent = graph.fill(Shape::scalar(), -0.001);
    graph.add_into(filter, graph.mul(gradients.of(filter), descent));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations = (0..16)
        .map(|index| ((index * 37) % 101) as f32 / 101.0 - 0.5)
        .collect::<Vec<_>>();
    for _ in 0..2 {
        runtime.write(&program, images, &observations);
        runtime.run(&program);
    }
    (
        runtime.read(&program, filter),
        runtime.read(&program, loss)[0],
        program.weight_windows(),
    )
}

#[test]
fn a_streamed_store_pages_the_filter_of_a_convolution() {
    assert!(
        (CONV_CHANNELS * 9).div_ceil(4096) >= 2,
        "a filter of {CONV_PAGES} pages holds no more than the budget books it",
    );
    for backends in Backends::PLATFORM {
        let (resident, resident_loss, _) = convolution_run(backends, MemoryRequest::default());
        let (streamed, streamed_loss, windows) = convolution_run(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(CONV_BYTES),
                ..Default::default()
            },
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a store of {CONV_BYTES} bytes reached {observed} where the resident store reached {expected} at {at}",
            );
        }
        assert!(
            (resident_loss - streamed_loss).abs() <= 1e-5,
            "a streamed convolution held a loss of {streamed_loss} where the resident store held {resident_loss}",
        );
        assert!(
            windows > 1,
            "a store of {CONV_BYTES} bytes walked {windows} weight windows over a filter of {CONV_PAGES} pages",
        );
    }
}

const INPUT_GRADIENT_CHANNELS: u32 = 8192;
const INPUT_GRADIENT_BYTES: u64 = 6 * (1 << 14);
const INPUT_GRADIENT_PAGES: u32 = (INPUT_GRADIENT_CHANNELS * 9) / 4096;

fn input_gradient_run(backends: Backends, memory: MemoryRequest) -> (Vec<f32>, Vec<f32>, f32, u32) {
    let runtime = open(
        backends,
        MemoryRequest {
            readback_bytes: 4 << 20,
            ..memory
        },
    );
    let graph = Graph::new();
    let images = graph.gradient_input(Shape::of([1, 1, 4, 4]), Element::Single);
    let filter = graph.parameter(
        Shape::of([INPUT_GRADIENT_CHANNELS, 1, 3, 3]),
        Init::Uniform {
            low: -0.02,
            high: 0.02,
        },
        Element::Single,
    );
    let convolved = graph.conv2d(images, filter, neura_graph::Window::sliding([3, 3]));
    let loss = graph.sum(graph.mul(convolved, convolved));
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let descent = graph.fill(Shape::scalar(), -0.001);
    graph.add_into(filter, graph.mul(gradients.of(filter), descent));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let observations = (0..16)
        .map(|index| ((index * 37) % 101) as f32 / 101.0 - 0.5)
        .collect::<Vec<_>>();
    for _ in 0..STEPS {
        runtime.write(&program, images, &observations);
        runtime.run(&program);
    }
    (
        runtime.read(&program, filter),
        runtime.read(&program, gradients.of(images)),
        runtime.read(&program, loss)[0],
        program.weight_windows(),
    )
}

#[test]
fn a_streamed_store_trains_the_input_gradient_of_a_convolution() {
    assert!(
        INPUT_GRADIENT_PAGES > INPUT_GRADIENT_BYTES as u32 / (1 << 14),
        "a filter of {INPUT_GRADIENT_PAGES} pages holds no more than the budget books it",
    );
    for backends in Backends::PLATFORM {
        let (resident, resident_gradient, resident_loss, _) =
            input_gradient_run(backends, MemoryRequest::default());
        let (streamed, streamed_gradient, streamed_loss, windows) = input_gradient_run(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(INPUT_GRADIENT_BYTES),
                ..Default::default()
            },
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a store of {INPUT_GRADIENT_BYTES} bytes reached {observed} where the resident store reached {expected} at {at}",
            );
        }
        assert_eq!(resident_gradient.len(), streamed_gradient.len());
        for (at, (expected, observed)) in
            resident_gradient.iter().zip(&streamed_gradient).enumerate()
        {
            assert!(
                (expected - observed).abs() <= 1e-4,
                "a gradient the input of a streamed convolution walks reached {observed} where the resident store reached {expected} at {at}",
            );
        }
        assert!(
            (resident_loss - streamed_loss).abs() <= 1e-5,
            "a streamed convolution held a loss of {streamed_loss} where the resident store held {resident_loss}",
        );
        assert!(
            windows > 1,
            "a store of {INPUT_GRADIENT_BYTES} bytes walked {windows} weight windows over a filter of {INPUT_GRADIENT_PAGES} pages",
        );
    }
}

const GROUPED_CHANNELS: u32 = 1024;
const GROUPED_INPUTS: u32 = 256;
const GROUPED_GROUPS: u32 = 8;
const GROUPED_BOUND: u32 = 4;
const GROUPED_BYTES: u64 = 4 * (1 << 14);

fn grouped_run(backends: Backends, memory: MemoryRequest) -> (Vec<f32>, Vec<Vec<f32>>) {
    let runtime = open(
        backends,
        MemoryRequest {
            readback_bytes: 4 << 20,
            ..memory
        },
    );
    let graph = Graph::new();
    let batch = graph.free(GROUPED_BOUND);
    let images = graph.gradient_input(
        Shape::of([GROUPED_BOUND, GROUPED_INPUTS, 4, 4]).freed(&[(0, batch)]),
        Element::Single,
    );
    let filter = graph.parameter(
        Shape::of([GROUPED_CHANNELS, GROUPED_INPUTS / GROUPED_GROUPS, 3, 3]),
        Init::Uniform {
            low: -0.02,
            high: 0.02,
        },
        Element::Single,
    );
    let convolved = graph.conv2d(images, filter, neura_graph::Window::sliding([3, 3]));
    let loss = graph.sum(graph.mul(convolved, convolved));
    graph.retain(loss);
    let gradients = graph.backward(loss);
    graph.retain(gradients.of(images));
    let descent = graph.fill(Shape::scalar(), -0.001);
    graph.add_into(filter, graph.mul(gradients.of(filter), descent));
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let mut observed = Vec::new();
    for live in [GROUPED_BOUND, 3, 1, 0] {
        runtime.bind(&program, &[live]);
        runtime.write(
            &program,
            images,
            &vec![0.25; (live * GROUPED_INPUTS * 16) as usize],
        );
        runtime.run(&program);
        observed.push(runtime.read(&program, gradients.of(images)));
    }
    (runtime.read(&program, filter), observed)
}

#[test]
fn a_streamed_store_trains_a_grouped_convolution_of_every_batch_a_binding_holds() {
    for backends in Backends::PLATFORM {
        let (resident, resident_gradients) = grouped_run(backends, MemoryRequest::default());
        let (streamed, streamed_gradients) = grouped_run(
            backends,
            MemoryRequest {
                resident_weight_bytes: Some(GROUPED_BYTES),
                ..Default::default()
            },
        );
        assert_eq!(resident.len(), streamed.len());
        for (at, (expected, observed)) in resident.iter().zip(&streamed).enumerate() {
            assert!(
                (expected - observed).abs() <= 1e-5,
                "a store of {GROUPED_BYTES} bytes reached {observed} where the resident store reached {expected} at {at}",
            );
        }
        assert_eq!(resident_gradients.len(), streamed_gradients.len());
        for (live, (expected, observed)) in resident_gradients
            .iter()
            .zip(&streamed_gradients)
            .enumerate()
        {
            assert_eq!(expected.len(), observed.len());
            for (at, (expected, observed)) in expected.iter().zip(observed).enumerate() {
                assert!(
                    (expected - observed).abs() <= 1e-4,
                    "a binding of {live} reached {observed} where the resident store reached {expected} at {at}",
                );
            }
        }
    }
}
