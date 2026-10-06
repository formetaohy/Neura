mod dataset;
mod report;
mod source;

use dataset::{Dataset, Order, materialize};
use neura::{
    AdamW, Conv2d, Element, Free, Graph, Init, Linear, Pool, Program, Runtime, RuntimeRequest,
    Shape, Value, Window, cross_entropy,
};
use source::Split;
use std::time::Instant;

pub(crate) const CLASSES: u32 = 10;
const BOUND: u32 = 128;
const EPOCHS: u32 = 4;
const HIDDEN: u32 = 64;
const FILTERS: [u32; 2] = [8, 16];
const WINDOW: [u32; 2] = [3, 3];
const POOL: [u32; 2] = [2, 2];
const REPORT_EVERY: u32 = 64;
const GALLERY: u32 = 6;
const PER_ROW: usize = 3;
const RATE: f32 = 1e-3;
const SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const HEAP_BYTES: u64 = 1 << 26;

struct Model<'g> {
    conv1: Conv2d<'g>,
    conv2: Conv2d<'g>,
    head: Linear<'g>,
    out: Linear<'g>,
    flattened: u32,
}

impl<'g> Model<'g> {
    fn new(graph: &Graph<'g>, flattened: u32) -> Self {
        let init = Init::Uniform {
            low: -0.25,
            high: 0.25,
        };
        let window = Window::new(WINDOW, [1, 1], [1, 1]);
        Self {
            conv1: Conv2d::new(
                graph,
                "conv1",
                [1, FILTERS[0]],
                1,
                window,
                init,
                Element::Single,
            ),
            conv2: Conv2d::new(
                graph,
                "conv2",
                [FILTERS[0], FILTERS[1]],
                1,
                window,
                init,
                Element::Single,
            ),
            head: Linear::new(graph, "head", flattened, HIDDEN, init, Element::Single),
            out: Linear::new(graph, "out", HIDDEN, CLASSES, init, Element::Single),
            flattened,
        }
    }

    fn logits(&self, graph: &Graph<'g>, images: Value<'g>, batch: Free) -> Value<'g> {
        let pool = Window::new(POOL, POOL, [0, 0]);
        let first = graph.pool2d(
            graph.relu(self.conv1.forward(graph, images)),
            pool,
            Pool::Max,
        );
        let second = graph.pool2d(
            graph.relu(self.conv2.forward(graph, first)),
            pool,
            Pool::Max,
        );
        let flattened = Shape::of([batch.bound(), 1, 1, self.flattened]).freed(&[(0, batch)]);
        let rows = graph.reshape(second, flattened);
        self.out
            .forward(graph, graph.relu(self.head.forward(graph, rows)))
    }

    fn parameters(&self) -> Vec<Value<'g>> {
        let mut parameters = self.conv1.parameters().to_vec();
        parameters.extend(self.conv2.parameters());
        parameters.extend(self.head.parameters());
        parameters.extend(self.out.parameters());
        parameters
    }
}

struct Session<'g> {
    images: Value<'g>,
    labels: Value<'g>,
    logits: Value<'g>,
    loss: Value<'g>,
}

fn session<'g>(graph: &Graph<'g>, model: &Model<'g>, rows: u32, columns: u32) -> Session<'g> {
    let batch = graph.free(BOUND);
    let images = graph.input(
        Shape::of([BOUND, 1, rows, columns]).freed(&[(0, batch)]),
        Element::Single,
    );
    let labels = graph.input(
        Shape::of([BOUND, 1, 1, 1]).freed(&[(0, batch)]),
        Element::Single,
    );
    let logits = model.logits(graph, images, batch);
    let loss = cross_entropy(graph, logits, graph.one_hot(labels, CLASSES));
    graph.retain(logits);
    graph.retain(loss);
    Session {
        images,
        labels,
        logits,
        loss,
    }
}

fn main() {
    let directory = source::directory();
    let train = Dataset::load(&directory, Split::Train);
    let test = Dataset::load(&directory, Split::Test);
    let (rows, columns) = (train.rows(), train.columns());
    assert!(
        rows % (POOL[0] * POOL[0]) == 0 && columns % (POOL[1] * POOL[1]) == 0,
        "two {POOL:?} pools halve {rows} by {columns} into nothing",
    );
    let flattened = FILTERS[1] * (rows / 4) * (columns / 4);

    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        heap_bytes: HEAP_BYTES,
        ..RuntimeRequest::default()
    }))
    .expect("a device to train on");
    report::device(runtime.context().adapter_info());
    report::dataset(&train, &test, &directory);

    let training = Graph::new();
    let model = Model::new(&training, flattened);
    let learn = session(&training, &model, rows, columns);
    let gradients = training.backward(learn.loss);
    let mut optimizer = AdamW::new(&training, "optimizer", RATE, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&training, &model.parameters());
    optimizer.step(&training, &gradients);

    let inference = Graph::new();
    let inference_model = Model::new(&inference, flattened);
    let evaluate = session(&inference, &inference_model, rows, columns);

    let weights = runtime.weights(&training);
    let learning = runtime.compile(&training, &weights);
    runtime.rebind(&weights, &inference);
    let inferring = runtime.compile(&inference, &weights);
    report::program("training graph", &learning);
    report::program("inference graph", &inferring);
    println!(
        "device: {} device programs for {} plans",
        runtime.declared_kernels(),
        runtime.built_plans(),
    );

    let order = Order::shuffled(train.len(), SEED);
    let steps = train.len().div_ceil(BOUND);
    let mut images = Vec::new();
    let mut labels = Vec::new();
    let mut device_seconds = 0.0;
    let mut host_seconds = 0.0;
    let mut taken = 0u32;
    for epoch in 1..=EPOCHS {
        let started = Instant::now();
        let mut sampled = 0.0f64;
        let mut samples = 0u32;
        for (index, batch) in order.batches(BOUND).enumerate() {
            materialize(&train, batch, &mut images, &mut labels);
            runtime.bind(&learning, &[batch.len() as u32]);
            runtime.write(&learning, learn.images, &images);
            runtime.write(&learning, learn.labels, &labels);
            let submitted = Instant::now();
            device_seconds += runtime.run(&learning).seconds();
            host_seconds += submitted.elapsed().as_secs_f64();
            taken += 1;
            if (index as u32).is_multiple_of(REPORT_EVERY) {
                let loss = runtime.read(&learning, learn.loss)[0];
                report::step(index as u32 + 1, steps, loss);
                sampled += f64::from(loss);
                samples += 1;
            }
        }
        let held_out = measure(&runtime, &inferring, &evaluate, &test);
        report::epoch(
            epoch,
            EPOCHS,
            (sampled / f64::from(samples)) as f32,
            held_out.loss,
            held_out.accuracy,
            test.len(),
            started.elapsed().as_secs_f64(),
        );
        if epoch == EPOCHS {
            report::confusion(&held_out.confusion);
        }
    }
    report::timing(
        device_seconds,
        host_seconds,
        taken,
        u64::from(train.len() * EPOCHS),
    );

    let gallery = classify(&runtime, &inferring, &evaluate, &test, GALLERY);
    println!("the inference program trains {BOUND} digits a step and classifies {GALLERY} here");
    report::gallery(
        &gallery.pixels,
        &gallery.labels,
        &gallery.guesses,
        rows,
        columns,
        PER_ROW,
    );
}

struct Held {
    accuracy: f64,
    loss: f32,
    confusion: [[u32; CLASSES as usize]; CLASSES as usize],
}

fn measure(
    runtime: &Runtime,
    program: &Program<'_>,
    session: &Session<'_>,
    test: &Dataset,
) -> Held {
    let order = Order::ordered(test.len());
    let mut images = Vec::new();
    let mut labels = Vec::new();
    let mut held = Held {
        accuracy: 0.0,
        loss: 0.0,
        confusion: [[0; CLASSES as usize]; CLASSES as usize],
    };
    let mut correct = 0u32;
    let mut loss = 0.0f64;
    for batch in order.batches(BOUND) {
        materialize(test, batch, &mut images, &mut labels);
        runtime.bind(program, &[batch.len() as u32]);
        runtime.write(program, session.images, &images);
        runtime.write(program, session.labels, &labels);
        runtime.run(program);
        let read = runtime.read_many(program, &[session.logits, session.loss]);
        loss += f64::from(read[1][0]) * batch.len() as f64;
        for (row, at) in batch.iter().enumerate() {
            let guess = guess_of(&read[0], row);
            held.confusion[test.label(*at) as usize][guess as usize] += 1;
            if guess == test.label(*at) {
                correct += 1;
            }
        }
    }
    held.accuracy = f64::from(correct) / f64::from(test.len());
    held.loss = (loss / f64::from(test.len())) as f32;
    held
}

struct Gallery {
    pixels: Vec<u8>,
    labels: Vec<u8>,
    guesses: Vec<u8>,
}

fn classify(
    runtime: &Runtime,
    program: &Program<'_>,
    session: &Session<'_>,
    test: &Dataset,
    count: u32,
) -> Gallery {
    let digits = (0..count).collect::<Vec<_>>();
    let mut images = Vec::new();
    let mut labels = Vec::new();
    materialize(test, &digits, &mut images, &mut labels);
    runtime.bind(program, &[count]);
    runtime.write(program, session.images, &images);
    runtime.write(program, session.labels, &labels);
    runtime.run(program);
    let logits = runtime.read(program, session.logits);
    let guesses = (0..count as usize)
        .map(|row| guess_of(&logits, row))
        .collect::<Vec<_>>();
    Gallery {
        pixels: digits
            .iter()
            .flat_map(|at| test.image(*at).iter().copied())
            .collect(),
        labels: digits.iter().map(|at| test.label(*at)).collect::<Vec<u8>>(),
        guesses,
    }
}

fn guess_of(logits: &[f32], row: usize) -> u8 {
    let row = &logits[row * CLASSES as usize..(row + 1) * CLASSES as usize];
    let mut best = 0usize;
    for (class, score) in row.iter().enumerate() {
        if *score > row[best] {
            best = class;
        }
    }
    best as u8
}
