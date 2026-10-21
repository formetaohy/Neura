use crate::config::Config;
use crate::model::{Model, Planes};
use crate::persona::{Plan, Table};
use crate::tokenizer::Tokenizer;
use neura::gguf::GgufFile;
use neura::{AdamW, Element, Graph, Overlay, Runtime, Shape, cross_entropy};
use std::path::PathBuf;
use std::time::Instant;

pub(crate) struct Request {
    pub corpus: String,
    pub rows: u32,
    pub steps: u32,
    pub rate: f32,
    pub seed: u64,
    pub persona: u32,
    pub init: Option<String>,
    pub save: PathBuf,
}

pub(crate) fn run(
    runtime: &Runtime,
    source: &GgufFile,
    tokenizer: &Tokenizer,
    config: &Config,
    plan: Plan,
    request: &Request,
) {
    let rows = request.rows;
    assert!(
        rows > 1,
        "a training window of {rows} tokens predicts no next token"
    );
    assert!(
        rows <= config.context,
        "a training window of {rows} tokens outruns the {} token context of this model",
        config.context,
    );
    assert!(
        request.steps > 0,
        "a training run of no step descends nothing",
    );
    let corpus = tokenizer.encode(&request.corpus);
    assert!(
        corpus.len() > rows as usize,
        "a corpus of {} tokens holds no window of {rows} tokens beside the token it predicts",
        corpus.len(),
    );
    let graph = Graph::new();
    let model = Model::new(
        &graph,
        source,
        config,
        Planes {
            capacity: rows + 1,
            window: rows + 1,
            bounded: false,
            batch: 1,
            rows,
            shared: false,
            prefix: false,
            kv: Element::Single,
            persona: Some(plan),
            training: true,
            training_persona: request.persona,
        },
    );
    let parameters = graph.parameters();
    let is_adapter = |value: neura::Value| {
        graph
            .name_of(value)
            .is_some_and(|name| name.starts_with("adapter."))
    };
    let adapters = parameters
        .iter()
        .copied()
        .filter(|value| is_adapter(*value))
        .collect::<Vec<_>>();
    let base = parameters
        .iter()
        .copied()
        .filter(|value| !is_adapter(*value))
        .collect::<Vec<_>>();
    assert!(
        !adapters.is_empty(),
        "a training run of no persona table learns nothing",
    );
    graph.freeze(&base);
    let token = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
    let label = graph.input(Shape::of([1, 1, rows, 1]), Element::Single);
    let cursor = graph.fill(Shape::scalar(), 0.0);
    let mask = graph.fill(Shape::of([1, 1, 1, 1]), 1.0);
    let hidden = model.layers(&graph, token, cursor, mask);
    let logits = model.logits(&graph, hidden);
    let loss = cross_entropy(&graph, logits, label);
    graph.retain(loss);
    let gradients = graph.backward(loss);
    let mut optimizer = AdamW::new(&graph, "optimizer", request.rate, 0.9, 0.999, 1e-8, 0.0);
    optimizer.track_all(&graph, &adapters);
    optimizer.step(&graph, &gradients);
    let weights = runtime.weights(&graph);
    match &request.init {
        Some(path) => {
            let table = Table::open(Some(path), request.seed);
            runtime.restore_weights_named(
                &weights,
                &Overlay::new(source, table.source(&graph)),
                &|_| true,
            );
        }
        None => {
            runtime.restore_weights_named(&weights, source, &|name| !name.starts_with("adapter."));
        }
    }
    let program = runtime.compile(&graph, &weights);
    let started = Instant::now();
    let mut state = request.seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut tokens = vec![0.0f32; rows as usize];
    let mut labels = vec![0.0f32; rows as usize];
    for step in 0..request.steps {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let start = (state >> 33) as usize % (corpus.len() - rows as usize);
        for at in 0..rows as usize {
            tokens[at] = corpus[start + at] as f32;
            labels[at] = corpus[start + at + 1] as f32;
        }
        runtime.write(&program, token, &tokens);
        runtime.write(&program, label, &labels);
        runtime.run(&program);
        let held = runtime.read(&program, loss)[0];
        println!(
            "step {:>4}/{}: loss {held:.4} over tokens {}..{}",
            step + 1,
            request.steps,
            start,
            start + rows as usize + 1,
        );
    }
    let seconds = started.elapsed().as_secs_f64();
    runtime.save_weights_named(&weights, &request.save, &|name| {
        name.starts_with("adapter.")
    });
    println!(
        "trained {} steps of {rows} tokens in {seconds:.2}s ({:.1} tokens/s)",
        request.steps,
        (f64::from(request.steps) * f64::from(rows)) / seconds,
    );
    println!(
        "saved {} persona tables of rank {} to {}",
        plan.count(),
        plan.rank(),
        request.save.display(),
    );
}
