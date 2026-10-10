use neura_abi::Element;
use neura_gpu::{Backend, PREFERENCE};
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{CheckpointFile, MemoryRequest, Program, Runtime, RuntimeRequest, Weights};
use std::path::{Path, PathBuf};

const WIDTH: u32 = 256;
const LAYERS: u32 = 4;
const BATCH: u32 = 16;
const STEPS: u32 = 3;
const SPILL_BYTES: u64 = 32 * (1 << 14);

fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("neura-spill-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("a spill directory");
    path
}

fn open(backend: Backend, memory: MemoryRequest) -> Runtime {
    Runtime::open(RuntimeRequest {
        gpu: neura_gpu::GpuRequest {
            backend: Some(backend),
            ..Default::default()
        },
        memory,
    })
    .unwrap_or_else(|error| panic!("no device runs the spill tests over {backend:?}: {error}"))
}

fn resident(backend: Backend) -> Runtime {
    open(
        backend,
        MemoryRequest {
            readback_bytes: 4 << 20,
            ..Default::default()
        },
    )
}

fn spilled(backend: Backend, directory: &Path) -> Runtime {
    spilled_with(backend, directory, SPILL_BYTES)
}

fn spilled_with(backend: Backend, directory: &Path, bytes: u64) -> Runtime {
    open(
        backend,
        MemoryRequest {
            readback_bytes: 4 << 20,
            resident_weight_bytes: Some(bytes),
            weight_spill: Some(directory.to_path_buf()),
            ..Default::default()
        },
    )
}

fn layers<'g>(graph: &Graph<'g>) -> Vec<(Value<'g>, Value<'g>)> {
    let init = Init::Uniform {
        low: -0.02,
        high: 0.02,
    };
    (0..LAYERS)
        .map(|layer| {
            (
                graph.named_parameter(
                    &format!("layer{layer}.weight"),
                    Shape::matrix(WIDTH, WIDTH),
                    init,
                    Element::Single,
                ),
                graph.named_parameter(
                    &format!("layer{layer}.bias"),
                    Shape::matrix(1, WIDTH),
                    Init::Zero,
                    Element::Single,
                ),
            )
        })
        .collect()
}

struct Session<'g> {
    layers: Vec<(Value<'g>, Value<'g>)>,
    data: Value<'g>,
    loss: Value<'g>,
}

fn training<'g>(graph: &Graph<'g>) -> Session<'g> {
    let layers = layers(graph);
    let data = graph.input(Shape::matrix(BATCH, WIDTH), Element::Single);
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
    Session { layers, data, loss }
}

fn numbers() -> Vec<f32> {
    (0..(BATCH * WIDTH))
        .map(|index| ((index % 13) as f32) * 0.01 - 0.06)
        .collect()
}

struct Trained {
    program: Program,
    loss: f32,
}

fn train(
    runtime: &Runtime,
    graph: &Graph<'_>,
    session: &Session<'_>,
    weights: &Weights,
) -> Trained {
    let program = runtime.compile(graph, weights);
    let input = numbers();
    let mut loss = 0.0;
    for _ in 0..STEPS {
        runtime.write(&program, session.data, &input);
        runtime.run(&program);
        loss = runtime.read(&program, session.loss)[0];
    }
    Trained { program, loss }
}

fn parameters(runtime: &Runtime, program: &Program, session: &Session<'_>) -> Vec<f32> {
    session
        .layers
        .iter()
        .flat_map(|(weight, bias)| {
            let mut values = runtime.read(program, *weight);
            values.extend(runtime.read(program, *bias));
            values
        })
        .collect()
}

fn keeps_its_pages_out_of_host_memory(backend: Backend, directory: &Path) {
    let resident_runtime = resident(backend);
    let resident_graph = Graph::new();
    let resident_session = training(&resident_graph);
    let resident_weights = resident_runtime.weights(&resident_graph);
    let kept = train(
        &resident_runtime,
        &resident_graph,
        &resident_session,
        &resident_weights,
    );
    let expected = parameters(&resident_runtime, &kept.program, &resident_session);

    let runtime = spilled(backend, directory);
    let graph = Graph::new();
    let session = training(&graph);
    let weights = runtime.weights(&graph);
    assert!(
        weights.pages() > weights.resident_pages(),
        "a store of {} pages keeps {} of them beside a budget of {SPILL_BYTES} bytes",
        weights.pages(),
        weights.resident_pages(),
    );
    assert_eq!(
        weights.host_bytes(),
        0,
        "a spilled store of {} pages holds {} bytes of page images in host memory",
        weights.pages(),
        weights.host_bytes(),
    );
    let spill = weights
        .spill_file()
        .expect("a spilled store names the file its pages live in");
    assert_eq!(spill.parent(), Some(directory));
    assert!(
        spill.exists(),
        "a spilled store of {} pages holds no file at {}",
        weights.pages(),
        spill.display(),
    );

    let trained = train(&runtime, &graph, &session, &weights);
    let observed = parameters(&runtime, &trained.program, &session);
    for (at, (expected, observed)) in expected.iter().zip(&observed).enumerate() {
        assert!(
            (expected - observed).abs() <= 1e-5,
            "a spilled store reached {observed} where the resident store reached {expected} at {at}",
        );
    }
    assert!(
        (kept.loss - trained.loss).abs() <= 1e-5,
        "a spilled store held a loss of {} where the resident store held {}",
        trained.loss,
        kept.loss,
    );
    assert!(
        weights.spill_read_bytes() > 0,
        "a spilled store of {} pages read no byte of its file",
        weights.pages(),
    );
    assert!(
        weights.spill_write_bytes() > 0,
        "a spilled store of {} pages wrote no byte of its file",
        weights.pages(),
    );
    assert!(
        weights.readback_pages() > 0,
        "a spilled store of {} pages read no page back from the device",
        weights.pages(),
    );

    drop(trained);
    drop(session);
    drop(weights);
    drop(graph);
    drop(runtime);
    assert!(
        !spill.exists(),
        "a spilled store left its file at {} behind",
        spill.display(),
    );
}

#[test]
fn a_spilled_store_keeps_its_pages_out_of_host_memory() {
    for &backend in PREFERENCE {
        keeps_its_pages_out_of_host_memory(backend, &directory("pages"));
    }
}

fn layer_names() -> Vec<String> {
    (0..LAYERS)
        .flat_map(|layer| [format!("layer{layer}.bias"), format!("layer{layer}.weight")])
        .collect()
}

fn carries_every_word_of_a_checkpoint(backend: Backend, directory: &Path) {
    let source = spilled(backend, directory);
    let source_graph = Graph::new();
    let source_session = training(&source_graph);
    let source_weights = source.weights(&source_graph);
    let trained = train(&source, &source_graph, &source_session, &source_weights);
    let expected = parameters(&source, &trained.program, &source_session);
    let checkpoint = source.checkpoint(&source_weights);

    let target = spilled(backend, directory);
    let target_graph = Graph::new();
    let target_session = training(&target_graph);
    let seeded = target.weights(&target_graph);
    let seeded_program = target.compile(&target_graph, &seeded);
    let before = parameters(&target, &seeded_program, &target_session);
    assert_ne!(
        before, expected,
        "a store that has not restored yet already holds the trained words",
    );
    target.restore(&seeded, &checkpoint);
    let restored = parameters(&target, &seeded_program, &target_session);
    for (at, (expected, observed)) in expected.iter().zip(&restored).enumerate() {
        assert!(
            (expected - observed).abs() <= 1e-5,
            "a restored checkpoint reached {observed} where the trained store reached {expected} at {at}",
        );
    }

    let file = directory.join("model.safetensors");
    std::fs::write(&file, checkpoint.bytes()).expect("a checkpoint file");
    let container = CheckpointFile::open(&file);
    assert_eq!(container.tensors(), 2 * LAYERS as usize);
    assert_eq!(container.names().collect::<Vec<_>>(), layer_names(),);
    let named = container
        .tensor("layer0.weight")
        .expect("a named tensor of a file container");
    assert_eq!(named.element, Some(Element::Single));
    assert_eq!(named.elements, u64::from(WIDTH) * u64::from(WIDTH));
    assert_eq!(named.payload_bytes, u64::from(WIDTH) * u64::from(WIDTH) * 4,);

    let loaded = spilled(backend, directory);
    let loaded_graph = Graph::new();
    let loaded_session = training(&loaded_graph);
    let weights = loaded.load_streamed(&loaded_graph, &container);
    let program = loaded.compile(&loaded_graph, &weights);
    let observed = parameters(&loaded, &program, &loaded_session);
    for (at, (expected, observed)) in expected.iter().zip(&observed).enumerate() {
        assert!(
            (expected - observed).abs() <= 1e-5,
            "a store loaded from a file reached {observed} where the trained store reached {expected} at {at}",
        );
    }
    assert!(
        weights.spill_read_bytes() > 0 && weights.host_bytes() == 0,
        "a store loaded from a file spilled no byte",
    );
    assert!(
        container.reads() > container.tensors() as u64,
        "a container of {} tensors read its payload in {} pieces and never streams its weights",
        container.tensors(),
        container.reads(),
    );

    let resting = spilled(backend, directory);
    let resting_graph = Graph::new();
    let resting_layers = layers(&resting_graph);
    let resting_data = resting_graph.input(Shape::matrix(BATCH, WIDTH), Element::Single);
    let resting_out = resting_graph.matmul(resting_data, resting_layers[0].0);
    resting_graph.retain(resting_out);
    let resting_weights = resting.load_streamed(&resting_graph, &container);
    let resting_program = resting.compile(&resting_graph, &resting_weights);
    let observed = resting.read(&resting_program, resting_layers[0].0).len();
    assert_eq!(
        observed as u64,
        u64::from(WIDTH) * u64::from(WIDTH),
        "a resting graph loaded from a file holds the weight its file names",
    );

    let overwritten = vec![0.0; (WIDTH * WIDTH) as usize];
    target.write(&seeded_program, target_session.layers[0].0, &overwritten);
    target.restore_streamed(&seeded, &container);
    let restored = parameters(&target, &seeded_program, &target_session);
    for (at, (expected, observed)) in expected.iter().zip(&restored).enumerate() {
        assert!(
            (expected - observed).abs() <= 1e-5,
            "a streamed restore reached {observed} where the trained store reached {expected} at {at}",
        );
    }
}

#[test]
fn a_spilled_store_carries_every_word_of_a_checkpoint() {
    for &backend in PREFERENCE {
        carries_every_word_of_a_checkpoint(backend, &directory("checkpoint"));
    }
}

fn saves_the_container_it_streams(backend: Backend, directory: &Path) {
    let source = spilled(backend, directory);
    let source_graph = Graph::new();
    let source_session = training(&source_graph);
    let source_weights = source.weights(&source_graph);
    let trained = train(&source, &source_graph, &source_session, &source_weights);
    let expected = parameters(&source, &trained.program, &source_session);
    let file = directory.join("streamed.safetensors");
    let container = source.save(&source_weights, &file);
    assert_eq!(
        std::fs::read(&file).expect("a container this store wrote"),
        source.checkpoint(&source_weights).bytes(),
        "a store streams the container it packs",
    );
    assert!(
        source_weights.read_peak_bytes() <= 1 << 16,
        "a store of {} bytes read itself out in a buffer of {} bytes",
        source_weights.bytes(),
        source_weights.read_peak_bytes(),
    );
    assert!(
        source_weights.host_bytes() == 0,
        "a store of {} bytes held {} bytes of its pages in host memory",
        source_weights.bytes(),
        source_weights.host_bytes(),
    );
    assert_eq!(container.tensors(), 2 * LAYERS as usize);
    assert_eq!(container.names().collect::<Vec<_>>(), layer_names());

    let target = resident(backend);
    let target_graph = Graph::new();
    let target_session = training(&target_graph);
    let target_weights = target.load_streamed(&target_graph, &container);
    let program = target.compile(&target_graph, &target_weights);
    let observed = parameters(&target, &program, &target_session);
    assert_eq!(expected.len(), observed.len());
    for (at, (expected, observed)) in expected.iter().zip(&observed).enumerate() {
        assert!(
            (expected - observed).abs() <= 1e-5,
            "a resident store loaded from a file reached {observed} where the spilled store reached {expected} at {at}",
        );
    }
    let resting = directory.join("resting.safetensors");
    target.save(&target_weights, &resting);
    assert_eq!(
        std::fs::read(&resting).expect("a container the resident store wrote"),
        std::fs::read(&file).expect("a container the spilled store wrote"),
        "a resident store writes the container a spilled store writes",
    );
}

#[test]
fn a_spilled_store_saves_the_container_it_streams() {
    for &backend in PREFERENCE {
        saves_the_container_it_streams(backend, &directory("save"));
    }
}

fn refuses_a_directory_it_cannot_spill_into(backend: Backend, directory: &Path) {
    let missing = directory.join("missing");
    let runtime = open(
        backend,
        MemoryRequest {
            resident_weight_bytes: Some(SPILL_BYTES),
            weight_spill: Some(missing),
            ..Default::default()
        },
    );
    let graph = Graph::new();
    let _ = layers(&graph);
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = runtime.weights(&graph);
    }));
    assert!(
        refused.is_err(),
        "a store spilled into a directory that does not stand was accepted",
    );
}

#[test]
fn a_spilled_store_refuses_a_directory_it_cannot_spill_into() {
    for &backend in PREFERENCE {
        refuses_a_directory_it_cannot_spill_into(backend, &directory("missing"));
    }
}

fn a_resident_store_holds_every_page_on_the_device(backend: Backend) {
    let runtime = resident(backend);
    let graph = Graph::new();
    let session = training(&graph);
    let weights = runtime.weights(&graph);
    let trained = train(&runtime, &graph, &session, &weights);
    assert_eq!(weights.host_bytes(), 0);
    assert_eq!(weights.spill_file(), None);
    assert_eq!(weights.spill_read_bytes(), 0);
    assert_eq!(weights.spill_write_bytes(), 0);
    let observed = parameters(&runtime, &trained.program, &session);
    assert!(
        observed.iter().all(|value| value.is_finite()),
        "a resident store ran a step of a spilled model",
    );
}

#[test]
fn a_resident_store_streams_nothing() {
    for &backend in PREFERENCE {
        a_resident_store_holds_every_page_on_the_device(backend);
    }
}

fn carries_every_quantum_of_a_quantized_model(backend: Backend, directory: &Path) {
    let resident_runtime = resident(backend);
    let resident_graph = Graph::new();
    let (resident_first, resident_out) = quantized(&resident_graph);
    let resident_weights = resident_runtime.weights(&resident_graph);
    let resident_program = resident_runtime.compile(&resident_graph, &resident_weights);

    let runtime = spilled_with(backend, directory, 3 * (1 << 14));
    let graph = Graph::new();
    let (first, out) = quantized(&graph);
    let weights = runtime.weights(&graph);
    assert!(
        weights.pages() > weights.resident_pages(),
        "two int4 weights of {} pages keep {} of them beside a budget of 3 pages",
        weights.pages(),
        weights.resident_pages(),
    );
    assert!(
        resident_weights.pages() == resident_weights.resident_pages(),
        "a resident quantized store of {} pages keeps {} of them",
        resident_weights.pages(),
        resident_weights.resident_pages(),
    );
    let program = runtime.compile(&graph, &weights);

    let written = (0..(1024 * 256))
        .map(|index| ((index % 17) as f32 - 8.0) * 0.05)
        .collect::<Vec<_>>();
    let (streamed_out, streamed_read) = run_quantized(&runtime, &program, first, out, &written);
    let (expected_out, expected_read) = run_quantized(
        &resident_runtime,
        &resident_program,
        resident_first,
        resident_out,
        &written,
    );
    for (at, (expected, observed)) in expected_read.iter().zip(&streamed_read).enumerate() {
        assert!(
            (expected - observed).abs() <= 1e-5,
            "a spilled quantized weight came back as {observed} where the resident store reached {expected} at {at}",
        );
    }
    for (at, (expected, observed)) in expected_out.iter().zip(&streamed_out).enumerate() {
        assert!(
            (expected - observed).abs() <= 1e-5,
            "a spilled quantized product reached {observed} where the resident store reached {expected} at {at}",
        );
    }
    assert!(
        weights.spill_read_bytes() > 0 && weights.host_bytes() == 0,
        "a spilled block quantized store of {} pages held its words in host memory",
        weights.pages(),
    );
}

fn run_quantized(
    runtime: &Runtime,
    program: &Program,
    first: Value<'_>,
    out: Value<'_>,
    written: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    runtime.write(program, first, written);
    let read = runtime.read(program, first);
    assert_eq!(read.len(), written.len());
    for (at, (written, read)) in written.iter().zip(&read).enumerate() {
        assert!(
            (written - read).abs() <= 0.06,
            "a quantized weight came back as {read} where {written} was stored at {at}",
        );
    }
    (runtime.read(program, out), read)
}

fn quantized<'g>(graph: &Graph<'g>) -> (Value<'g>, Value<'g>) {
    let first = graph.block_quantized_parameter(
        Shape::matrix(1024, 256),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Int4,
    );
    let second = graph.block_quantized_parameter(
        Shape::matrix(1024, 256),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Int4,
    );
    let data = graph.input(Shape::matrix(8, 1024), Element::Single);
    let out = graph.add(graph.matmul(data, first), graph.matmul(data, second));
    graph.retain(out);
    (first, out)
}

#[test]
fn a_container_rides_to_another_thread_of_the_frame() {
    fn assert_static<T: Send + Sync + 'static>() {}
    assert_static::<neura_runtime::Checkpoint>();
    assert_static::<CheckpointFile>();
}

#[test]
fn a_spilled_store_carries_every_quantum_of_a_quantized_model() {
    for &backend in PREFERENCE {
        carries_every_quantum_of_a_quantized_model(backend, &directory("quantized"));
    }
}

#[test]
fn a_spill_directory_sweeps_the_leftovers_of_a_crashed_run() {
    let directory = directory("sweep");
    let pid = std::process::id();
    let stale = (0..40u64)
        .map(|serial| directory.join(format!("neura-weights-{pid}-{}.spill", 1_000_000 + serial)))
        .chain((0..8u64).map(|serial| {
            directory.join(format!(
                "neura-weights-{}-{}.spill",
                pid + 1,
                1_000_000 + serial,
            ))
        }))
        .collect::<Vec<_>>();
    for path in &stale {
        std::fs::write(path, b"the pages of a run that is gone").expect("a leftover spill");
    }
    let held = directory.join(format!("neura-weights-{pid}-9001.spill"));
    let lock = std::fs::File::create(&held).expect("the spill of a live run");
    lock.try_lock().expect("a live run holds its spill");
    let foreign = directory.join("engine-cache.bin");
    std::fs::write(&foreign, b"a file that belongs to the engine").expect("a foreign file");

    let runtime = spilled_with(PREFERENCE[0], &directory, SPILL_BYTES);
    let graph = Graph::new();
    let session = training(&graph);
    let weights = runtime.weights(&graph);

    for path in &stale {
        assert!(
            !path.exists(),
            "a sweep left the spill at {} behind",
            path.display(),
        );
    }
    assert!(
        held.exists(),
        "a sweep removed the spill a live run holds at {}",
        held.display(),
    );
    assert!(
        foreign.exists(),
        "a sweep removed the file at {}",
        foreign.display(),
    );
    let file = weights
        .spill_file()
        .expect("a spilled store names the file its pages live in");
    assert!(
        file.exists(),
        "a swept directory created no spill at {}",
        file.display(),
    );
    let trained = train(&runtime, &graph, &session, &weights);
    assert!(
        trained.loss.is_finite(),
        "a spilled store that swept its directory ran a step of a spilled model",
    );
    drop(trained);
    drop(session);
    drop(weights);
    assert!(
        !file.exists(),
        "a spilled store left its file at {} behind",
        file.display(),
    );
    drop(lock);
    let _ = std::fs::remove_file(&held);
}

#[test]
fn a_sweep_keeps_the_spill_a_live_run_holds() {
    let directory = directory("live");
    let first = spilled_with(PREFERENCE[0], &directory, SPILL_BYTES);
    let first_graph = Graph::new();
    let first_session = training(&first_graph);
    let first_weights = first.weights(&first_graph);
    let file = first_weights
        .spill_file()
        .expect("a spilled store names the file its pages live in");
    assert!(
        file.exists(),
        "a spilled store holds no file at {}",
        file.display(),
    );

    let second = spilled_with(PREFERENCE[0], &directory, SPILL_BYTES);
    let second_graph = Graph::new();
    let second_session = training(&second_graph);
    let second_weights = second.weights(&second_graph);
    assert!(
        file.exists(),
        "a sweep removed the spill at {} a live run holds",
        file.display(),
    );
    let trained = train(&first, &first_graph, &first_session, &first_weights);
    assert!(
        trained.loss.is_finite(),
        "the sweep of another run broke the model of a live spill",
    );
    assert!(
        first_weights.spill_read_bytes() > 0,
        "a live spill read no byte of the file it holds",
    );
    let other = train(&second, &second_graph, &second_session, &second_weights);
    assert!(
        other.loss.is_finite(),
        "the run that swept a directory of live spills ran no model",
    );
    assert!(
        second_weights
            .spill_file()
            .is_some_and(|path| path.exists()),
        "the run that swept a directory of live spills holds no file",
    );
}
