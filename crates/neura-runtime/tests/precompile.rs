use neura_abi::Element;
use neura_gpu::Backend;
use neura_graph::{Graph, Init, Shape, Value};
use neura_runtime::{GpuRequest, MemoryRequest, Runtime, RuntimeRequest};
use std::path::{Path, PathBuf};

#[path = "support/reference.rs"]
mod reference;

use reference::{matmul_reference, random};

fn directory() -> PathBuf {
    directory_named("precompile")
}

fn directory_named(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("neura-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn open(directory: &Path) -> Runtime {
    open_bounded(directory, GpuRequest::default().artifact_bytes)
}

fn open_bounded(directory: &Path, artifact_bytes: u64) -> Runtime {
    Runtime::open(RuntimeRequest {
        gpu: GpuRequest {
            artifacts: Some(directory.to_path_buf()),
            artifact_bytes,
            ..Default::default()
        },
        memory: MemoryRequest {
            readback_bytes: 1 << 16,
            ..Default::default()
        },
    })
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

struct Model {
    graph: Graph<'static>,
    input: Value<'static>,
    weight: Value<'static>,
    bias: Value<'static>,
    out: Value<'static>,
}

fn model(samples: u32) -> Model {
    let graph: Graph<'static> = Graph::new();
    let input = graph.input(Shape::matrix(samples, 8), Element::Single);
    let weight = graph.parameter(Shape::matrix(8, 4), Init::Zero, Element::Single);
    let bias = graph.parameter(Shape::vector(4), Init::Zero, Element::Single);
    let out = graph.relu(graph.add(graph.matmul(input, weight), bias));
    graph.retain(out);
    Model {
        graph,
        input,
        weight,
        bias,
        out,
    }
}

fn run(runtime: &Runtime, model: &Model, samples: u32) {
    let weights = runtime.weights(&model.graph);
    let program = runtime.compile(&model.graph, &weights);
    let input = random(samples * 8, samples + 7);
    let weight = random(8 * 4, 11);
    let bias = random(4, 13);
    runtime.write(&program, model.input, &input);
    runtime.write(&program, model.weight, &weight);
    runtime.write(&program, model.bias, &bias);
    runtime.run(&program);
    let mut expected = matmul_reference(&input, &weight, samples, 8, 4);
    for (index, value) in expected.iter_mut().enumerate() {
        *value = (*value + bias[index % 4]).max(0.0);
    }
    let produced = runtime.read(&program, model.out);
    assert_eq!(produced.len(), expected.len());
    for (index, (actual, wanted)) in produced.iter().zip(&expected).enumerate() {
        assert!(
            (actual - wanted).abs() <= 1e-5,
            "element {index} came back as {actual} where {wanted} was expected",
        );
    }
}

#[test]
fn a_precompiled_kernel_serves_every_shape_of_its_model() {
    let directory = directory();
    let narrow = model(4);
    {
        let runtime = open(&directory);
        let cache = runtime.context().artifact_cache();
        runtime.precompile(&narrow.graph, runtime.default_profile());
        assert!(
            cache.stores() >= 1,
            "a precompiled kernel writes its artifact",
        );
    }
    let stored = {
        let runtime = open(&directory);
        let cache = runtime.context().artifact_cache();
        assert_eq!(cache.loads(), 0, "a fresh runtime holds no artifact");
        run(&runtime, &narrow, 4);
        assert!(
            cache.loads() >= 1,
            "a precompiled kernel loads its artifact",
        );
        cache.stores()
    };
    let wide = model(16);
    let runtime = open(&directory);
    let cache = runtime.context().artifact_cache();
    run(&runtime, &wide, 16);
    assert!(
        cache.loads() >= 1,
        "a kernel of another shape loads the artifact of its model",
    );
    if runtime.context().adapter_info().backend == Backend::Dx12 {
        assert_eq!(
            cache.stores(),
            stored,
            "a precompiled kernel is not compiled a second time",
        );
    }
    std::fs::remove_dir_all(&directory).expect("a test artifact directory is removable");
}

#[test]
fn a_bounded_artifact_cache_recompiles_the_programs_it_evicted() {
    let directory = directory_named("precompile-bounded");
    let narrow = model(4);
    let wide = model(16);
    let budget = {
        let runtime = open(&directory);
        run(&runtime, &narrow, 4);
        run(&runtime, &wide, 16);
        let held = runtime.context().artifact_cache().bytes();
        assert!(held > 0, "a compiled kernel writes an artifact");
        held * 2 / 3
    };
    let runtime = open_bounded(&directory, budget);
    let cache = runtime.context().artifact_cache();
    assert!(
        cache.evictions() >= 1,
        "a cache smaller than what its directory holds reclaims room",
    );
    for samples in [4u32, 16, 4, 16] {
        let model = if samples == 4 { &narrow } else { &wide };
        run(&runtime, model, samples);
        assert!(
            cache.bytes() <= budget,
            "a cache of a {budget} byte budget holds {} bytes",
            cache.bytes(),
        );
    }
    std::fs::remove_dir_all(&directory).expect("a test artifact directory is removable");
}
