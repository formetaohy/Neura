use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_runtime::{GpuRequest, Runtime, RuntimeRequest};
use std::path::{Path, PathBuf};

fn directory() -> PathBuf {
    let path = std::env::temp_dir().join(format!("neura-runtime-cache-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn step(directory: &Path) -> (Vec<f32>, u64, u64) {
    let runtime = pollster::block_on(Runtime::open(RuntimeRequest {
        gpu: GpuRequest {
            pipeline_cache: Some(directory.to_path_buf()),
            ..Default::default()
        },
        readback_bytes: 1 << 16,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"));
    let graph = Graph::new();
    let input = graph.input(Shape::matrix(4, 8), Element::Single);
    let weight = graph.parameter(
        Shape::matrix(8, 4),
        Init::Uniform {
            low: -0.5,
            high: 0.5,
        },
        Element::Single,
    );
    let shifted = graph.add(
        graph.matmul(input, weight),
        graph.fill(Shape::matrix(4, 4), 0.25),
    );
    let out = graph.relu(shifted);
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    let data = (0..32)
        .map(|value| value as f32 * 0.125 - 2.0)
        .collect::<Vec<_>>();
    runtime.write(&program, input, &data);
    runtime.run(&program);
    let cache = runtime
        .context()
        .pipeline_cache()
        .expect("a runtime that names a cache carries it");
    let counts = (cache.loads(), cache.stores());
    let values = runtime.read(&program, out);
    (values, counts.0, counts.1)
}

#[test]
fn a_megakernel_reloads_the_shader_it_compiled() {
    let directory = directory();
    let (first, loads, stores) = step(&directory);
    assert_eq!(loads, 0, "a cold cache holds no artifact");
    assert!(stores >= 1, "a cold compile writes an artifact");
    let (second, loads, _) = step(&directory);
    assert_eq!(second, first, "a cache serves the very shader it holds");
    assert!(loads >= 1, "a warm cache serves its artifact");
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}
