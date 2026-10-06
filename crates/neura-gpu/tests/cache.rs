use neura_compiler::{Read, ReadWrite, kernel};
use neura_gpu::{
    ArtifactCache, Backend, Backends, Binding, BufferUsages, GpuBuffer, GpuContext, GpuRequest,
    Submission,
};
use std::path::{Path, PathBuf};

#[kernel(workgroup_size = 64)]
fn scale(lid: u32, input: Read<u32>, output: ReadWrite<u32>) {
    output[lid] = input[lid] * 3u32;
}

fn directory(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("neura-cache-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    path
}

fn round_trip(backends: Backends, directory: &Path) -> (Vec<u32>, u64, u64) {
    let backend = backends.backend();
    let context = GpuContext::open(&GpuRequest {
        backends,
        artifacts: Some(directory.to_path_buf()),
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("{backend:?} could not run native compute: {error}"));
    assert_eq!(context.adapter_info().backend, backend);
    let device = context.device().clone();
    let queue = context.queue().clone();
    let input = GpuBuffer::new(
        &device,
        "cached input",
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let output = GpuBuffer::new(
        &device,
        "cached output",
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let readback = GpuBuffer::new(
        &device,
        "cached readback",
        256,
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );
    let pipeline = context.declare(scale());
    let group = pipeline.bind_group(&[
        Binding {
            index: 0,
            buffer: input.binding(0, 256),
        },
        Binding {
            index: 1,
            buffer: output.binding(0, 256),
        },
    ]);
    input.write_at(
        &queue,
        0,
        bytemuck::cast_slice(&(1..=64u32).collect::<Vec<_>>()),
    );
    let mut submission = Submission::new(&device, "native cache");
    submission.dispatch(&pipeline, &group, [1, 1, 1]);
    submission.submit(&queue);
    let mut transfer = Submission::new(&device, "native cache transfer");
    transfer.copy(&output, 0, &readback, 0, 256);
    let index = transfer.submit(&queue);
    let cache = context.artifact_cache();
    let counts = (cache.loads(), cache.stores());
    let values = bytemuck::cast_slice::<u8, u32>(&readback.read(&queue, index, 256)).to_vec();
    (values, counts.0, counts.1)
}

fn cached(backends: Backends) {
    let backend = backends.backend();
    let directory = directory(&format!("{backend:?}"));
    let (first, loads, stores) = round_trip(backends, &directory);
    assert_eq!(
        first,
        (1..=64u32).map(|value| value * 3).collect::<Vec<_>>()
    );
    assert_eq!(loads, 0, "a cold cache holds no artifact");
    assert_eq!(stores, 1, "a cold compile writes one artifact");
    let (second, warm_loads, warm_stores) = round_trip(backends, &directory);
    assert_eq!(second, first, "a cache serves the very pipeline it holds");
    assert!(warm_loads > loads, "a warm cache serves its artifact");
    if backend == Backend::Dx12 {
        assert_eq!(
            warm_stores, stores,
            "a cached DXIL feeds the pipeline without DXC",
        );
    }
    if backend == Backend::Vulkan {
        assert_eq!(
            warm_stores,
            stores + 1,
            "a Vulkan pipeline cache persists every compile",
        );
    }
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}

#[test]
fn an_artifact_cache_publishes_its_artifacts() {
    let directory = directory("host");
    let cache = ArtifactCache::at(&directory);
    assert!(cache.load("dx12/one.dxil").is_none());
    cache.store("dx12/one.dxil", b"first");
    assert_eq!(cache.load("dx12/one.dxil").as_deref(), Some(&b"first"[..]));
    cache.store("dx12/one.dxil", b"second");
    assert_eq!(cache.load("dx12/one.dxil").as_deref(), Some(&b"second"[..]));
    assert_eq!(cache.stores(), 2);
    assert_eq!(cache.loads(), 2);
    let leftovers = std::fs::read_dir(cache.root().expect("a named cache owns a root"))
        .expect("a cache directory is readable")
        .filter(|entry| {
            entry
                .as_ref()
                .expect("a cache entry is readable")
                .file_name()
                .to_string_lossy()
                .contains("partial")
        })
        .count();
    assert_eq!(leftovers, 0, "a published artifact leaves no partial file");
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}

#[test]
#[should_panic(expected = "holds no byte")]
fn an_empty_artifact_is_refused() {
    let directory = directory("empty");
    let cache = ArtifactCache::at(&directory);
    std::fs::write(
        cache
            .file("dx12/empty.dxil")
            .expect("a named cache names a file"),
        [],
    )
    .expect("a test writes an empty artifact");
    cache.load("dx12/empty.dxil");
}

#[test]
fn every_platform_backend_reuses_its_compiled_pipeline() {
    for backends in Backends::PLATFORM {
        cached(backends);
    }
}
