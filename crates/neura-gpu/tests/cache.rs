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

fn aged(path: &Path, seconds: u64) {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(path)
        .expect("a test artifact is writable");
    file.set_modified(std::time::SystemTime::now() - std::time::Duration::from_secs(seconds))
        .expect("a test artifact takes a modification time");
}

#[test]
fn a_bounded_cache_leaves_out_the_artifact_it_used_least_recently() {
    let directory = directory("bounded");
    let cache = ArtifactCache::bounded(&directory, 64);
    cache.store("dx12/a.dxil", &[1u8; 24]);
    cache.store("dx12/b.dxil", &[2u8; 24]);
    assert_eq!(cache.bytes(), 48);
    assert_eq!(cache.evictions(), 0, "a cache within its bound holds both");
    assert_eq!(cache.load("dx12/a.dxil").as_deref(), Some(&[1u8; 24][..]));
    cache.store("dx12/c.dxil", &[3u8; 24]);
    assert_eq!(
        cache.bytes(),
        48,
        "a cache over its bound reclaims what it holds"
    );
    assert_eq!(cache.evictions(), 1);
    assert!(
        cache.load("dx12/a.dxil").is_some(),
        "the artifact a reader touched stays",
    );
    assert!(
        !cache.holds("dx12/b.dxil"),
        "the artifact no reader touched leaves",
    );
    assert!(cache.load("dx12/c.dxil").is_some());
    assert!(!directory.join("dx12/b.dxil").exists());
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}

#[test]
fn a_bounded_cache_never_holds_more_than_its_budget() {
    let directory = directory("ceiling");
    let cache = ArtifactCache::bounded(&directory, 100);
    for index in 0..40u8 {
        cache.store(&format!("dx12/{index}.dxil"), &[index; 24]);
        assert!(
            cache.bytes() <= 100,
            "a cache of a 100 byte budget holds {} bytes after {} writes",
            cache.bytes(),
            index + 1,
        );
    }
    assert!(cache.evictions() > 0);
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}

#[test]
fn a_bounded_cache_adopts_the_artifacts_it_already_holds() {
    let directory = directory("adopt");
    let stale = directory.join("vulkan").join("a-device-that-left");
    let warm = directory.join("vulkan").join("the-device-at-hand");
    std::fs::create_dir_all(&stale).expect("a test makes an artifact directory");
    std::fs::create_dir_all(&warm).expect("a test makes an artifact directory");
    std::fs::write(stale.join("one.bin"), [1u8; 32]).expect("a test writes an artifact");
    std::fs::write(warm.join("two.bin"), [2u8; 32]).expect("a test writes an artifact");
    aged(&stale.join("one.bin"), 3600);
    let cache = ArtifactCache::bounded(&directory, 40);
    assert_eq!(cache.bytes(), 32, "a cache bounds what it adopted");
    assert!(
        cache.holds("vulkan/the-device-at-hand/two.bin"),
        "the artifact written most recently stays",
    );
    assert!(
        !cache.holds("vulkan/a-device-that-left/one.bin"),
        "the artifact a device no longer writes leaves",
    );
    assert!(
        !stale.exists(),
        "the namespace of a device that left leaves with its artifact",
    );
    assert!(cache.evictions() >= 1);
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}

#[test]
fn a_bounded_cache_refuses_an_artifact_larger_than_its_budget() {
    let directory = directory("refused");
    let cache = ArtifactCache::bounded(&directory, 16);
    cache.store("dx12/big.dxil", &[7u8; 32]);
    assert_eq!(cache.refused(), 1);
    assert_eq!(
        cache.stores(),
        0,
        "a refused artifact never reaches the disk"
    );
    assert_eq!(cache.bytes(), 0);
    assert!(!cache.holds("dx12/big.dxil"));
    assert!(cache.load("dx12/big.dxil").is_none());
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}

#[test]
fn a_bounded_cache_counts_the_artifacts_its_writers_record() {
    let directory = directory("recorded");
    let cache = ArtifactCache::bounded(&directory, 40);
    let written = cache
        .file("dx12/written.dxil")
        .expect("a named cache names a file");
    std::fs::write(&written, [3u8; 32]).expect("a test writes an artifact");
    cache.record("dx12/written.dxil");
    assert_eq!(
        cache.bytes(),
        32,
        "a recorded artifact counts toward the budget"
    );
    assert!(cache.holds("dx12/written.dxil"));
    cache.store("dx12/other.dxil", &[4u8; 24]);
    assert!(cache.bytes() <= 40);
    assert_eq!(cache.evictions(), 1, "a recorded artifact can be reclaimed");
    let large = cache
        .file("metal/large.bin")
        .expect("a named cache names a file");
    std::fs::write(&large, [5u8; 64]).expect("a test writes an artifact");
    cache.record("metal/large.bin");
    assert_eq!(cache.refused(), 1);
    assert!(
        !cache.holds("metal/large.bin"),
        "an artifact larger than the budget leaves",
    );
    assert_eq!(cache.bytes(), 24);
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}

#[test]
#[should_panic(expected = "holds no byte")]
fn a_recorded_artifact_of_no_byte_is_refused() {
    let directory = directory("recorded-empty");
    let cache = ArtifactCache::bounded(&directory, 1 << 20);
    let empty = cache
        .file("dx12/empty.dxil")
        .expect("a named cache names a file");
    std::fs::write(&empty, []).expect("a test writes an empty artifact");
    cache.record("dx12/empty.dxil");
}

#[test]
fn a_bounded_cache_sweeps_the_leftovers_of_a_crashed_writer() {
    let directory = directory("leftovers");
    let artifacts = directory.join("dx12");
    std::fs::create_dir_all(&artifacts).expect("a test makes an artifact directory");
    let abandoned = artifacts.join("crashed.partial-9999-1");
    let writing = artifacts.join("live.partial-9999-2");
    let empty = artifacts.join("empty.dxil");
    std::fs::write(&abandoned, [0u8; 8]).expect("a test writes a partial artifact");
    std::fs::write(&writing, [0u8; 8]).expect("a test writes a partial artifact");
    std::fs::write(&empty, []).expect("a test writes an empty artifact");
    aged(&abandoned, 3600);
    let cache = ArtifactCache::bounded(&directory, 1 << 20);
    assert!(!abandoned.exists(), "a partial left by a crash leaves");
    assert!(writing.exists(), "a partial another writer holds stays");
    assert!(!empty.exists(), "an artifact of no byte leaves");
    assert_eq!(cache.bytes(), 0);
    assert_eq!(cache.evictions(), 2);
    std::fs::remove_dir_all(&directory).expect("a test cache directory is removable");
}
