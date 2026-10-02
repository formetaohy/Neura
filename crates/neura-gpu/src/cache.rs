use std::fs;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(any(vulkan_backend, dx12_backend))]
const FNV_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
#[cfg(any(vulkan_backend, dx12_backend))]
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
#[cfg(any(vulkan_backend, dx12_backend))]
const SEPARATOR: u8 = 0xff;

static PARTIAL: AtomicU64 = AtomicU64::new(1);

struct State {
    root: PathBuf,
    loads: AtomicU64,
    stores: AtomicU64,
}

#[derive(Clone)]
pub struct PipelineCache {
    state: Arc<State>,
}

impl PipelineCache {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        fs::create_dir_all(&root).unwrap_or_else(|error| {
            panic!(
                "creating the device pipeline cache {}: {error}",
                root.display(),
            )
        });
        Self {
            state: Arc::new(State {
                root,
                loads: AtomicU64::new(0),
                stores: AtomicU64::new(0),
            }),
        }
    }

    pub fn root(&self) -> &Path {
        &self.state.root
    }

    pub fn loads(&self) -> u64 {
        self.state.loads.load(Ordering::Relaxed)
    }

    pub fn stores(&self) -> u64 {
        self.state.stores.load(Ordering::Relaxed)
    }

    pub fn load(&self, key: &str) -> Option<Vec<u8>> {
        let path = self.file(key);
        let payload = match fs::read(&path) {
            Ok(payload) => payload,
            Err(error) if error.kind() == ErrorKind::NotFound => return None,
            Err(error) => panic!("reading the device artifact {}: {error}", path.display()),
        };
        assert!(
            !payload.is_empty(),
            "the device artifact {} holds no byte",
            path.display(),
        );
        self.state.loads.fetch_add(1, Ordering::Relaxed);
        Some(payload)
    }

    pub fn store(&self, key: &str, payload: &[u8]) {
        assert!(
            !payload.is_empty(),
            "a device artifact of {key} holds no byte",
        );
        let path = self.file(key);
        let partial = path.with_extension(format!(
            "partial-{}-{}",
            std::process::id(),
            PARTIAL.fetch_add(1, Ordering::Relaxed),
        ));
        fs::write(&partial, payload).unwrap_or_else(|error| {
            panic!("writing the device artifact {}: {error}", partial.display())
        });
        fs::rename(&partial, &path).unwrap_or_else(|error| {
            panic!("publishing the device artifact {}: {error}", path.display())
        });
        self.state.stores.fetch_add(1, Ordering::Relaxed);
    }

    pub fn file(&self, key: &str) -> PathBuf {
        let key = Path::new(key);
        assert!(
            !key.as_os_str().is_empty()
                && key
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "a device artifact key {} is not a plain relative path",
            key.display(),
        );
        let path = self.state.root.join(key);
        fs::create_dir_all(path.parent().expect("an artifact key names a file"))
            .unwrap_or_else(|error| panic!("creating the device artifact directory: {error}"));
        path
    }
}

#[cfg(any(vulkan_backend, dx12_backend))]
pub(crate) fn fingerprint(parts: &[&[u8]]) -> String {
    let mut state = FNV_BASIS;
    for part in parts {
        for byte in part.iter().copied().chain([SEPARATOR]) {
            state ^= u128::from(byte);
            state = state.wrapping_mul(FNV_PRIME);
        }
    }
    format!("{state:032x}")
}
