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
pub struct ArtifactCache {
    state: Option<Arc<State>>,
}

impl ArtifactCache {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        fs::create_dir_all(&root).unwrap_or_else(|error| {
            panic!(
                "creating the device artifact cache {}: {error}",
                root.display(),
            )
        });
        Self::of(root)
    }

    pub fn default_location() -> Self {
        for root in candidate_roots() {
            if fs::create_dir_all(&root).is_ok() {
                return Self::of(root);
            }
        }
        Self { state: None }
    }

    fn of(root: PathBuf) -> Self {
        Self {
            state: Some(Arc::new(State {
                root,
                loads: AtomicU64::new(0),
                stores: AtomicU64::new(0),
            })),
        }
    }

    pub fn root(&self) -> Option<&Path> {
        self.state.as_deref().map(|state| state.root.as_path())
    }

    pub fn loads(&self) -> u64 {
        self.state
            .as_deref()
            .map_or(0, |state| state.loads.load(Ordering::Relaxed))
    }

    pub fn stores(&self) -> u64 {
        self.state
            .as_deref()
            .map_or(0, |state| state.stores.load(Ordering::Relaxed))
    }

    pub fn load(&self, key: &str) -> Option<Vec<u8>> {
        let state = self.state.as_deref()?;
        let path = self.file(key)?;
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
        state.loads.fetch_add(1, Ordering::Relaxed);
        Some(payload)
    }

    pub fn store(&self, key: &str, payload: &[u8]) {
        assert!(
            !payload.is_empty(),
            "a device artifact of {key} holds no byte",
        );
        let Some(state) = self.state.as_deref() else {
            return;
        };
        let path = self
            .file(key)
            .expect("a live artifact cache names a file for every key");
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
        state.stores.fetch_add(1, Ordering::Relaxed);
    }

    pub fn file(&self, key: &str) -> Option<PathBuf> {
        let state = self.state.as_deref()?;
        let key = Path::new(key);
        assert!(
            !key.as_os_str().is_empty()
                && key
                    .components()
                    .all(|component| matches!(component, Component::Normal(_))),
            "a device artifact key {} is not a plain relative path",
            key.display(),
        );
        let path = state.root.join(key);
        fs::create_dir_all(path.parent().expect("an artifact key names a file"))
            .unwrap_or_else(|error| panic!("creating the device artifact directory: {error}"));
        Some(path)
    }
}

fn candidate_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(cache) = std::env::var_os("XDG_CACHE_HOME") {
        roots.push(PathBuf::from(cache).join("neura"));
    }
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        roots.push(PathBuf::from(local).join("neura").join("artifacts"));
    }
    if let Some(home) = std::env::var_os("HOME") {
        let home = PathBuf::from(home);
        roots.push(home.join(".cache").join("neura"));
        roots.push(home.join("Library").join("Caches").join("neura"));
    }
    if let Some(temp) = std::env::var_os("TMPDIR") {
        roots.push(PathBuf::from(temp).join("neura-artifacts"));
    }
    roots.push(std::env::temp_dir().join("neura-artifacts"));
    roots.dedup();
    roots
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
