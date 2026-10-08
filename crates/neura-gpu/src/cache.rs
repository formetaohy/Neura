use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{ErrorKind, Read};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

pub const DEFAULT_ARTIFACT_BYTES: u64 = 256 << 20;
const PARTIAL_GRACE: Duration = Duration::from_secs(600);

#[cfg(any(vulkan_backend, dx12_backend, metal_backend))]
const FNV_BASIS: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
#[cfg(any(vulkan_backend, dx12_backend, metal_backend))]
const FNV_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
#[cfg(any(vulkan_backend, dx12_backend, metal_backend))]
const SEPARATOR: u8 = 0xff;

static PARTIAL: AtomicU64 = AtomicU64::new(1);

struct Entry {
    bytes: u64,
    used: u64,
}

#[derive(Default)]
struct Held {
    entries: HashMap<PathBuf, Entry>,
    bytes: u64,
    clock: u64,
}

struct State {
    root: PathBuf,
    budget: u64,
    held: Mutex<Held>,
    loads: AtomicU64,
    stores: AtomicU64,
    evictions: AtomicU64,
    refused: AtomicU64,
}

#[derive(Clone)]
pub struct ArtifactCache {
    state: Option<Arc<State>>,
}

impl ArtifactCache {
    pub fn at(root: impl Into<PathBuf>) -> Self {
        Self::bounded(root, DEFAULT_ARTIFACT_BYTES)
    }

    pub fn bounded(root: impl Into<PathBuf>, budget: u64) -> Self {
        assert!(
            budget > 0,
            "a device artifact cache of no byte holds nothing",
        );
        let root = root.into();
        fs::create_dir_all(&root).unwrap_or_else(|error| {
            panic!(
                "creating the device artifact cache {}: {error}",
                root.display(),
            )
        });
        Self::of(root, budget)
    }

    pub fn default_location(budget: u64) -> Self {
        for root in candidate_roots() {
            if fs::create_dir_all(&root).is_ok() {
                return Self::of(root, budget);
            }
        }
        Self { state: None }
    }

    fn of(root: PathBuf, budget: u64) -> Self {
        let state = Arc::new(State {
            root,
            budget,
            held: Mutex::new(Held::default()),
            loads: AtomicU64::new(0),
            stores: AtomicU64::new(0),
            evictions: AtomicU64::new(0),
            refused: AtomicU64::new(0),
        });
        adopt(&state);
        Self { state: Some(state) }
    }

    pub fn root(&self) -> Option<&Path> {
        self.state.as_deref().map(|state| state.root.as_path())
    }

    pub fn budget(&self) -> u64 {
        self.state.as_deref().map_or(0, |state| state.budget)
    }

    pub fn bytes(&self) -> u64 {
        let Some(state) = self.state.as_deref() else {
            return 0;
        };
        state
            .held
            .lock()
            .expect("a device artifact cache is never poisoned")
            .bytes
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

    pub fn evictions(&self) -> u64 {
        self.state
            .as_deref()
            .map_or(0, |state| state.evictions.load(Ordering::Relaxed))
    }

    pub fn refused(&self) -> u64 {
        self.state
            .as_deref()
            .map_or(0, |state| state.refused.load(Ordering::Relaxed))
    }

    pub fn holds(&self, key: &str) -> bool {
        let Some(path) = self.path(key) else {
            return false;
        };
        match fs::metadata(&path) {
            Ok(meta) => meta.is_file() && meta.len() > 0,
            Err(error) if error.kind() == ErrorKind::NotFound => false,
            Err(error) => panic!("inspecting the device artifact {}: {error}", path.display()),
        }
    }

    pub fn load(&self, key: &str) -> Option<Vec<u8>> {
        let state = self.state.as_deref()?;
        let path = self.path(key)?;
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::NotFound => return None,
            Err(error) => panic!("reading the device artifact {}: {error}", path.display()),
        };
        let mut payload = Vec::new();
        file.read_to_end(&mut payload).unwrap_or_else(|error| {
            panic!("reading the device artifact {}: {error}", path.display())
        });
        assert!(
            !payload.is_empty(),
            "the device artifact {} holds no byte",
            path.display(),
        );
        state.loads.fetch_add(1, Ordering::Relaxed);
        let mut held = state
            .held
            .lock()
            .expect("a device artifact cache is never poisoned");
        held.clock += 1;
        let used = held.clock;
        let bytes = payload.len() as u64;
        let previous = match held.entries.get_mut(&path) {
            Some(entry) => {
                entry.used = used;
                let previous = entry.bytes;
                entry.bytes = bytes;
                previous
            }
            None => 0,
        };
        held.entries
            .entry(path.clone())
            .or_insert(Entry { bytes, used });
        held.bytes = held.bytes + bytes - previous;
        let reclaim = held.bytes.saturating_sub(state.budget);
        drop(held);
        if reclaim > 0 {
            enforce(state, Some(&path), reclaim);
        }
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
        let bytes = payload.len() as u64;
        if bytes > state.budget {
            state.refused.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let path = self
            .file(key)
            .expect("an enabled device artifact cache names a file for every key");
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
        let reclaim = index(state, &path, bytes);
        if reclaim > 0 {
            enforce(state, Some(&path), reclaim);
        }
    }

    pub fn record(&self, key: &str) {
        let Some(state) = self.state.as_deref() else {
            return;
        };
        let path = self
            .path(key)
            .expect("an enabled device artifact cache names a file for every key");
        let bytes = match fs::metadata(&path) {
            Ok(meta) if meta.is_file() => meta.len(),
            Ok(_) => panic!(
                "the device artifact {} holds something beside bytes",
                path.display(),
            ),
            Err(error) if error.kind() == ErrorKind::NotFound => return,
            Err(error) => panic!("inspecting the device artifact {}: {error}", path.display()),
        };
        assert!(
            bytes > 0,
            "the device artifact {} holds no byte",
            path.display(),
        );
        if bytes > state.budget {
            state.refused.fetch_add(1, Ordering::Relaxed);
            discard(state, &path);
            return;
        }
        let reclaim = index(state, &path, bytes);
        if reclaim > 0 {
            enforce(state, Some(&path), reclaim);
        }
    }

    pub fn file(&self, key: &str) -> Option<PathBuf> {
        let path = self.path(key)?;
        fs::create_dir_all(path.parent().expect("an artifact key names a file"))
            .unwrap_or_else(|error| panic!("creating the device artifact directory: {error}"));
        Some(path)
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
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
        Some(state.root.join(key))
    }
}

fn adopt(state: &Arc<State>) {
    let mut found = Vec::new();
    collect(state, &state.root, &mut found);
    found.sort_by_key(|(_, _, modified)| *modified);
    let mut held = state
        .held
        .lock()
        .expect("a device artifact cache is never poisoned");
    for (path, bytes, _) in found {
        held.clock += 1;
        let used = held.clock;
        held.entries.insert(path, Entry { bytes, used });
        held.bytes += bytes;
    }
    let reclaim = held.bytes.saturating_sub(state.budget);
    drop(held);
    if reclaim > 0 {
        enforce(state, None, reclaim);
    }
}

fn collect(state: &Arc<State>, directory: &Path, found: &mut Vec<(PathBuf, u64, SystemTime)>) {
    let entries = fs::read_dir(directory).unwrap_or_else(|error| {
        panic!(
            "reading the device artifact cache {}: {error}",
            directory.display(),
        )
    });
    for entry in entries {
        let entry = entry.unwrap_or_else(|error| {
            panic!(
                "reading the device artifact cache {}: {error}",
                directory.display(),
            )
        });
        let path = entry.path();
        let meta = fs::symlink_metadata(&path).unwrap_or_else(|error| {
            panic!("inspecting the device artifact {}: {error}", path.display())
        });
        if meta.is_dir() {
            collect(state, &path, found);
            continue;
        }
        if !meta.is_file() {
            continue;
        }
        let bytes = meta.len();
        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let partial = path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().contains("partial-"));
        if bytes == 0 || (partial && abandoned(modified)) {
            discard(state, &path);
            continue;
        }
        if partial {
            continue;
        }
        found.push((path, bytes, modified));
    }
}

fn abandoned(modified: SystemTime) -> bool {
    SystemTime::now()
        .duration_since(modified)
        .is_ok_and(|age| age > PARTIAL_GRACE)
}

fn index(state: &State, path: &Path, bytes: u64) -> u64 {
    let mut held = state
        .held
        .lock()
        .expect("a device artifact cache is never poisoned");
    held.bytes -= held.entries.get(path).map_or(0, |entry| entry.bytes);
    held.clock += 1;
    let used = held.clock;
    held.entries
        .insert(path.to_path_buf(), Entry { bytes, used });
    held.bytes += bytes;
    held.bytes.saturating_sub(state.budget)
}

fn enforce(state: &State, keep: Option<&Path>, mut reclaim: u64) {
    while reclaim > 0 {
        let mut held = state
            .held
            .lock()
            .expect("a device artifact cache is never poisoned");
        let victim = held
            .entries
            .iter()
            .filter(|(path, _)| Some(path.as_path()) != keep)
            .min_by_key(|(_, entry)| entry.used)
            .map(|(path, entry)| (path.clone(), entry.bytes));
        let Some((path, bytes)) = victim else {
            return;
        };
        held.entries.remove(&path);
        held.bytes -= bytes;
        drop(held);
        discard(state, &path);
        reclaim = reclaim.saturating_sub(bytes);
    }
}

fn discard(state: &State, path: &Path) {
    match fs::remove_file(path) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => panic!("evicting the device artifact {}: {error}", path.display()),
    }
    if let Some(parent) = path.parent()
        && parent != state.root
    {
        let _ = fs::remove_dir(parent);
    }
    state.evictions.fetch_add(1, Ordering::Relaxed);
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

#[cfg(any(vulkan_backend, dx12_backend, metal_backend))]
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
