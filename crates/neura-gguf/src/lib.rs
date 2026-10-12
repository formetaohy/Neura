mod header;
mod value;

pub use header::Tensor;
pub use value::Value;

use header::{Fault, Header, expect, parse};
use neura_source::{Placed, Source};
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

const COPY_CHUNK: usize = 1 << 16;
const HEADER_STEP: usize = 1 << 20;

pub struct Gguf {
    bytes: Vec<u8>,
    header: Header,
    index: HashMap<String, usize>,
}

impl Gguf {
    pub fn decode(bytes: &[u8]) -> Self {
        let header = expect(parse(bytes));
        bounds(&header, bytes.len() as u64, "a gguf");
        let index = index_of(&header);
        Self {
            bytes: bytes.to_vec(),
            header,
            index,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.header.metadata
    }

    pub fn value(&self, key: &str) -> Option<&Value> {
        self.header.metadata.get(key)
    }

    pub fn tensors(&self) -> usize {
        self.header.tensors.len()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.iter().map(|(name, _)| name.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<Tensor> {
        Some(self.header.tensors[*self.index.get(name)?].1)
    }

    fn range(&self, name: &str) -> std::ops::Range<usize> {
        let tensor = &self.header.tensors[*self.index.get(name).unwrap_or_else(|| {
            panic!("this gguf holds no tensor named {name}");
        })];
        let start = self.header.data + tensor.1.at;
        start as usize..(start + tensor.1.bytes) as usize
    }
}

impl Source for Gguf {
    fn placed(&self, name: &str) -> Option<Placed> {
        Some(placed(self.tensor(name)?))
    }

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        for chunk in self.bytes[self.range(name)].chunks(COPY_CHUNK) {
            sink(chunk);
        }
    }

    fn table(&self, name: &str, _sink: &mut dyn FnMut(&[u8])) {
        assert!(
            self.index.contains_key(name),
            "this gguf holds no tensor named {name}",
        );
    }
}

pub struct GgufFile {
    file: Mutex<File>,
    path: PathBuf,
    header: Header,
    index: HashMap<String, usize>,
    reads: AtomicU64,
}

impl GgufFile {
    pub fn open(path: &Path) -> Self {
        let mut file = File::open(path).unwrap_or_else(|error| {
            panic!("a gguf at {} could not be opened: {error}", path.display(),)
        });
        let len = file
            .metadata()
            .unwrap_or_else(|error| {
                panic!(
                    "a gguf at {} could not be measured: {error}",
                    path.display(),
                )
            })
            .len();
        let mut held = HEADER_STEP.min(len.max(1) as usize);
        let header = loop {
            let mut bytes = vec![0u8; held];
            read_exact_at(&mut file, 0, &mut bytes, path);
            match parse(&bytes) {
                Ok(header) => break header,
                Err(Fault::Malformed(message)) => {
                    panic!("a gguf at {} holds {message}", path.display(),)
                }
                Err(Fault::Truncated) => {
                    assert!(
                        (held as u64) < len,
                        "a gguf at {} holds {len} bytes where its metadata and its tensor table run past them, and a header closes before the first tensor of a gguf",
                        path.display(),
                    );
                    held = held.saturating_mul(2).min(len as usize);
                }
            }
        };
        bounds(&header, len, &format!("a gguf at {}", path.display()));
        let index = index_of(&header);
        Self {
            file: Mutex::new(file),
            path: path.to_owned(),
            header,
            index,
            reads: AtomicU64::new(0),
        }
    }

    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.header.metadata
    }

    pub fn value(&self, key: &str) -> Option<&Value> {
        self.header.metadata.get(key)
    }

    pub fn tensors(&self) -> usize {
        self.header.tensors.len()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.header.tensors.iter().map(|(name, _)| name.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<Tensor> {
        Some(self.header.tensors[*self.index.get(name)?].1)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn range(&self, name: &str) -> std::ops::Range<u64> {
        let tensor = &self.header.tensors[*self.index.get(name).unwrap_or_else(|| {
            panic!("this gguf holds no tensor named {name}");
        })];
        let start = self.header.data + tensor.1.at;
        start..start + tensor.1.bytes
    }

    fn stream(&self, range: std::ops::Range<u64>, sink: &mut dyn FnMut(&[u8])) {
        let mut file = self.file.lock().unwrap_or_else(PoisonError::into_inner);
        let mut left = range.end - range.start;
        let mut at = range.start;
        let mut chunk = Vec::new();
        while left > 0 {
            let take = left.min(COPY_CHUNK as u64) as usize;
            chunk.resize(take, 0);
            read_exact_at(&mut file, at, &mut chunk, &self.path);
            self.reads.fetch_add(1, Ordering::Relaxed);
            sink(&chunk);
            at += take as u64;
            left -= take as u64;
        }
    }
}

impl Source for GgufFile {
    fn placed(&self, name: &str) -> Option<Placed> {
        Some(placed(self.tensor(name)?))
    }

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        self.stream(self.range(name), sink);
    }

    fn table(&self, name: &str, _sink: &mut dyn FnMut(&[u8])) {
        assert!(
            self.index.contains_key(name),
            "this gguf holds no tensor named {name}",
        );
    }
}

fn placed(tensor: Tensor) -> Placed {
    Placed {
        element: tensor.element,
        elements: tensor.elements,
        scale: 1.0,
        payload_bytes: tensor.bytes,
        table_bytes: 0,
    }
}

fn bounds(header: &Header, held: u64, what: &str) {
    for (name, tensor) in &header.tensors {
        let end = header.data + tensor.at + tensor.bytes;
        assert!(
            end <= held,
            "{what} holds {held} bytes where the tensor {name} runs to byte {end}",
        );
    }
}

fn index_of(header: &Header) -> HashMap<String, usize> {
    let mut index = HashMap::new();
    for (at, (name, _)) in header.tensors.iter().enumerate() {
        assert!(
            index.insert(name.clone(), at).is_none(),
            "two tensors of one gguf carry the name {name}, and a name identifies one tensor",
        );
    }
    index
}

fn read_exact_at(file: &mut File, at: u64, bytes: &mut [u8], path: &Path) {
    file.seek(SeekFrom::Start(at))
        .and_then(|_| file.read_exact(bytes))
        .unwrap_or_else(|error| {
            panic!(
                "a gguf at {} could not be read at {at}: {error}",
                path.display(),
            )
        });
}

impl std::fmt::Debug for Gguf {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("Gguf")
            .field("tensors", &self.header.tensors.len())
            .field("metadata", &self.header.metadata.len())
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl std::fmt::Debug for GgufFile {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("GgufFile")
            .field("path", &self.path)
            .field("tensors", &self.header.tensors.len())
            .field("metadata", &self.header.metadata.len())
            .field("reads", &self.reads())
            .finish()
    }
}
