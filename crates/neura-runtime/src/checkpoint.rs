pub(crate) use container::Container;
use container::{Entry, LENGTH_BYTES, Reader, Tensor, parse};
use neura_abi::Element;
use neura_plan::Arena;
use neura_source::{Placed, Source};
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

mod container;
mod json;

const JSON_DEPTH_CEILING: usize = 32;
const COPY_CHUNK: usize = 1 << 16;

pub struct TensorView<'c> {
    pub element: Option<Element>,
    pub elements: u64,
    pub scale: f32,
    pub payload: &'c [u8],
    pub quanta: Option<&'c [u8]>,
}

pub struct Checkpoint {
    bytes: Vec<u8>,
    tensors: Vec<Tensor>,
    index: HashMap<String, usize>,
}

impl Checkpoint {
    pub(crate) fn pack(container: &Container, read: &mut Reader<'_>) -> Self {
        let held = usize::try_from(container.bytes()).unwrap_or_else(|_| {
            panic!(
                "a container of {} bytes outruns the address space of this machine",
                container.bytes(),
            )
        });
        let mut bytes = vec![0u8; held];
        bytes[..container.header().len()].copy_from_slice(container.header());
        container.stream(read, &mut |at, chunk| {
            let start = at as usize;
            bytes[start..start + chunk.len()].copy_from_slice(chunk);
        });
        Self::read(bytes)
    }

    pub fn decode(bytes: &[u8]) -> Self {
        Self::read(bytes.to_vec())
    }

    fn read(bytes: Vec<u8>) -> Self {
        let (length, header) = header_of(&bytes);
        let (tensors, index) = parse(header, LENGTH_BYTES + length, bytes.len());
        Self {
            bytes,
            tensors,
            index,
        }
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.name.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<TensorView<'_>> {
        let tensor = self.tensors.get(*self.index.get(name)?)?;
        Some(TensorView {
            element: tensor.element,
            elements: tensor.elements,
            scale: tensor.scale,
            payload: &self.bytes[tensor.payload.clone()],
            quanta: tensor
                .quanta
                .as_ref()
                .map(|quanta| &self.bytes[quanta.clone()]),
        })
    }
}

fn header_of(bytes: &[u8]) -> (usize, &str) {
    assert!(
        bytes.len() >= LENGTH_BYTES,
        "a container holds {} bytes where its length alone takes {LENGTH_BYTES}",
        bytes.len(),
    );
    let length = u64::from_le_bytes(
        bytes[..LENGTH_BYTES]
            .try_into()
            .expect("a container opens with a length word"),
    ) as usize;
    assert!(
        length >= 2 && LENGTH_BYTES + length <= bytes.len(),
        "a container of {} bytes declares a {length} byte header",
        bytes.len(),
    );
    (
        length,
        std::str::from_utf8(&bytes[LENGTH_BYTES..LENGTH_BYTES + length])
            .expect("a container header holds UTF-8"),
    )
}

fn placed_of(tensor: &Tensor) -> Placed {
    let element = tensor.element.unwrap_or_else(|| {
        panic!(
            "a container holds the tensor {} in a dtype this framework does not read",
            tensor.name,
        )
    });
    Placed {
        element,
        elements: tensor.elements,
        scale: tensor.scale,
        payload_bytes: tensor.payload.len() as u64,
        table_bytes: tensor
            .quanta
            .as_ref()
            .map_or(0, |quanta| quanta.len() as u64),
    }
}

impl Source for Checkpoint {
    fn placed(&self, name: &str) -> Option<Placed> {
        let tensor = self.tensors.get(*self.index.get(name)?)?;
        Some(placed_of(tensor))
    }

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        let tensor = &self.tensors[*self.index.get(name).unwrap_or_else(|| {
            panic!("this container holds no tensor named {name}");
        })];
        for chunk in self.bytes[tensor.payload.clone()].chunks(COPY_CHUNK) {
            sink(chunk);
        }
    }

    fn table(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        let tensor = &self.tensors[*self.index.get(name).unwrap_or_else(|| {
            panic!("this container holds no tensor named {name}");
        })];
        if let Some(quanta) = &tensor.quanta {
            for chunk in self.bytes[quanta.clone()].chunks(COPY_CHUNK) {
                sink(chunk);
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CheckpointTensor {
    pub element: Option<Element>,
    pub elements: u64,
    pub scale: f32,
    pub payload_bytes: u64,
    pub table_bytes: u64,
}

pub struct CheckpointFile {
    file: Mutex<File>,
    path: PathBuf,
    tensors: Vec<Tensor>,
    index: HashMap<String, usize>,
    reads: AtomicU64,
}

impl CheckpointFile {
    pub fn open(path: &Path) -> Self {
        let mut file = File::open(path).unwrap_or_else(|error| {
            panic!(
                "a container at {} could not be opened: {error}",
                path.display(),
            )
        });
        let len = file
            .metadata()
            .unwrap_or_else(|error| {
                panic!(
                    "a container at {} could not be measured: {error}",
                    path.display(),
                )
            })
            .len();
        assert!(
            len >= LENGTH_BYTES as u64,
            "a container at {} holds {len} bytes where its length alone takes {LENGTH_BYTES}",
            path.display(),
        );
        let mut word = [0u8; LENGTH_BYTES];
        read_exact_at(&mut file, 0, &mut word, path);
        let length = u64::from_le_bytes(word) as usize;
        assert!(
            length >= 2 && LENGTH_BYTES as u64 + length as u64 <= len,
            "a container of {len} bytes at {} declares a {length} byte header",
            path.display(),
        );
        let mut header = vec![0u8; length];
        read_exact_at(&mut file, LENGTH_BYTES as u64, &mut header, path);
        let header = String::from_utf8(header).expect("a container header holds UTF-8");
        let (tensors, index) = parse(&header, LENGTH_BYTES + length, len as usize);
        Self {
            file: Mutex::new(file),
            path: path.to_owned(),
            tensors,
            index,
            reads: AtomicU64::new(0),
        }
    }

    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    pub fn tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|tensor| tensor.name.as_str())
    }

    pub fn tensor(&self, name: &str) -> Option<CheckpointTensor> {
        let tensor = &self.tensors[*self.index.get(name)?];
        Some(CheckpointTensor {
            element: tensor.element,
            elements: tensor.elements,
            scale: tensor.scale,
            payload_bytes: tensor.payload.len() as u64,
            table_bytes: tensor
                .quanta
                .as_ref()
                .map_or(0, |quanta| quanta.len() as u64),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn stream(&self, range: Range<usize>, sink: &mut dyn FnMut(&[u8])) {
        let mut file = self.file.lock().unwrap_or_else(PoisonError::into_inner);
        let mut left = range.len();
        let mut at = range.start;
        let mut chunk = Vec::new();
        while left > 0 {
            let take = left.min(COPY_CHUNK);
            chunk.resize(take, 0);
            read_exact_at(&mut file, at as u64, &mut chunk, &self.path);
            self.reads.fetch_add(1, Ordering::Relaxed);
            sink(&chunk);
            at += take;
            left -= take;
        }
    }
}

fn read_exact_at(file: &mut File, at: u64, bytes: &mut [u8], path: &Path) {
    file.seek(SeekFrom::Start(at))
        .and_then(|_| file.read_exact(bytes))
        .unwrap_or_else(|error| {
            panic!(
                "a container at {} could not be read at {at}: {error}",
                path.display(),
            )
        });
}

impl Source for CheckpointFile {
    fn placed(&self, name: &str) -> Option<Placed> {
        let tensor = self.tensors.get(*self.index.get(name)?)?;
        Some(placed_of(tensor))
    }

    fn payload(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        let tensor = &self.tensors[*self.index.get(name).unwrap_or_else(|| {
            panic!("this container holds no tensor named {name}");
        })];
        self.stream(tensor.payload.clone(), sink);
    }

    fn table(&self, name: &str, sink: &mut dyn FnMut(&[u8])) {
        let tensor = &self.tensors[*self.index.get(name).unwrap_or_else(|| {
            panic!("this container holds no tensor named {name}");
        })];
        if let Some(quanta) = &tensor.quanta {
            self.stream(quanta.clone(), sink);
        }
    }
}

impl std::fmt::Debug for Checkpoint {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("Checkpoint")
            .field("tensors", &self.tensors.len())
            .field("bytes", &self.bytes.len())
            .finish()
    }
}

impl std::fmt::Debug for CheckpointFile {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct("CheckpointFile")
            .field("path", &self.path)
            .field("tensors", &self.tensors.len())
            .field("reads", &self.reads())
            .finish()
    }
}

pub(crate) fn described<'r>(region: &'r Arena, section: &str) -> Vec<Entry<'r>> {
    region
        .entries()
        .iter()
        .map(|entry| Entry {
            name: entry.name.as_deref().unwrap_or_else(|| {
                panic!(
                    "the {section} at word {} of {} carries no name, and a container names every tensor it holds; declare it with a named parameter or a named state",
                    entry.word,
                    entry.element.name(),
                )
            }),
            shape: entry.shape,
            element: entry.element,
            elements: entry.elements,
            scale: entry.scale,
            word: entry.word,
        })
        .collect()
}
