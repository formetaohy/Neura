use neura_abi::{Element, WORD_BYTES};
use neura_plan::Region;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

mod json;

use json::Json;

const LENGTH_BYTES: usize = 8;
const METADATA_KEY: &str = "__metadata__";
const ELEMENT_PREFIX: &str = "neura.element.";
const ELEMENTS_PREFIX: &str = "neura.elements.";
const QUANTUM_PREFIX: &str = "neura.quantum.";
const QUANTA_SUFFIX: &str = ".quanta";
const JSON_DEPTH_CEILING: usize = 32;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Dtype {
    F32,
    F16,
    Bfloat16,
    Fp8E4M3,
    Fp8E5M2,
    Int8,
    U8,
}

impl Dtype {
    const ALL: &'static [Dtype] = &[
        Self::F32,
        Self::F16,
        Self::Bfloat16,
        Self::Fp8E4M3,
        Self::Fp8E5M2,
        Self::Int8,
        Self::U8,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bfloat16 => "BF16",
            Self::Fp8E4M3 => "F8_E4M3",
            Self::Fp8E5M2 => "F8_E5M2",
            Self::Int8 => "I8",
            Self::U8 => "U8",
        }
    }

    fn bits(self) -> u64 {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::Bfloat16 => 16,
            Self::Fp8E4M3 | Self::Fp8E5M2 | Self::Int8 | Self::U8 => 8,
        }
    }

    fn of(name: &str) -> Option<Self> {
        Self::ALL.iter().copied().find(|dtype| dtype.name() == name)
    }

    fn of_element(element: Element) -> Option<Self> {
        match element {
            Element::Single => Some(Self::F32),
            Element::Half => Some(Self::F16),
            Element::Bfloat16 => Some(Self::Bfloat16),
            Element::Fp8E4M3 => Some(Self::Fp8E4M3),
            Element::Fp8E5M2 => Some(Self::Fp8E5M2),
            Element::Int8 => Some(Self::Int8),
            Element::Int4 | Element::Fp4E2M1 => None,
        }
    }

    fn element(self) -> Option<Element> {
        match self {
            Self::F32 => Some(Element::Single),
            Self::F16 => Some(Element::Half),
            Self::Bfloat16 => Some(Element::Bfloat16),
            Self::Fp8E4M3 => Some(Element::Fp8E4M3),
            Self::Fp8E5M2 => Some(Element::Fp8E5M2),
            Self::Int8 => Some(Element::Int8),
            Self::U8 => None,
        }
    }
}

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

struct Tensor {
    name: String,
    dtype: Option<Dtype>,
    elements: u64,
    element: Option<Element>,
    scale: f32,
    payload: Range<usize>,
    quanta: Option<Range<usize>>,
}

pub(crate) struct TensorData<'a> {
    pub name: &'a str,
    pub shape: [u32; 4],
    pub element: Element,
    pub elements: u64,
    pub scale: f32,
    pub payload: &'a [u8],
    pub quanta: &'a [u8],
}

struct FileTensor<'a> {
    name: String,
    dtype: Dtype,
    shape: Vec<u64>,
    bytes: &'a [u8],
}

impl Checkpoint {
    pub(crate) fn pack(tensors: &[TensorData<'_>]) -> Self {
        let mut files = Vec::new();
        let mut metadata = Vec::new();
        for tensor in tensors {
            assert!(
                !tensor.name.is_empty() && tensor.name != METADATA_KEY,
                "a container names every tensor it holds, and one of {} tensors carries the reserved name {}",
                tensors.len(),
                tensor.name,
            );
            let tight = tight_bytes(tensor.element, tensor.elements) as usize;
            assert!(
                tensor.payload.len() >= tight,
                "tensor {} packs {tight} bytes of {} numbers where its store holds {}",
                tensor.name,
                tensor.element.name(),
                tensor.payload.len(),
            );
            if tensor.element.per_block() {
                let quanta = tensor.element.quanta(tensor.elements);
                assert!(
                    tensor.quanta.len() as u64 == quanta * WORD_BYTES,
                    "tensor {} walks {quanta} quanta where its store holds {} bytes",
                    tensor.name,
                    tensor.quanta.len(),
                );
                files.push(FileTensor {
                    name: tensor.name.to_owned(),
                    dtype: Dtype::U8,
                    shape: vec![tight as u64],
                    bytes: &tensor.payload[..tight],
                });
                files.push(FileTensor {
                    name: format!("{}{QUANTA_SUFFIX}", tensor.name),
                    dtype: Dtype::F32,
                    shape: vec![quanta],
                    bytes: tensor.quanta,
                });
                metadata.push((
                    format!("{ELEMENT_PREFIX}{}", tensor.name),
                    tensor.element.name().to_owned(),
                ));
                metadata.push((
                    format!("{ELEMENTS_PREFIX}{}", tensor.name),
                    tensor.elements.to_string(),
                ));
                continue;
            }
            let dtype = Dtype::of_element(tensor.element).unwrap_or_else(|| {
                panic!(
                    "tensor {} holds {} numbers a container stores by its element",
                    tensor.name,
                    tensor.element.name(),
                )
            });
            files.push(FileTensor {
                name: tensor.name.to_owned(),
                dtype,
                shape: natural(tensor.shape),
                bytes: &tensor.payload[..tight],
            });
            if tensor.element.per_tensor() {
                metadata.push((
                    format!("{QUANTUM_PREFIX}{}", tensor.name),
                    tensor.scale.to_string(),
                ));
            }
        }
        files.sort_by(|left, right| {
            right
                .dtype
                .bits()
                .cmp(&left.dtype.bits())
                .then(left.name.cmp(&right.name))
        });
        let mut keys = HashMap::new();
        for file in &files {
            assert!(
                keys.insert(file.name.clone(), ()).is_none(),
                "two tensors of one container carry the name {}, and a name identifies one tensor",
                file.name,
            );
        }
        let mut header = String::from("{");
        if !metadata.is_empty() {
            header.push_str("\"__metadata__\":{");
            for (index, (key, value)) in metadata.iter().enumerate() {
                if index > 0 {
                    header.push(',');
                }
                escape(&mut header, key);
                header.push(':');
                escape(&mut header, value);
            }
            header.push_str("},");
        }
        let mut data = 0u64;
        for (index, file) in files.iter().enumerate() {
            if index > 0 {
                header.push(',');
            }
            escape(&mut header, &file.name);
            header.push_str(":{\"dtype\":");
            escape(&mut header, file.dtype.name());
            header.push_str(",\"shape\":[");
            for (axis, dim) in file.shape.iter().enumerate() {
                if axis > 0 {
                    header.push(',');
                }
                header.push_str(&dim.to_string());
            }
            header.push_str("],\"data_offsets\":[");
            header.push_str(&data.to_string());
            data += file.bytes.len() as u64;
            header.push(',');
            header.push_str(&data.to_string());
            header.push_str("]}");
        }
        header.push('}');
        let mut header = header.into_bytes();
        let padded = header.len().next_multiple_of(LENGTH_BYTES);
        header.resize(padded, b' ');
        let mut bytes = Vec::with_capacity(LENGTH_BYTES + padded + data as usize);
        bytes.extend_from_slice(&(padded as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        for file in &files {
            bytes.extend_from_slice(file.bytes);
        }
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

fn parse(header: &str, data: usize, container: usize) -> (Vec<Tensor>, HashMap<String, usize>) {
    let parsed = Json::parse(header.trim_end_matches(' '));
    let Json::Object(fields) = parsed else {
        panic!(
            "a container header holds {} where it holds a JSON object",
            parsed.kind(),
        );
    };
    let mut metadata = HashMap::new();
    let mut described = Vec::new();
    for (key, value) in fields {
        if key == METADATA_KEY {
            let Json::Object(pairs) = value else {
                panic!(
                    "container metadata holds {} where it holds a JSON object",
                    value.kind(),
                );
            };
            for (key, value) in pairs {
                let Json::String(value) = value else {
                    panic!(
                        "the metadata key {key} holds {} where metadata values are strings",
                        value.kind(),
                    );
                };
                assert!(
                    metadata.insert(key.clone(), value).is_none(),
                    "two metadata keys of one container carry the name {key}",
                );
            }
            continue;
        }
        let Json::Object(fields) = value else {
            panic!(
                "tensor {key} holds {} where a tensor describes a JSON object",
                value.kind(),
            );
        };
        let mut dtype = None;
        let mut shape = None;
        let mut offsets = None;
        for (field, value) in fields {
            match field.as_str() {
                "dtype" => {
                    let Json::String(name) = value else {
                        panic!(
                            "tensor {key} declares {} where a dtype is a string",
                            value.kind(),
                        );
                    };
                    dtype = Some(Dtype::of(&name));
                }
                "shape" => {
                    let Json::Array(dims) = value else {
                        panic!(
                            "tensor {key} declares {} where a shape is a JSON array",
                            value.kind(),
                        );
                    };
                    shape = Some(
                        dims.into_iter()
                            .map(|dim| dim.into_integer(&format!("a dimension of tensor {key}")))
                            .collect::<Vec<u64>>(),
                    );
                }
                "data_offsets" => {
                    let Json::Array(bounds) = value else {
                        panic!(
                            "tensor {key} declares {} where data offsets are a JSON array",
                            value.kind(),
                        );
                    };
                    assert!(
                        bounds.len() == 2,
                        "tensor {key} declares {} data offsets where a tensor holds two",
                        bounds.len(),
                    );
                    let mut bounds = bounds.into_iter();
                    let begin = bounds
                        .next()
                        .expect("a bounds array holds its begin")
                        .into_integer(&format!("the begin of tensor {key}"));
                    let end = bounds
                        .next()
                        .expect("a bounds array holds its end")
                        .into_integer(&format!("the end of tensor {key}"));
                    offsets = Some((begin, end));
                }
                _ => {}
            }
        }
        let dtype = dtype.unwrap_or_else(|| panic!("tensor {key} declares no dtype"));
        let shape = shape.unwrap_or_else(|| panic!("tensor {key} declares no shape"));
        let (begin, end) = offsets.unwrap_or_else(|| panic!("tensor {key} declares no offsets"));
        described.push(Described {
            name: key,
            dtype,
            shape,
            begin,
            end,
        });
    }
    let mut spans = described
        .iter()
        .map(|tensor| (tensor.begin, tensor.end))
        .collect::<Vec<(u64, u64)>>();
    spans.sort_unstable();
    let mut at = 0u64;
    for (begin, end) in &spans {
        assert!(
            *begin == at && begin <= end,
            "a container holds {} bytes of buffer where the next tensor begins at {begin}",
            container - data,
        );
        at = *end;
    }
    assert!(
        at == (container - data) as u64,
        "a container of {} bytes of buffer indexes {at} of them, and every byte of a container belongs to a tensor",
        container - data,
    );
    let mut index = HashMap::new();
    let mut tensors = Vec::with_capacity(described.len());
    for tensor in &described {
        assert!(
            index.insert(tensor.name.clone(), tensors.len()).is_none(),
            "two tensors of one container carry the name {}",
            tensor.name,
        );
        let elements = tensor.shape.iter().fold(1u64, |total, dim| {
            total.checked_mul(*dim).unwrap_or_else(|| {
                panic!(
                    "tensor {} spans more numbers than a word counts",
                    tensor.name,
                )
            })
        });
        if let Some(dtype) = tensor.dtype {
            let packed = elements
                .checked_mul(dtype.bits())
                .map(|bits| bits / 8)
                .unwrap_or_else(|| {
                    panic!(
                        "tensor {} spans more bytes than a container indexes",
                        tensor.name,
                    )
                });
            assert!(
                tensor.end - tensor.begin == packed,
                "tensor {} declares {elements} {} numbers where its bytes pack {packed}",
                tensor.name,
                dtype.name(),
            );
        }
        let scale = metadata
            .get(&format!("{QUANTUM_PREFIX}{}", tensor.name))
            .map_or(1.0, |value| {
                value.parse::<f32>().unwrap_or_else(|_| {
                    panic!(
                        "tensor {} quantizes by {value}, which is not a number",
                        tensor.name,
                    )
                })
            });
        tensors.push(Tensor {
            name: tensor.name.clone(),
            dtype: tensor.dtype,
            elements,
            element: tensor.dtype.and_then(Dtype::element),
            scale,
            payload: (data + tensor.begin as usize)..(data + tensor.end as usize),
            quanta: None,
        });
    }
    let mut quantized = Vec::new();
    for (at, tensor) in tensors.iter().enumerate() {
        let Some(name) = metadata.get(&format!("{ELEMENT_PREFIX}{}", tensor.name)) else {
            continue;
        };
        let element = find_element(name).unwrap_or_else(|| {
            panic!(
                "tensor {} holds numbers of {name}, an element this framework does not declare",
                tensor.name,
            )
        });
        let elements = metadata
            .get(&format!("{ELEMENTS_PREFIX}{}", tensor.name))
            .map_or_else(
                || {
                    panic!(
                        "tensor {} packs {} numbers and declares no element count",
                        tensor.name,
                        element.name(),
                    )
                },
                |value| {
                    value.parse::<u64>().unwrap_or_else(|_| {
                        panic!(
                            "tensor {} holds {value} numbers, which is not a count",
                            tensor.name,
                        )
                    })
                },
            );
        assert!(
            element.per_block(),
            "tensor {} holds {} numbers a container stores by its element",
            tensor.name,
            element.name(),
        );
        assert!(
            tensor.dtype == Some(Dtype::U8),
            "tensor {} holds {} where a block quantized tensor packs bytes",
            tensor.name,
            tensor
                .dtype
                .map_or("an unreadable dtype".to_owned(), |dtype| dtype
                    .name()
                    .to_owned()),
        );
        let packed = tight_bytes(element, elements);
        assert!(
            tensor.payload.len() as u64 == packed,
            "tensor {} holds {} bytes where {elements} {} numbers pack {packed}",
            tensor.name,
            tensor.payload.len(),
            element.name(),
        );
        let quanta_name = format!("{}{QUANTA_SUFFIX}", tensor.name);
        let quanta = tensors.get(*index.get(&quanta_name).unwrap_or_else(|| {
            panic!(
                "tensor {} quantizes every {} numbers and the container holds no {quanta_name}",
                tensor.name,
                element.block(),
            )
        }));
        let quanta = quanta.unwrap_or_else(|| {
            panic!("the companion {quanta_name} holds no tensor");
        });
        let count = element.quanta(elements);
        assert!(
            quanta.dtype == Some(Dtype::F32) && quanta.elements == count,
            "tensor {} walks {count} quanta where {quanta_name} holds {} numbers of {}",
            tensor.name,
            quanta.elements,
            quanta
                .dtype
                .map_or("an unreadable dtype".to_owned(), |dtype| dtype
                    .name()
                    .to_owned()),
        );
        quantized.push((at, element, elements, quanta.payload.clone()));
    }
    for (at, element, elements, quanta) in quantized {
        tensors[at].element = Some(element);
        tensors[at].elements = elements;
        tensors[at].quanta = Some(quanta);
    }
    (tensors, index)
}

impl Checkpoint {
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

pub(crate) struct Placed {
    pub(crate) element: Option<Element>,
    pub(crate) elements: u64,
    pub(crate) scale: f32,
    pub(crate) payload: Range<usize>,
    pub(crate) quanta: Option<Range<usize>>,
}

const COPY_CHUNK: usize = 1 << 16;

pub(crate) trait Source {
    fn placed(&self, name: &str) -> Option<Placed>;
    fn payload(&self, range: Range<usize>, sink: &mut dyn FnMut(&[u8]));

    fn image(&self, weights: &Region, state: &Region, words: u64) -> Vec<u8> {
        let mut image = vec![0u8; (words * WORD_BYTES) as usize];
        self.pour(weights, state, words, &mut |at, bytes| {
            image[at as usize..at as usize + bytes.len()].copy_from_slice(bytes);
        });
        image
    }

    fn pour(
        &self,
        weights: &Region,
        state: &Region,
        words: u64,
        write: &mut dyn FnMut(u64, &[u8]),
    ) {
        for entry in weights.entries().iter().chain(state.entries()) {
            let name = entry.name.as_deref().unwrap_or_else(|| {
                panic!(
                    "the tensor at word {} of {} carries no name, and a container names every tensor it holds; declare it with a named parameter or a named state",
                    entry.word,
                    entry.element.name(),
                )
            });
            let placed = self.placed(name).unwrap_or_else(|| {
                panic!(
                    "this container holds no tensor named {name}, and the graph declares it at word {}",
                    entry.word,
                )
            });
            let element = placed.element.unwrap_or_else(|| {
                panic!("this container holds {name} in a dtype this framework does not read")
            });
            assert!(
                element == entry.element && placed.elements == entry.elements,
                "this container holds {name} as {} numbers of {} where the graph declares {} numbers of {}",
                placed.elements,
                element.name(),
                entry.elements,
                entry.element.name(),
            );
            assert!(
                placed.scale == entry.scale,
                "this container quantizes {name} by {} where the graph quantizes by {}",
                placed.scale,
                entry.scale,
            );
            let at = entry.word * WORD_BYTES;
            let payload = entry.element.payload_words(entry.elements) * WORD_BYTES;
            let tight = tight_bytes(entry.element, entry.elements);
            assert!(
                at + payload <= words * WORD_BYTES,
                "this graph declares {name} at word {} of a store of {words} words",
                entry.word,
            );
            assert!(
                placed.payload.len() as u64 >= tight,
                "this container holds {} bytes of {name} where {} numbers of {} pack {tight}",
                placed.payload.len(),
                entry.elements,
                entry.element.name(),
            );
            let range = placed.payload.start..placed.payload.start + tight as usize;
            let mut copied = 0u64;
            self.payload(range, &mut |chunk| {
                write(at + copied, chunk);
                copied += chunk.len() as u64;
            });
            if entry.element.per_tensor() {
                write(at + payload, &entry.scale.to_bits().to_le_bytes());
                continue;
            }
            if entry.element.per_block() {
                let quanta = placed.quanta.unwrap_or_else(|| {
                    panic!("this container holds no quanta beside the block quantized {name}")
                });
                let count = entry.element.quanta(entry.elements) * WORD_BYTES;
                assert!(
                    quanta.len() as u64 == count,
                    "this container holds {} bytes of quanta of {name} where {} numbers of {} walk {count}",
                    quanta.len(),
                    entry.elements,
                    entry.element.name(),
                );
                let mut copied = 0u64;
                self.payload(quanta.clone(), &mut |chunk| {
                    write(at + payload + copied, chunk);
                    copied += chunk.len() as u64;
                });
            }
        }
    }
}

impl Source for Checkpoint {
    fn placed(&self, name: &str) -> Option<Placed> {
        let tensor = &self.tensors[*self.index.get(name)?];
        Some(Placed {
            element: tensor.element,
            elements: tensor.elements,
            scale: tensor.scale,
            payload: tensor.payload.clone(),
            quanta: tensor.quanta.clone(),
        })
    }

    fn payload(&self, range: Range<usize>, sink: &mut dyn FnMut(&[u8])) {
        sink(&self.bytes[range]);
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CheckpointTensor {
    pub element: Option<Element>,
    pub elements: u64,
    pub scale: f32,
    pub payload_bytes: u64,
    pub quanta_bytes: u64,
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
            quanta_bytes: tensor
                .quanta
                .as_ref()
                .map_or(0, |quanta| quanta.len() as u64),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
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
        let tensor = &self.tensors[*self.index.get(name)?];
        Some(Placed {
            element: tensor.element,
            elements: tensor.elements,
            scale: tensor.scale,
            payload: tensor.payload.clone(),
            quanta: tensor.quanta.clone(),
        })
    }

    fn payload(&self, range: Range<usize>, sink: &mut dyn FnMut(&[u8])) {
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

struct Described {
    name: String,
    dtype: Option<Dtype>,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
}

fn natural(shape: [u32; 4]) -> Vec<u64> {
    let mut dims = shape.iter().copied().map(u64::from).collect::<Vec<u64>>();
    while dims.len() > 1 && dims[0] == 1 {
        dims.remove(0);
    }
    while dims.len() > 1 && dims[dims.len() - 1] == 1 {
        dims.pop();
    }
    dims
}

fn tight_bytes(element: Element, elements: u64) -> u64 {
    let bits = 32 / element.elements_per_word();
    elements
        .checked_mul(bits)
        .unwrap_or_else(|| {
            panic!("a tensor of {elements} numbers spans more bytes than a container indexes")
        })
        .div_ceil(8)
}

fn find_element(name: &str) -> Option<Element> {
    Element::ALL
        .iter()
        .copied()
        .find(|element| element.name() == name)
}

fn escape(out: &mut String, text: &str) {
    out.push('"');
    for character in text.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            character if (character as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => out.push(character),
        }
    }
    out.push('"');
}
