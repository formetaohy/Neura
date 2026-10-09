use super::json::Json;
use neura_abi::{Element, WORD_BYTES};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::ops::Range;
use std::path::Path;

pub(crate) const LENGTH_BYTES: usize = 8;
pub(crate) const METADATA_KEY: &str = "__metadata__";
const ELEMENT_PREFIX: &str = "neura.element.";
const ELEMENTS_PREFIX: &str = "neura.elements.";
const QUANTUM_PREFIX: &str = "neura.quantum.";
const QUANTA_SUFFIX: &str = ".quanta";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Dtype {
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

pub(crate) type Reader<'a> = dyn FnMut(u64, u64, &mut dyn FnMut(&[u8])) + 'a;

pub(crate) struct Tensor {
    pub(crate) name: String,
    pub(crate) dtype: Option<Dtype>,
    pub(crate) elements: u64,
    pub(crate) element: Option<Element>,
    pub(crate) scale: f32,
    pub(crate) payload: Range<usize>,
    pub(crate) quanta: Option<Range<usize>>,
}

pub(crate) struct Entry<'a> {
    pub(crate) name: &'a str,
    pub(crate) shape: [u32; 4],
    pub(crate) element: Element,
    pub(crate) elements: u64,
    pub(crate) scale: f32,
    pub(crate) word: u64,
}

struct Part {
    name: String,
    dtype: Dtype,
    shape: Vec<u64>,
    word: u64,
    bytes: u64,
}

pub(crate) struct Container {
    header: Vec<u8>,
    files: Vec<Part>,
    data: u64,
}

impl Container {
    pub(crate) fn of(entries: &[Entry<'_>]) -> Self {
        let mut files = Vec::new();
        let mut metadata = Vec::new();
        for entry in entries {
            assert!(
                !entry.name.is_empty() && entry.name != METADATA_KEY,
                "a container names every tensor it holds, and one of {} tensors carries the reserved name {}",
                entries.len(),
                entry.name,
            );
            let tight = tight_bytes(entry.element, entry.elements);
            if entry.element.per_block() {
                let quanta = entry.element.quanta(entry.elements);
                files.push(Part {
                    name: entry.name.to_owned(),
                    dtype: Dtype::U8,
                    shape: vec![tight],
                    word: entry.word,
                    bytes: tight,
                });
                files.push(Part {
                    name: format!("{}{QUANTA_SUFFIX}", entry.name),
                    dtype: Dtype::F32,
                    shape: vec![quanta],
                    word: entry.word + entry.element.payload_words(entry.elements),
                    bytes: quanta * WORD_BYTES,
                });
                metadata.push((
                    format!("{ELEMENT_PREFIX}{}", entry.name),
                    entry.element.name().to_owned(),
                ));
                metadata.push((
                    format!("{ELEMENTS_PREFIX}{}", entry.name),
                    entry.elements.to_string(),
                ));
                continue;
            }
            let dtype = Dtype::of_element(entry.element).unwrap_or_else(|| {
                panic!(
                    "tensor {} holds {} numbers a container stores by its element",
                    entry.name,
                    entry.element.name(),
                )
            });
            files.push(Part {
                name: entry.name.to_owned(),
                dtype,
                shape: natural(entry.shape),
                word: entry.word,
                bytes: tight,
            });
            if entry.element.per_tensor() {
                metadata.push((
                    format!("{QUANTUM_PREFIX}{}", entry.name),
                    entry.scale.to_string(),
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
        let mut names = HashMap::new();
        for file in &files {
            assert!(
                names.insert(file.name.clone(), ()).is_none(),
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
            data += file.bytes;
            header.push(',');
            header.push_str(&data.to_string());
            header.push_str("]}");
        }
        header.push('}');
        let mut header = header.into_bytes();
        let padded = header.len().next_multiple_of(LENGTH_BYTES);
        header.resize(padded, b' ');
        let mut framed = Vec::with_capacity(LENGTH_BYTES + padded);
        framed.extend_from_slice(&(padded as u64).to_le_bytes());
        framed.extend_from_slice(&header);
        Self {
            header: framed,
            files,
            data,
        }
    }

    pub(crate) fn header(&self) -> &[u8] {
        &self.header
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.header.len() as u64 + self.data
    }

    pub(crate) fn stream(&self, read: &mut Reader<'_>, write: &mut dyn FnMut(u64, &[u8])) {
        let mut at = self.header.len() as u64;
        for file in &self.files {
            let mut handed = 0u64;
            read(file.word, file.bytes, &mut |chunk| {
                write(at, chunk);
                at += chunk.len() as u64;
                handed += chunk.len() as u64;
            });
            assert_eq!(
                handed, file.bytes,
                "tensor {} hands a container {handed} bytes where it holds {}",
                file.name, file.bytes,
            );
        }
        assert_eq!(
            at,
            self.bytes(),
            "a container of {} bytes streams {at} of them",
            self.bytes(),
        );
    }

    pub(crate) fn write(&self, path: &Path, read: &mut Reader<'_>) {
        let mut file = File::create(path).unwrap_or_else(|error| {
            panic!(
                "a container at {} could not be created: {error}",
                path.display(),
            )
        });
        file.write_all(self.header()).unwrap_or_else(|error| {
            panic!(
                "a container at {} could not be written at its header: {error}",
                path.display(),
            )
        });
        let mut at = self.header.len() as u64;
        self.stream(read, &mut |offset, chunk| {
            assert_eq!(
                offset,
                at,
                "a container at {} writes its bytes in the order its header names",
                path.display(),
            );
            file.write_all(chunk).unwrap_or_else(|error| {
                panic!(
                    "a container at {} could not be written at {at}: {error}",
                    path.display(),
                )
            });
            at += chunk.len() as u64;
        });
        file.flush().unwrap_or_else(|error| {
            panic!(
                "a container at {} could not be flushed: {error}",
                path.display(),
            )
        });
    }
}

struct Described {
    name: String,
    dtype: Option<Dtype>,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
}

pub(crate) fn parse(
    header: &str,
    data: usize,
    container: usize,
) -> (Vec<Tensor>, HashMap<String, usize>) {
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

pub(crate) fn natural(shape: [u32; 4]) -> Vec<u64> {
    let mut dims = shape.iter().copied().map(u64::from).collect::<Vec<u64>>();
    while dims.len() > 1 && dims[0] == 1 {
        dims.remove(0);
    }
    while dims.len() > 1 && dims[dims.len() - 1] == 1 {
        dims.pop();
    }
    dims
}

pub(crate) fn tight_bytes(element: Element, elements: u64) -> u64 {
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
