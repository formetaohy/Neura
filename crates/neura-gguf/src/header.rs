use crate::value::Value;
use neura_abi::Element;
use std::collections::BTreeMap;

pub(crate) const MAGIC: [u8; 4] = *b"GGUF";
pub(crate) const DEFAULT_ALIGNMENT: u64 = 32;
pub(crate) const VERSIONS: [u32; 2] = [2, 3];

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Tensor {
    pub element: Element,
    pub elements: u64,
    pub shape: [u64; 4],
    pub rank: u32,
    pub at: u64,
    pub bytes: u64,
}

pub(crate) struct Header {
    pub(crate) metadata: BTreeMap<String, Value>,
    pub(crate) tensors: Vec<(String, Tensor)>,
    pub(crate) data: u64,
}

#[derive(PartialEq, Debug)]
pub(crate) enum Fault {
    Truncated,
    Malformed(String),
}

pub(crate) type Parsed<T> = Result<T, Fault>;

pub(crate) fn expect<T>(parsed: Parsed<T>) -> T {
    match parsed {
        Ok(value) => value,
        Err(Fault::Truncated) => panic!(
            "a gguf header runs past the bytes this file holds, and a gguf opens with its metadata and its tensor table"
        ),
        Err(Fault::Malformed(message)) => panic!("{message}"),
    }
}

pub(crate) fn parse(bytes: &[u8]) -> Parsed<Header> {
    let mut reader = Reader { bytes, at: 0 };
    let magic = reader.take(4)?;
    if magic != MAGIC {
        return Err(Fault::Malformed(format!(
            "this file opens with {magic:02x?} where a gguf opens with GGUF",
        )));
    }
    let version = reader.word()?;
    if !VERSIONS.contains(&version) {
        return Err(Fault::Malformed(format!(
            "this gguf declares version {version}, and this reader holds the versions {VERSIONS:?} whose layout it follows",
        )));
    }
    let tensors = reader.long()?;
    let pairs = reader.long()?;
    let mut metadata = BTreeMap::new();
    for _ in 0..pairs {
        let key = reader.text()?;
        let value = reader.value(0)?;
        if metadata.insert(key.clone(), value).is_some() {
            return Err(Fault::Malformed(format!(
                "two entries of this gguf carry the metadata key {key}",
            )));
        }
    }
    let alignment = match metadata.get("general.alignment") {
        Some(value) => value.integer().ok_or_else(|| {
            Fault::Malformed(
                "this gguf measures its alignment by a value that is not an integer".to_owned(),
            )
        })?,
        None => DEFAULT_ALIGNMENT,
    };
    if alignment == 0 || !alignment.is_multiple_of(8) {
        return Err(Fault::Malformed(format!(
            "this gguf aligns its tensors by {alignment}, and a gguf aligns them by a multiple of eight",
        )));
    }
    let mut held = Vec::with_capacity(usize::try_from(tensors).unwrap_or(0));
    for _ in 0..tensors {
        let name = reader.text()?;
        let rank = reader.word()?;
        if rank == 0 || rank > 4 {
            return Err(Fault::Malformed(format!(
                "the tensor {name} declares {rank} dimensions, and a gguf tensor carries one to four",
            )));
        }
        let mut shape = [1u64; 4];
        let mut elements = 1u64;
        for dim in shape.iter_mut().take(rank as usize) {
            let held = reader.long()?;
            *dim = held;
            elements = elements.checked_mul(held).ok_or_else(|| {
                Fault::Malformed(format!(
                    "the tensor {name} spans more numbers than a gguf counts",
                ))
            })?;
        }
        let kind = reader.word()?;
        let at = reader.long()?;
        if at % alignment != 0 {
            return Err(Fault::Malformed(format!(
                "the tensor {name} begins at byte {at} of the tensor data, and a gguf aligns it by {alignment}",
            )));
        }
        let element = type_element(kind).ok_or_else(|| {
            Fault::Malformed(format!(
                "the tensor {name} holds numbers of ggml type {kind} ({}), which this framework does not read",
                type_name(kind),
            ))
        })?;
        let bytes = match element.block() {
            Some(block) => {
                if !elements.is_multiple_of(u64::from(block.elements)) {
                    return Err(Fault::Malformed(format!(
                        "the tensor {name} holds {elements} numbers of {} storage, and a {} block packs {} of them",
                        element.name(),
                        element.name(),
                        block.elements,
                    )));
                }
                elements / u64::from(block.elements) * u64::from(block.bytes)
            }
            None => elements * u64::from(element.bits_per_element()) / 8,
        };
        held.push((
            name,
            Tensor {
                element,
                elements,
                shape,
                rank,
                at,
                bytes,
            },
        ));
    }
    let data = reader.at as u64;
    let data = data.next_multiple_of(alignment);
    Ok(Header {
        metadata,
        tensors: held,
        data,
    })
}

pub(crate) fn type_element(kind: u32) -> Option<Element> {
    match kind {
        0 => Some(Element::Single),
        1 => Some(Element::Half),
        8 => Some(Element::Q8_0),
        12 => Some(Element::Q4K),
        13 => Some(Element::Q5K),
        14 => Some(Element::Q6K),
        30 => Some(Element::Bfloat16),
        _ => None,
    }
}

pub(crate) fn type_name(kind: u32) -> &'static str {
    match kind {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        4 | 5 => "removed four bit",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        31..=33 => "removed four bit plane",
        34 => "TQ1_0",
        35 => "TQ2_0",
        36..=38 => "removed four bit plane",
        39 => "MXFP4",
        _ => "unknown",
    }
}

pub(crate) struct Reader<'a> {
    bytes: &'a [u8],
    pub(crate) at: usize,
}

impl<'a> Reader<'a> {
    pub(crate) fn take(&mut self, count: usize) -> Parsed<&'a [u8]> {
        let end = self.at.checked_add(count).ok_or(Fault::Truncated)?;
        if end > self.bytes.len() {
            return Err(Fault::Truncated);
        }
        let held = &self.bytes[self.at..end];
        self.at = end;
        Ok(held)
    }

    pub(crate) fn byte(&mut self) -> Parsed<u8> {
        Ok(self.take(1)?[0])
    }

    pub(crate) fn half(&mut self) -> Parsed<u16> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().expect("two bytes"),
        ))
    }

    pub(crate) fn word(&mut self) -> Parsed<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    pub(crate) fn long(&mut self) -> Parsed<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    pub(crate) fn text(&mut self) -> Parsed<String> {
        let length = self.long()?;
        let held = usize::try_from(length).map_err(|_| Fault::Truncated)?;
        let bytes = self.take(held)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Fault::Malformed("this gguf holds a string that is not UTF-8".to_owned()))
    }

    pub(crate) fn value(&mut self, depth: usize) -> Parsed<Value> {
        let kind = self.word()?;
        self.body(kind, depth)
    }

    fn body(&mut self, kind: u32, depth: usize) -> Parsed<Value> {
        if depth > 8 {
            return Err(Fault::Malformed(
                "this gguf nests arrays deeper than eight deep".to_owned(),
            ));
        }
        Ok(match kind {
            0 => Value::Unsigned8(self.byte()?),
            1 => Value::Signed8(self.byte()? as i8),
            2 => Value::Unsigned16(self.half()?),
            3 => Value::Signed16(self.half()? as i16),
            4 => Value::Unsigned32(self.word()?),
            5 => Value::Signed32(self.word()? as i32),
            6 => Value::Single(f32::from_bits(self.word()?)),
            7 => Value::Flag(match self.byte()? {
                0 => false,
                1 => true,
                other => {
                    return Err(Fault::Malformed(format!(
                        "this gguf holds {other} where a flag holds zero or one",
                    )));
                }
            }),
            8 => Value::Text(self.text()?),
            9 => {
                let held = self.word()?;
                let count = self.long()?;
                let mut values = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
                for _ in 0..count {
                    values.push(self.body(held, depth + 1)?);
                }
                Value::Array(values)
            }
            10 => Value::Unsigned64(self.long()?),
            11 => Value::Signed64(self.long()? as i64),
            12 => Value::Double(f64::from_bits(self.long()?)),
            other => {
                return Err(Fault::Malformed(format!(
                    "this gguf holds a metadata value of type {other}, and a gguf declares thirteen of them",
                )));
            }
        })
    }
}
