use std::collections::BTreeMap;

pub enum Meta<'a> {
    Text(&'a str),
    Unsigned8(u8),
    Signed8(i8),
    Unsigned32(u32),
    Single(f32),
    Flag(bool),
    Unsigned64(u64),
    Double(f64),
    Array(&'a [Meta<'a>]),
}

pub struct Fixture {
    pub bytes: Vec<u8>,
    pub data: u64,
    pub at: BTreeMap<String, u64>,
}

pub fn build(
    alignment: u64,
    metadata: &[(&str, Meta<'_>)],
    tensors: &[(&str, u32, &[u64], &[u8])],
) -> Fixture {
    let mut offsets = Vec::new();
    let mut cursor = 0u64;
    for (_, _, _, payload) in tensors {
        offsets.push(cursor);
        cursor = (cursor + payload.len() as u64).next_multiple_of(alignment);
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        text(&mut bytes, key);
        meta(&mut bytes, value);
    }
    for (at, (name, kind, shape, _)) in tensors.iter().enumerate() {
        text(&mut bytes, name);
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in *shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&kind.to_le_bytes());
        bytes.extend_from_slice(&offsets[at].to_le_bytes());
    }
    while !(bytes.len() as u64).is_multiple_of(alignment) {
        bytes.push(0);
    }
    let data = bytes.len() as u64;
    let mut at = BTreeMap::new();
    for (index, (name, _, _, payload)) in tensors.iter().enumerate() {
        assert_eq!(bytes.len() as u64 - data, offsets[index]);
        at.insert((*name).to_owned(), offsets[index]);
        bytes.extend_from_slice(payload);
        while !(bytes.len() as u64).is_multiple_of(alignment) {
            bytes.push(0);
        }
    }
    Fixture { bytes, data, at }
}

fn meta(bytes: &mut Vec<u8>, value: &Meta<'_>) {
    bytes.extend_from_slice(&meta_kind(value).to_le_bytes());
    body(bytes, value);
}

fn meta_kind(value: &Meta<'_>) -> u32 {
    match value {
        Meta::Text(_) => 8,
        Meta::Unsigned8(_) => 0,
        Meta::Signed8(_) => 1,
        Meta::Unsigned32(_) => 4,
        Meta::Single(_) => 6,
        Meta::Flag(_) => 7,
        Meta::Unsigned64(_) => 10,
        Meta::Double(_) => 12,
        Meta::Array(_) => 9,
    }
}

fn body(bytes: &mut Vec<u8>, value: &Meta<'_>) {
    match value {
        Meta::Text(text) => self::text(bytes, text),
        Meta::Unsigned8(held) => bytes.push(*held),
        Meta::Signed8(held) => bytes.push(*held as u8),
        Meta::Unsigned32(held) => bytes.extend_from_slice(&held.to_le_bytes()),
        Meta::Single(held) => bytes.extend_from_slice(&held.to_bits().to_le_bytes()),
        Meta::Flag(held) => bytes.push(u8::from(*held)),
        Meta::Unsigned64(held) => bytes.extend_from_slice(&held.to_le_bytes()),
        Meta::Double(held) => bytes.extend_from_slice(&held.to_bits().to_le_bytes()),
        Meta::Array(values) => {
            let kind = values.first().map_or(0, meta_kind);
            bytes.extend_from_slice(&kind.to_le_bytes());
            bytes.extend_from_slice(&(values.len() as u64).to_le_bytes());
            for value in *values {
                body(bytes, value);
            }
        }
    }
}

fn text(bytes: &mut Vec<u8>, held: &str) {
    bytes.extend_from_slice(&(held.len() as u64).to_le_bytes());
    bytes.extend_from_slice(held.as_bytes());
}
