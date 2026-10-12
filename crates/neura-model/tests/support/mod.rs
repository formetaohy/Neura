pub enum Meta<'a> {
    Text(&'a str),
    Unsigned32(u32),
    Single(f32),
    Texts(&'a [&'a str]),
    Words(&'a [u32]),
}

const ALIGNMENT: u64 = 32;

pub fn build(metadata: &[(&str, Meta<'_>)], tensors: &[(&str, u32, &[u64], &[u8])]) -> Vec<u8> {
    let mut offsets = Vec::new();
    let mut cursor = 0u64;
    for (_, _, _, payload) in tensors {
        offsets.push(cursor);
        cursor = (cursor + payload.len() as u64).next_multiple_of(ALIGNMENT);
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
    while !(bytes.len() as u64).is_multiple_of(ALIGNMENT) {
        bytes.push(0);
    }
    let data = bytes.len() as u64;
    for (at, (_, _, _, payload)) in tensors.iter().enumerate() {
        assert_eq!(
            offsets[at],
            bytes.len() as u64 - data,
            "a fixture lays its tensors out in the order it names them",
        );
        bytes.extend_from_slice(payload);
        while !(bytes.len() as u64).is_multiple_of(ALIGNMENT) {
            bytes.push(0);
        }
    }
    bytes
}

pub fn singles(elements: u64) -> Vec<u8> {
    vec![0u8; elements as usize * 4]
}

fn text(bytes: &mut Vec<u8>, held: &str) {
    bytes.extend_from_slice(&(held.len() as u64).to_le_bytes());
    bytes.extend_from_slice(held.as_bytes());
}

fn meta(bytes: &mut Vec<u8>, value: &Meta<'_>) {
    match value {
        Meta::Text(held) => {
            bytes.extend_from_slice(&8u32.to_le_bytes());
            text(bytes, held);
        }
        Meta::Unsigned32(held) => {
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&held.to_le_bytes());
        }
        Meta::Single(held) => {
            bytes.extend_from_slice(&6u32.to_le_bytes());
            bytes.extend_from_slice(&held.to_le_bytes());
        }
        Meta::Texts(held) => {
            bytes.extend_from_slice(&9u32.to_le_bytes());
            bytes.extend_from_slice(&8u32.to_le_bytes());
            bytes.extend_from_slice(&(held.len() as u64).to_le_bytes());
            for text_held in *held {
                text(bytes, text_held);
            }
        }
        Meta::Words(held) => {
            bytes.extend_from_slice(&9u32.to_le_bytes());
            bytes.extend_from_slice(&4u32.to_le_bytes());
            bytes.extend_from_slice(&(held.len() as u64).to_le_bytes());
            for word in *held {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
    }
}
