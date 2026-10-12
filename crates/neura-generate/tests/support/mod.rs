pub enum Meta<'a> {
    Text(&'a str),
    Texts(&'a [&'a str]),
    Words(&'a [u32]),
}

pub fn build(metadata: &[(&str, Meta<'_>)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
    for (key, value) in metadata {
        text(&mut bytes, key);
        meta(&mut bytes, value);
    }
    bytes
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
