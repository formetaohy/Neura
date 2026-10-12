const ALIGNMENT: u64 = 32;

pub fn build(tensors: &[(&str, u32, &[u64], &[u8])]) -> Vec<u8> {
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
    bytes.extend_from_slice(&0u64.to_le_bytes());
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

pub fn singles(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|held| held.to_le_bytes()).collect()
}

pub fn quantized(blocks: usize) -> Vec<u8> {
    vec![0u8; blocks * 34]
}

fn text(bytes: &mut Vec<u8>, held: &str) {
    bytes.extend_from_slice(&(held.len() as u64).to_le_bytes());
    bytes.extend_from_slice(held.as_bytes());
}
