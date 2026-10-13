use neura_abi::Element;

pub fn kind(element: Element) -> u32 {
    match element {
        Element::Half => 1,
        Element::Q8_0 => 8,
        Element::Q4K => 12,
        Element::Q5K => 13,
        Element::Q6K => 14,
        other => panic!("{} storage carries no ggml type", other.name()),
    }
}

pub fn build(alignment: u64, tensors: &[(Element, &[u64], &[u8])]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&1u64.to_le_bytes());
    let key = "general.alignment";
    bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
    bytes.extend_from_slice(key.as_bytes());
    bytes.extend_from_slice(&4u32.to_le_bytes());
    bytes.extend_from_slice(&(alignment as u32).to_le_bytes());
    let mut at = 0u64;
    for (element, shape, payload) in tensors {
        let name = element.name();
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for dim in *shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
        bytes.extend_from_slice(&kind(*element).to_le_bytes());
        bytes.extend_from_slice(&at.to_le_bytes());
        at = (at + payload.len() as u64).next_multiple_of(alignment);
    }
    while !(bytes.len() as u64).is_multiple_of(alignment) {
        bytes.push(0);
    }
    for (_, _, payload) in tensors {
        bytes.extend_from_slice(payload);
        while !(bytes.len() as u64).is_multiple_of(alignment) {
            bytes.push(0);
        }
    }
    bytes
}
