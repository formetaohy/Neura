fn refuse(kind: u32, code: u32) {
    atomicStore(&cursor[CURSOR_REFUSED], (kind << 16u) | (code + 1u));
}

fn value_offset(flat: u32, dims: vec4<u32>, strides: vec4<u32>) -> u32 {
    let w = flat % dims.w;
    let z = (flat / dims.w) % dims.z;
    let y = (flat / (dims.w * dims.z)) % dims.y;
    let x = flat / (dims.w * dims.z * dims.y);
    return x * strides.x + y * strides.y + z * strides.z + w * strides.w;
}
