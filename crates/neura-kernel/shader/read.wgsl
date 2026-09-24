fn coordinates(flat: u32, dims: vec4<u32>) -> vec4<u32> {
    let w = flat % dims.w;
    let z = (flat / dims.w) % dims.z;
    let y = (flat / (dims.w * dims.z)) % dims.y;
    let x = flat / (dims.w * dims.z * dims.y);
    return vec4<u32>(x, y, z, w);
}

fn read_address(at: vec4<u32>, strides: vec4<u32>) -> u32 {
    return at.x * strides.x + at.y * strides.y + at.z * strides.z + at.w * strides.w;
}

fn component(at: vec4<u32>, axis: u32) -> u32 {
    return select(select(at.x, at.y, axis == 1u), select(at.z, at.w, axis == 3u), axis >= 2u);
}

fn whole_index(value: f32, rows: u32, kind: u32) -> u32 {
    if (trunc(value) != value || !(value >= 0.0) || !(value < f32(rows))) {
        refuse(kind, 0u);
        return 0u;
    }
    return u32(value);
}
