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
