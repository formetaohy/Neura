fn refuse(kind: u32, code: u32) {
    atomicStore(&refusal[0u], (kind << 16u) | (code + 1u));
}
