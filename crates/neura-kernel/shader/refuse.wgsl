fn refuse(kind: u32, code: u32) {
    atomicStore(&cursor[CURSOR_REFUSED], (kind << 16u) | (code + 1u));
}
