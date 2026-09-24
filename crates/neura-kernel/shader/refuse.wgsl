fn refuse(slot: u32, kind: u32, code: u32) {
    let word = (kind << 16u) | (code + 1u);
    loop {
        let verdict = atomicCompareExchangeWeak(&refusal[slot], 0u, word);
        if (verdict.exchanged || verdict.old_value != 0u) {
            break;
        }
    }
    loop {
        let verdict = atomicCompareExchangeWeak(&refusal[0u], 0u, word);
        if (verdict.exchanged || verdict.old_value != 0u) {
            break;
        }
    }
}
