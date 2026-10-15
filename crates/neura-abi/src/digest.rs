const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const PRIME: u64 = 0x0000_0100_0000_01b3;

#[derive(Clone, Copy)]
pub struct Digest {
    state: u64,
}

impl Default for Digest {
    fn default() -> Self {
        Self::new()
    }
}

impl Digest {
    pub const fn new() -> Self {
        Self { state: BASIS }
    }

    pub fn word(&mut self, value: u32) -> &mut Self {
        self.state ^= u64::from(value);
        self.state = self.state.wrapping_mul(PRIME);
        self
    }

    pub fn long(&mut self, value: u64) -> &mut Self {
        self.word(value as u32);
        self.word((value >> 32) as u32);
        self
    }

    pub fn single(&mut self, value: f32) -> &mut Self {
        self.word(value.to_bits())
    }

    pub fn flag(&mut self, value: bool) -> &mut Self {
        self.word(u32::from(value))
    }

    pub fn length(&mut self, count: usize) -> &mut Self {
        self.word(count as u32)
    }

    pub fn text(&mut self, text: &str) -> &mut Self {
        self.length(text.len());
        for byte in text.bytes() {
            self.word(u32::from(byte));
        }
        self
    }

    pub const fn finish(&self) -> u64 {
        self.state
    }
}
