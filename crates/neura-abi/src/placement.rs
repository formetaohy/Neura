#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Placement {
    tensors: u64,
    weights: u64,
}

impl Placement {
    pub const fn new(tensors: u64, weights: u64) -> Self {
        Self { tensors, weights }
    }

    pub const fn tensors(self) -> u64 {
        self.tensors
    }

    pub const fn weights(self) -> u64 {
        self.weights
    }
}
