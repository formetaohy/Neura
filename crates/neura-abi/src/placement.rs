#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Placement {
    heap: u64,
    weights: u64,
    tensors: u64,
}

impl Placement {
    pub const fn new(heap: u64, weights: u64, tensors: u64) -> Self {
        Self {
            heap,
            weights,
            tensors,
        }
    }

    pub const fn heap(self) -> u64 {
        self.heap
    }

    pub const fn weights(self) -> u64 {
        self.weights
    }

    pub const fn tensors(self) -> u64 {
        self.tensors
    }
}
