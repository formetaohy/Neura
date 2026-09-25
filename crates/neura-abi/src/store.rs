pub const TENSORS: u32 = 0;
pub const WEIGHTS: u32 = 1;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Store {
    Tensors,
    Weights,
}

impl Store {
    pub const ALL: &'static [Store] = &[Self::Tensors, Self::Weights];

    pub const fn code(self) -> u32 {
        match self {
            Self::Tensors => TENSORS,
            Self::Weights => WEIGHTS,
        }
    }
}
