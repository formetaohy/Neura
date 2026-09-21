use std::fmt::Write as _;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Store {
    Tensors,
    Weights,
}

impl Store {
    pub const ALL: &'static [Store] = &[Self::Tensors, Self::Weights];

    pub const fn code(self) -> u32 {
        match self {
            Self::Tensors => 0,
            Self::Weights => 1,
        }
    }

    pub const fn constant(self) -> &'static str {
        match self {
            Self::Tensors => "STORE_TENSORS",
            Self::Weights => "STORE_WEIGHTS",
        }
    }
}

pub fn declarations() -> String {
    let mut out = String::new();
    for store in Store::ALL {
        writeln!(out, "const {}: u32 = {}u;", store.constant(), store.code()).unwrap();
    }
    out
}
