#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Pool {
    Max,
    Mean,
}

impl Pool {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Max => "max",
            Self::Mean => "mean",
        }
    }
}
