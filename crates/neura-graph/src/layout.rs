use crate::Shape;
use neura_abi::MAX_RANK;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Layout {
    Dense,
    Ranked([u8; 4]),
    Strided { strides: [u32; 4], origin: u64 },
}

impl Layout {
    pub fn strided(dims: [u32; 4], strides: [u32; 4], origin: u64) -> Self {
        let mut held = strides;
        for axis in 0..MAX_RANK as usize {
            if dims[axis] == 1 {
                held[axis] = 0;
            }
        }
        Self::Strided {
            strides: held,
            origin,
        }
    }

    pub fn strides(self, dims: [u32; 4]) -> [u32; 4] {
        match self {
            Self::Dense => Shape::dense_strides(dims),
            Self::Ranked(rank) => Shape::ranked_strides(dims, rank),
            Self::Strided { strides, .. } => strides,
        }
    }

    pub fn rank(self) -> [u8; 4] {
        match self {
            Self::Ranked(rank) => rank,
            _ => [0, 1, 2, 3],
        }
    }

    pub fn origin(self) -> u64 {
        match self {
            Self::Strided { origin, .. } => origin,
            _ => 0,
        }
    }

    pub fn strided_parts(self) -> Option<([u32; 4], u64)> {
        match self {
            Self::Strided { strides, origin } => Some((strides, origin)),
            _ => None,
        }
    }
}
