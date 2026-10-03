use crate::module::Space;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TypeId(pub(crate) u32);

impl TypeId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(pub(crate) u32);

impl ValueId {
    pub const fn index(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Scalar {
    U32,
    I32,
    F32,
    F16,
    Bool,
}

impl Scalar {
    pub const U32_BYTES: u32 = 4;

    pub const fn name(self) -> &'static str {
        match self {
            Self::U32 => "u32",
            Self::I32 => "i32",
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bool => "bool",
        }
    }

    pub const fn bytes(self) -> u32 {
        match self {
            Self::U32 | Self::I32 | Self::F32 => 4,
            Self::F16 => 2,
            Self::Bool => panic!("a device boolean occupies no storage"),
        }
    }

    pub const fn floating(self) -> bool {
        matches!(self, Self::F32 | Self::F16)
    }

    pub const fn integer(self) -> bool {
        matches!(self, Self::U32 | Self::I32)
    }

    pub const fn signed(self) -> bool {
        matches!(self, Self::I32)
    }

    pub const fn storable(self) -> bool {
        !matches!(self, Self::Bool)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Member {
    pub name: String,
    pub ty: TypeId,
    pub offset: u32,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Type {
    Scalar(Scalar),
    Vector {
        scalar: Scalar,
        length: u32,
    },
    Array {
        element: TypeId,
        count: Option<u32>,
    },
    Struct {
        name: String,
        members: Vec<Member>,
        span: u32,
    },
    Atomic(Scalar),
    Pointer {
        space: Space,
        base: TypeId,
    },
    CooperativeMatrix {
        scalar: Scalar,
        rows: u32,
        columns: u32,
        usage: MatrixUse,
    },
}

impl Type {
    pub const fn scalar(&self) -> Option<Scalar> {
        match self {
            Self::Scalar(scalar) => Some(*scalar),
            Self::Atomic(scalar) => Some(*scalar),
            _ => None,
        }
    }

    pub const fn pointer(&self) -> Option<(Space, TypeId)> {
        match self {
            Self::Pointer { space, base } => Some((*space, *base)),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MatrixUse {
    A,
    B,
    Accumulator,
}

impl MatrixUse {
    pub const fn name(self) -> &'static str {
        match self {
            Self::A => "matrix a",
            Self::B => "matrix b",
            Self::Accumulator => "matrix accumulator",
        }
    }

    pub const fn code(self) -> u32 {
        match self {
            Self::A => 0,
            Self::B => 1,
            Self::Accumulator => 2,
        }
    }
}
