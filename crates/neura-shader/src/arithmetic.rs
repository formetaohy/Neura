use crate::{BinaryOp, Scalar};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IntegerArithmetic {
    TruncatedQuotient,
    UnsignedQuotient,
    TruncatedRemainder,
    UnsignedRemainder,
}

impl BinaryOp {
    pub const fn integer_arithmetic(self, scalar: Scalar) -> Option<IntegerArithmetic> {
        match self {
            Self::Divide => Some(match scalar {
                Scalar::I32 => IntegerArithmetic::TruncatedQuotient,
                Scalar::U32 => IntegerArithmetic::UnsignedQuotient,
                _ => return None,
            }),
            Self::Modulo => Some(match scalar {
                Scalar::I32 => IntegerArithmetic::TruncatedRemainder,
                Scalar::U32 => IntegerArithmetic::UnsignedRemainder,
                _ => return None,
            }),
            _ => None,
        }
    }
}
