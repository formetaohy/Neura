use crate::ty::{TypeId, ValueId};

pub type Block = Vec<Instruction>;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Address {
    Global(u32),
    Local(u32),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BuiltIn {
    LocalInvocationIndex,
    WorkGroupId,
    GlobalInvocationId,
}

impl BuiltIn {
    pub const ALL: &'static [BuiltIn] = &[
        Self::LocalInvocationIndex,
        Self::WorkGroupId,
        Self::GlobalInvocationId,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::LocalInvocationIndex => "lid",
            Self::WorkGroupId => "group",
            Self::GlobalInvocationId => "global",
        }
    }

    pub const fn uniform(self) -> bool {
        matches!(self, Self::WorkGroupId | Self::GlobalInvocationId)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Constant {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Zero(TypeId),
}

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::U32(left), Self::U32(right)) => left == right,
            (Self::I32(left), Self::I32(right)) => left == right,
            (Self::F32(left), Self::F32(right)) => left.to_bits() == right.to_bits(),
            (Self::Bool(left), Self::Bool(right)) => left == right,
            (Self::Zero(left), Self::Zero(right)) => left == right,
            _ => false,
        }
    }
}

impl Eq for Constant {}

impl std::hash::Hash for Constant {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Self::U32(value) => value.hash(state),
            Self::I32(value) => value.hash(state),
            Self::F32(value) => value.to_bits().hash(state),
            Self::Bool(value) => value.hash(state),
            Self::Zero(ty) => ty.hash(state),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum UnaryOp {
    Negate,
    LogicalNot,
    BitwiseNot,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Xor,
    LogicalAnd,
    LogicalOr,
    ShiftLeft,
    ShiftRight,
}

impl BinaryOp {
    pub const ALL: &'static [BinaryOp] = &[
        Self::Add,
        Self::Subtract,
        Self::Multiply,
        Self::Divide,
        Self::Modulo,
        Self::Equal,
        Self::NotEqual,
        Self::Less,
        Self::LessEqual,
        Self::Greater,
        Self::GreaterEqual,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::LogicalAnd,
        Self::LogicalOr,
        Self::ShiftLeft,
        Self::ShiftRight,
    ];

    pub const fn comparison(self) -> bool {
        matches!(
            self,
            Self::Equal
                | Self::NotEqual
                | Self::Less
                | Self::LessEqual
                | Self::Greater
                | Self::GreaterEqual
        )
    }

    pub const fn logical(self) -> bool {
        matches!(self, Self::LogicalAnd | Self::LogicalOr)
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "+",
            Self::Subtract => "-",
            Self::Multiply => "*",
            Self::Divide => "/",
            Self::Modulo => "%",
            Self::Equal => "==",
            Self::NotEqual => "!=",
            Self::Less => "<",
            Self::LessEqual => "<=",
            Self::Greater => ">",
            Self::GreaterEqual => ">=",
            Self::And => "&",
            Self::Or => "|",
            Self::Xor => "^",
            Self::LogicalAnd => "&&",
            Self::LogicalOr => "||",
            Self::ShiftLeft => "<<",
            Self::ShiftRight => ">>",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MathFun {
    Min,
    Max,
    Abs,
    Floor,
    Trunc,
    Sqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Pow,
    UnpackHalf2x16,
}

impl MathFun {
    pub const ALL: &'static [MathFun] = &[
        Self::Min,
        Self::Max,
        Self::Abs,
        Self::Floor,
        Self::Trunc,
        Self::Sqrt,
        Self::Exp,
        Self::Log,
        Self::Sin,
        Self::Cos,
        Self::Tanh,
        Self::Pow,
        Self::UnpackHalf2x16,
    ];

    pub const fn arity(self) -> usize {
        match self {
            Self::Min | Self::Max | Self::Pow => 2,
            _ => 1,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::Min => "min",
            Self::Max => "max",
            Self::Abs => "abs",
            Self::Floor => "floor",
            Self::Trunc => "trunc",
            Self::Sqrt => "sqrt",
            Self::Exp => "exp",
            Self::Log => "log",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::Tanh => "tanh",
            Self::Pow => "pow",
            Self::UnpackHalf2x16 => "unpack2x16float",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AtomicOp {
    Add,
    Subtract,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exchange,
}

impl AtomicOp {
    pub const ALL: &'static [AtomicOp] = &[
        Self::Add,
        Self::Subtract,
        Self::Min,
        Self::Max,
        Self::And,
        Self::Or,
        Self::Xor,
        Self::Exchange,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Subtract => "subtract",
            Self::Min => "min",
            Self::Max => "max",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Exchange => "exchange",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Barrier {
    WorkGroup,
    Storage,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

#[derive(Clone, PartialEq, Hash, Debug)]
pub enum Instruction {
    Argument {
        index: u32,
        result: ValueId,
    },
    Address {
        address: Address,
        result: ValueId,
    },
    Access {
        base: ValueId,
        index: ValueId,
        result: ValueId,
    },
    AccessIndex {
        base: ValueId,
        index: u32,
        result: ValueId,
    },
    Load {
        pointer: ValueId,
        result: ValueId,
    },
    Store {
        pointer: ValueId,
        value: ValueId,
    },
    Unary {
        op: UnaryOp,
        value: ValueId,
        result: ValueId,
    },
    Binary {
        op: BinaryOp,
        left: ValueId,
        right: ValueId,
        result: ValueId,
    },
    Select {
        condition: ValueId,
        accept: ValueId,
        reject: ValueId,
        result: ValueId,
    },
    Convert {
        value: ValueId,
        result: ValueId,
    },
    Bitcast {
        value: ValueId,
        result: ValueId,
    },
    Math {
        fun: MathFun,
        arguments: Vec<ValueId>,
        result: ValueId,
    },
    Compose {
        constituents: Vec<ValueId>,
        result: ValueId,
    },
    Call {
        function: u32,
        arguments: Vec<ValueId>,
        result: Option<ValueId>,
    },
    Atomic {
        op: AtomicOp,
        pointer: ValueId,
        value: ValueId,
        result: ValueId,
    },
    WorkGroupUniformLoad {
        pointer: ValueId,
        result: ValueId,
    },
    Barrier(Barrier),
    MatrixFill {
        value: ValueId,
        result: ValueId,
    },
    MatrixLoad {
        pointer: ValueId,
        stride: ValueId,
        layout: MatrixLayout,
        result: ValueId,
    },
    MatrixStore {
        pointer: ValueId,
        value: ValueId,
        stride: ValueId,
        layout: MatrixLayout,
    },
    MatrixMulAdd {
        left: ValueId,
        right: ValueId,
        accumulate: ValueId,
        result: ValueId,
    },
    MatrixLength {
        ty: TypeId,
        result: ValueId,
    },
    MatrixExtract {
        value: ValueId,
        index: ValueId,
        result: ValueId,
    },
    MatrixInsert {
        value: ValueId,
        index: ValueId,
        component: ValueId,
        result: ValueId,
    },
    If {
        condition: ValueId,
        accept: Block,
        reject: Block,
    },
    Switch {
        selector: ValueId,
        cases: Vec<(u32, Block)>,
        default: Block,
    },
    Loop {
        body: Block,
        continuing: Block,
    },
    Break,
    Continue,
    Return {
        value: Option<ValueId>,
    },
    Block(Block),
}

pub(crate) fn leaves_a_loop(body: &[Instruction]) -> bool {
    body.iter().any(Instruction::leaves_a_loop)
}

impl Instruction {
    pub(crate) fn leaves_a_loop(&self) -> bool {
        match self {
            Self::Break => true,
            Self::If { accept, reject, .. } => leaves_a_loop(accept) || leaves_a_loop(reject),
            Self::Switch { cases, default, .. } => {
                cases.iter().any(|(_, body)| leaves_a_loop(body)) || leaves_a_loop(default)
            }
            Self::Block(body) => leaves_a_loop(body),
            Self::Loop { .. } => false,
            _ => false,
        }
    }

    pub const fn result(&self) -> Option<ValueId> {
        match self {
            Self::Argument { result, .. }
            | Self::Address { result, .. }
            | Self::Access { result, .. }
            | Self::AccessIndex { result, .. }
            | Self::Load { result, .. }
            | Self::Unary { result, .. }
            | Self::Binary { result, .. }
            | Self::Select { result, .. }
            | Self::Convert { result, .. }
            | Self::Bitcast { result, .. }
            | Self::Math { result, .. }
            | Self::Compose { result, .. }
            | Self::Atomic { result, .. }
            | Self::WorkGroupUniformLoad { result, .. }
            | Self::MatrixFill { result, .. }
            | Self::MatrixLoad { result, .. }
            | Self::MatrixMulAdd { result, .. }
            | Self::MatrixLength { result, .. }
            | Self::MatrixExtract { result, .. }
            | Self::MatrixInsert { result, .. } => Some(*result),
            Self::Call { result, .. } => *result,
            Self::Store { .. }
            | Self::Barrier(_)
            | Self::MatrixStore { .. }
            | Self::If { .. }
            | Self::Switch { .. }
            | Self::Loop { .. }
            | Self::Break
            | Self::Continue
            | Self::Return { .. }
            | Self::Block(_) => None,
        }
    }

    pub fn operands(&self) -> Vec<ValueId> {
        match self {
            Self::Argument { .. }
            | Self::Address { .. }
            | Self::Barrier(_)
            | Self::Break
            | Self::Continue
            | Self::Loop { .. }
            | Self::MatrixLength { .. }
            | Self::Block(_) => Vec::new(),
            Self::Access { base, index, .. } => vec![*base, *index],
            Self::AccessIndex { base, .. } | Self::Load { pointer: base, .. } => vec![*base],
            Self::Store { pointer, value } => vec![*pointer, *value],
            Self::Unary { value, .. }
            | Self::Convert { value, .. }
            | Self::Bitcast { value, .. }
            | Self::MatrixFill { value, .. }
            | Self::WorkGroupUniformLoad { pointer: value, .. } => vec![*value],
            Self::Binary { left, right, .. } => vec![*left, *right],
            Self::Select {
                condition,
                accept,
                reject,
                ..
            } => vec![*condition, *accept, *reject],
            Self::Math { arguments, .. }
            | Self::Compose {
                constituents: arguments,
                ..
            }
            | Self::Call { arguments, .. } => arguments.clone(),
            Self::Atomic { pointer, value, .. } => vec![*pointer, *value],
            Self::MatrixLoad {
                pointer, stride, ..
            } => vec![*pointer, *stride],
            Self::MatrixStore {
                pointer,
                value,
                stride,
                ..
            } => vec![*pointer, *value, *stride],
            Self::MatrixMulAdd {
                left,
                right,
                accumulate,
                ..
            } => vec![*left, *right, *accumulate],
            Self::MatrixExtract { value, index, .. } => vec![*value, *index],
            Self::MatrixInsert {
                value,
                index,
                component,
                ..
            } => vec![*value, *index, *component],
            Self::If { condition, .. } => vec![*condition],
            Self::Switch { selector, .. } => vec![*selector],
            Self::Return { value } => value.iter().copied().collect(),
        }
    }

    pub const fn declares(&self) -> bool {
        matches!(self, Self::Argument { .. } | Self::Address { .. })
    }
}
