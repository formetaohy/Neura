mod arithmetic;
pub mod hlsl;
pub mod instruction;
mod literal;
pub mod module;
pub mod msl;
pub mod program;
pub mod spirv;
pub mod ty;
mod validate;

pub use arithmetic::IntegerArithmetic;
pub use instruction::{
    Address, AtomicOp, Barrier, BinaryOp, Block, BuiltIn, Constant, Instruction, MathFun,
    MatrixLayout, UnaryOp,
};
pub use module::{
    Access, Argument, Binding, Function, Global, Local, Module, Requirements, Space, Target,
    element_name,
};
pub use program::{
    Backend, BindingKind, BindingSpec, ComputeProgram, METAL_SIZE_BUFFER_SLOT, ShaderBinding,
    ShaderTranslation, describe, reflect,
};
pub use ty::{MatrixUse, Member, Scalar, Type, TypeId, ValueId};
