mod arithmetic;
pub mod instruction;
mod literal;
pub mod module;
pub mod program;
pub mod spirv;
pub mod text;
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
    Backend, BindingKind, BindingSpec, ComputeProgram, MAX_BINDING_BYTES, METAL_SIZE_BUFFER_SLOT,
    ShaderBinding, ShaderTranslation, describe, reflect,
};
pub use text::Language;
pub use ty::{MatrixUse, Member, Scalar, Type, TypeId, ValueId};
