use super::{FunctionLower, Symbol, Typed};
use crate::{Compiler, DeviceInstruction, ir};
use neura_shader::{
    AtomicOp, BinaryOp, Constant, MatrixLayout, MatrixUse, Scalar, Type, TypeId, UnaryOp,
};

impl Compiler {
    pub(crate) fn rust_type(&mut self, ty: &ir::Type) -> TypeId {
        match ty {
            ir::Type::Named(name) => match name.as_str() {
                "f16" => self.module_mut().scalar(Scalar::F16),
                "coopmat_a" => self.cooperative(MatrixUse::A),
                "coopmat_b" => self.cooperative(MatrixUse::B),
                "coopmat_accumulator" => self.cooperative(MatrixUse::Accumulator),
                other => self.ty(other),
            },
            ir::Type::Array { element, length } => {
                let element = self.rust_type(element);
                let count = self.evaluate(length);
                self.array(element, count)
            }
        }
    }

    fn cooperative(&mut self, usage: MatrixUse) -> TypeId {
        let rows = self.constant_u32("COOPMAT_ROWS");
        let columns = self.constant_u32("COOPMAT_COLUMNS");
        let depth = self.constant_u32("COOPMAT_DEPTH");
        let (scalar, rows, columns) = match usage {
            MatrixUse::A => (Scalar::F16, rows, depth),
            MatrixUse::B => (Scalar::F16, depth, columns),
            MatrixUse::Accumulator => (Scalar::F32, rows, columns),
        };
        self.module_mut()
            .cooperative_matrix(scalar, rows, columns, usage)
    }

    pub(crate) fn constant_u32(&self, name: &str) -> u32 {
        self.constant_value(name)
            .unwrap_or_else(|| panic!("the device constant {name} is not defined"))
    }

    pub(crate) fn evaluate(&self, expr: &ir::Expression) -> u32 {
        use ir::{BinaryOperator as Op, Expression as E, IntegerType};
        match expr {
            E::Integer { value, ty } if !matches!(ty, IntegerType::Signed) => *value,
            E::Name(name) => self.constant_u32(name),
            E::Binary { op, left, right } => {
                let left = self.evaluate(left);
                let right = self.evaluate(right);
                match op {
                    Op::Add => left.checked_add(right).expect("device constant overflows"),
                    Op::Subtract => left.checked_sub(right).expect("device constant underflows"),
                    Op::Multiply => left.checked_mul(right).expect("device constant overflows"),
                    Op::Divide => left
                        .checked_div(right)
                        .expect("device constant divides by zero"),
                    Op::Modulo => left
                        .checked_rem(right)
                        .expect("device constant divides by zero"),
                    _ => panic!("unsupported device constant expression"),
                }
            }
            other => panic!("a device constant must be known at compilation: {other:?}"),
        }
    }

    pub(crate) fn member(&self, ty: TypeId, field: &str) -> (u32, TypeId) {
        match self.module().ty(ty) {
            Type::Struct { name, members, .. } => members
                .iter()
                .enumerate()
                .find(|(_, member)| member.name == field)
                .map(|(index, member)| (index as u32, member.ty))
                .unwrap_or_else(|| panic!("Rust device struct {name} has no field {field}")),
            other => panic!(
                "a field of {} is not accessible",
                neura_shader::element_name(other)
            ),
        }
    }
}

impl FunctionLower<'_> {
    fn literal(&mut self, value: u32, kind: ir::IntegerType, hint: Option<TypeId>) -> Typed {
        let signed = matches!(kind, ir::IntegerType::Signed)
            || matches!(kind, ir::IntegerType::Inferred)
                && hint.is_some_and(|ty| ty == self.compiler.scalar("i32"));
        let ty = if signed {
            self.compiler.scalar("i32")
        } else {
            self.compiler.scalar("u32")
        };
        let constant = if signed {
            Constant::I32(i32::try_from(value).expect("a signed device integer fits in i32"))
        } else {
            Constant::U32(value)
        };
        let value = self.compiler.module_mut().constant(constant);
        Typed { value, ty }
    }

    fn field(&mut self, base: Typed, name: &str) -> Typed {
        match self.compiler.module().ty(base.ty).clone() {
            Type::Struct { .. } => {
                let (index, ty) = self.compiler.member(base.ty, name);
                self.emit(ty, |result| DeviceInstruction::AccessIndex {
                    base: base.value,
                    index,
                    result,
                })
            }
            Type::Vector { scalar, length } => {
                let index = match name {
                    "x" => 0,
                    "y" => 1,
                    "z" => 2,
                    "w" => 3,
                    _ => panic!("Rust device vector has no component {name}"),
                };
                assert!(index < length, "a device vector has no {name} lane");
                let ty = self.compiler.module_mut().scalar(scalar);
                self.emit(ty, |result| DeviceInstruction::AccessIndex {
                    base: base.value,
                    index,
                    result,
                })
            }
            other => panic!(
                "a field of {} is not accessible",
                neura_shader::element_name(&other)
            ),
        }
    }

    fn element_type(&self, ty: TypeId) -> TypeId {
        match self.compiler.module().ty(ty).clone() {
            Type::Array { element, .. } => element,
            Type::Vector { scalar, .. } => self
                .compiler
                .module()
                .scalar_type(scalar)
                .expect("a device vector lane exists"),
            other => panic!(
                "a Rust device indexes {} instead of an array or vector",
                neura_shader::element_name(&other)
            ),
        }
    }

    fn indexed(&mut self, base: Typed, index: &ir::Expression) -> Typed {
        let element = self.element_type(base.ty);
        if matches!(self.compiler.module().ty(base.ty), Type::Vector { .. }) {
            let index = self
                .evaluate_u32(index)
                .expect("a device vector requires a compile-time lane index");
            let length = match self.compiler.module().ty(base.ty) {
                Type::Vector { length, .. } => *length,
                _ => unreachable!(),
            };
            assert!(index < length, "a vector index lies inside its lanes");
            self.emit(element, |result| DeviceInstruction::AccessIndex {
                base: base.value,
                index,
                result,
            })
        } else {
            let index = self.value(index);
            self.emit(element, |result| DeviceInstruction::Access {
                base: base.value,
                index: index.value,
                result,
            })
        }
    }

    pub(super) fn place(&mut self, expr: &ir::Expression) -> Option<Typed> {
        use ir::Expression as E;
        match expr {
            E::Name(name) => match self.lookup(name) {
                Some(Symbol::Local(local)) => Some(local),
                Some(Symbol::Value(_) | Symbol::Array(_) | Symbol::Constant(_)) => None,
                None if self.compiler.declares_global(name) => {
                    let global = self.compiler.workgroup_global(name);
                    let ty = self.compiler.module().global(global).ty;
                    let space = self.compiler.module().global(global).space;
                    let pointer = self.compiler.pointer(space, ty);
                    Some(self.emit(pointer, |result| DeviceInstruction::Address {
                        address: neura_shader::Address::Global(global),
                        result,
                    }))
                }
                None => panic!("Rust device name {name} is not defined"),
            },
            E::Field { base, name } => {
                let base = self.place(base)?;
                let pointee = pointee(self, base.ty);
                let (index, ty) = self.compiler.member(pointee, name);
                let pointer = self.compiler.pointer(pointer_space(self, base.ty), ty);
                Some(self.emit(pointer, |result| DeviceInstruction::AccessIndex {
                    base: base.value,
                    index,
                    result,
                }))
            }
            E::Index { base, index } => {
                if let E::Name(name) = base.as_ref()
                    && let Some(Symbol::Array(registers)) = self.lookup(name)
                {
                    let offset = self
                        .evaluate_u32(index)
                        .expect("a scalar register requires a compile-time index");
                    return Some(
                        *registers
                            .get(offset as usize)
                            .expect("a scalar register index lies inside its array"),
                    );
                }
                let base = self.place(base)?;
                let space = pointer_space(self, base.ty);
                let pointee = pointee(self, base.ty);
                let element = self.element_type(pointee);
                let pointer = self.compiler.pointer(space, element);
                if matches!(self.compiler.module().ty(pointee), Type::Vector { .. }) {
                    let offset = self
                        .evaluate_u32(index)
                        .expect("a device vector requires a compile-time lane index");
                    return Some(self.emit(pointer, |result| DeviceInstruction::AccessIndex {
                        base: base.value,
                        index: offset,
                        result,
                    }));
                }
                let index = self.value(index);
                Some(self.emit(pointer, |result| DeviceInstruction::Access {
                    base: base.value,
                    index: index.value,
                    result,
                }))
            }
            _ => None,
        }
    }

    pub(super) fn value(&mut self, expr: &ir::Expression) -> Typed {
        self.value_with_hint(expr, None)
    }

    pub(super) fn value_with_hint(&mut self, expr: &ir::Expression, hint: Option<TypeId>) -> Typed {
        use ir::Expression as E;
        if matches!(expr, E::Name(_) | E::Field { .. } | E::Index { .. })
            && let Some(place) = self.place(expr)
        {
            let ty = self.compiler.module().loaded_ty(place.ty);
            let loaded = self.emit(ty, |result| DeviceInstruction::Load {
                pointer: place.value,
                result,
            });
            if let Some(ty) = hint {
                assert_eq!(loaded.ty, ty, "device operand has a different type");
            }
            return loaded;
        }
        let value = match expr {
            E::Integer { value, ty } => self.literal(*value, *ty, hint),
            E::Float(value) => {
                let ty = self.compiler.scalar("f32");
                let value = self.compiler.module_mut().constant(Constant::F32(*value));
                Typed { value, ty }
            }
            E::Bool(value) => {
                let ty = self.compiler.scalar("bool");
                let value = self.compiler.module_mut().constant(Constant::Bool(*value));
                Typed { value, ty }
            }
            E::Name(name) => match self.lookup(name) {
                Some(Symbol::Value(value)) => value,
                Some(Symbol::Constant(value)) => {
                    let ty = self.compiler.scalar("u32");
                    let value = self.compiler.module_mut().constant(Constant::U32(value));
                    Typed { value, ty }
                }
                Some(Symbol::Array(_)) => {
                    panic!("a scalar register array is accessed by constant index")
                }
                Some(Symbol::Local(_)) => unreachable!("a local is a device place"),
                None => panic!("Rust device name {name} is not defined"),
            },
            E::Field { base, name } => {
                let base = self.value(base);
                self.field(base, name)
            }
            E::Index { base, index } => {
                let base = self.value(base);
                self.indexed(base, index)
            }
            E::Unary { op, value } => {
                let arg = self.value(value);
                let scalar = self.scalar_of(arg.ty);
                let op = match op {
                    ir::UnaryOperator::Negate => UnaryOp::Negate,
                    ir::UnaryOperator::Not if scalar == Scalar::Bool => UnaryOp::LogicalNot,
                    ir::UnaryOperator::Not => UnaryOp::BitwiseNot,
                };
                self.emit(arg.ty, |result| DeviceInstruction::Unary {
                    op,
                    value: arg.value,
                    result,
                })
            }
            E::Binary { op, left, right } => self.binary(*op, left, right),
            E::Call { name, arguments } => self.call(name, arguments),
            E::Repeat { value, length } => {
                let count = self.compiler.evaluate(length);
                let value = self.value(value);
                assert!(
                    matches!(
                        self.compiler.module().constant_of(value.value),
                        Some(Constant::F32(0.0)) | Some(Constant::U32(0))
                    ),
                    "a device array starts from zero"
                );
                let ty = self.compiler.array(value.ty, count);
                let value = self.compiler.module_mut().constant(Constant::Zero(ty));
                Typed { value, ty }
            }
            E::Cast { value, ty } => {
                let source = self.value(value);
                let target = self.rust_type(ty);
                self.emit(target, |result| DeviceInstruction::Convert {
                    value: source.value,
                    result,
                })
            }
            E::Reference(_) => panic!("a device reference is only passed to an atomic store"),
        };
        if let Some(ty) = hint {
            assert_eq!(value.ty, ty, "device operand has a different type");
        }
        value
    }

    pub(super) fn scalar_of(&self, ty: TypeId) -> Scalar {
        self.compiler
            .module()
            .ty(ty)
            .scalar()
            .unwrap_or_else(|| panic!("a device operand is not a scalar"))
    }

    pub(super) fn evaluate_u32(&self, expr: &ir::Expression) -> Option<u32> {
        use ir::{BinaryOperator as Op, Expression as E, IntegerType};
        match expr {
            E::Integer { value, ty } if !matches!(ty, IntegerType::Signed) => Some(*value),
            E::Name(name) => match self.lookup(name)? {
                Symbol::Constant(value) => Some(value),
                _ => None,
            },
            E::Binary { op, left, right } => {
                let left = self.evaluate_u32(left)?;
                let right = self.evaluate_u32(right)?;
                match op {
                    Op::Add => Some(left.checked_add(right).expect("device constant overflows")),
                    Op::Subtract => {
                        Some(left.checked_sub(right).expect("device constant underflows"))
                    }
                    Op::Multiply => {
                        Some(left.checked_mul(right).expect("device constant overflows"))
                    }
                    Op::Divide => Some(
                        left.checked_div(right)
                            .expect("device constant divides by zero"),
                    ),
                    Op::Modulo => Some(
                        left.checked_rem(right)
                            .expect("device constant divides by zero"),
                    ),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn binary(
        &mut self,
        operator: ir::BinaryOperator,
        left: &ir::Expression,
        right: &ir::Expression,
    ) -> Typed {
        use ir::BinaryOperator as Op;
        let op = match operator {
            Op::Add => BinaryOp::Add,
            Op::Subtract => BinaryOp::Subtract,
            Op::Multiply => BinaryOp::Multiply,
            Op::Divide => BinaryOp::Divide,
            Op::Modulo => BinaryOp::Modulo,
            Op::Equal => BinaryOp::Equal,
            Op::NotEqual => BinaryOp::NotEqual,
            Op::Less => BinaryOp::Less,
            Op::LessEqual => BinaryOp::LessEqual,
            Op::Greater => BinaryOp::Greater,
            Op::GreaterEqual => BinaryOp::GreaterEqual,
            Op::BitAnd => BinaryOp::And,
            Op::BitOr => BinaryOp::Or,
            Op::BitXor => BinaryOp::Xor,
            Op::LogicalAnd => BinaryOp::LogicalAnd,
            Op::LogicalOr => BinaryOp::LogicalOr,
            Op::ShiftLeft => BinaryOp::ShiftLeft,
            Op::ShiftRight => BinaryOp::ShiftRight,
        };
        if op.logical() {
            let boolean = self.compiler.scalar("bool");
            let left = self.value_with_hint(left, Some(boolean));
            let name = format!("branch{}", self.function.locals.len());
            let local = self.declare_local(&name, boolean);
            self.push(DeviceInstruction::Store {
                pointer: local.value,
                value: left.value,
            });
            let branch = self.block(|lower| {
                let right = lower.value_with_hint(right, Some(boolean));
                lower.push(DeviceInstruction::Store {
                    pointer: local.value,
                    value: right.value,
                });
            });
            let empty = Vec::new();
            let (accept, reject) = if op == BinaryOp::LogicalAnd {
                (branch, empty)
            } else {
                (empty, branch)
            };
            self.push(DeviceInstruction::If {
                condition: left.value,
                accept,
                reject,
            });
            return self.emit(boolean, |result| DeviceInstruction::Load {
                pointer: local.value,
                result,
            });
        }
        let left = self.value(left);
        let right = if op == BinaryOp::Multiply
            && matches!(self.compiler.module().ty(left.ty), Type::Vector { .. })
        {
            self.value(right)
        } else if matches!(op, BinaryOp::ShiftLeft | BinaryOp::ShiftRight) {
            self.value_with_hint(right, Some(self.compiler.scalar("u32")))
        } else {
            self.value_with_hint(right, Some(left.ty))
        };
        let ty = if op.comparison() {
            self.compiler.scalar("bool")
        } else {
            left.ty
        };
        self.emit(ty, |result| DeviceInstruction::Binary {
            op,
            left: left.value,
            right: right.value,
            result,
        })
    }

    pub(super) fn reference(&mut self, argument: &ir::Expression, subject: &str) -> Typed {
        let ir::Expression::Reference(reference) = argument else {
            panic!("{subject} takes a reference");
        };
        self.place(reference)
            .unwrap_or_else(|| panic!("{subject} refers to a writable device place"))
    }

    fn call(&mut self, name: &str, args: &[ir::Expression]) -> Typed {
        match name {
            "uvec4" => {
                assert_eq!(args.len(), 4);
                let ty = self.compiler.scalar("u32");
                let constituents = args
                    .iter()
                    .map(|arg| self.value_with_hint(arg, Some(ty)).value)
                    .collect::<Vec<_>>();
                let ty = self.compiler.ty("uvec4");
                self.emit(ty, |result| DeviceInstruction::Compose {
                    constituents,
                    result,
                })
            }
            "select" => {
                assert_eq!(args.len(), 3);
                let reject = self.value(&args[0]);
                let accept = self.value_with_hint(&args[1], Some(reject.ty));
                let condition = self.value_with_hint(&args[2], Some(self.compiler.scalar("bool")));
                self.emit(reject.ty, |result| DeviceInstruction::Select {
                    condition: condition.value,
                    accept: accept.value,
                    reject: reject.value,
                    result,
                })
            }
            "bitcast_u32" => {
                assert_eq!(args.len(), 1);
                let source = self.value_with_hint(&args[0], Some(self.compiler.scalar("f32")));
                let ty = self.compiler.scalar("u32");
                self.emit(ty, |result| DeviceInstruction::Bitcast {
                    value: source.value,
                    result,
                })
            }
            "bitcast_f32" => {
                assert_eq!(args.len(), 1);
                let source = self.value_with_hint(&args[0], Some(self.compiler.scalar("u32")));
                let ty = self.compiler.scalar("f32");
                self.emit(ty, |result| DeviceInstruction::Bitcast {
                    value: source.value,
                    result,
                })
            }
            "f32" | "i32" | "u32" | "f16" => {
                assert_eq!(args.len(), 1);
                let source = self.value(&args[0]);
                let ty = match name {
                    "f16" => self.compiler.module_mut().scalar(Scalar::F16),
                    other => self.compiler.ty(other),
                };
                self.emit(ty, |result| DeviceInstruction::Convert {
                    value: source.value,
                    result,
                })
            }
            "unpack2x16float" => {
                assert_eq!(args.len(), 1);
                let source = self.value_with_hint(&args[0], Some(self.compiler.scalar("u32")));
                let ty = self.compiler.ty("fvec2");
                self.emit(ty, |result| DeviceInstruction::Math {
                    fun: neura_shader::MathFun::UnpackHalf2x16,
                    arguments: vec![source.value],
                    result,
                })
            }
            "max" | "min" | "abs" | "sqrt" | "exp" | "log" | "tanh" | "trunc" | "sin" | "cos"
            | "pow" | "floor" => {
                let fun = match name {
                    "max" => neura_shader::MathFun::Max,
                    "min" => neura_shader::MathFun::Min,
                    "abs" => neura_shader::MathFun::Abs,
                    "sqrt" => neura_shader::MathFun::Sqrt,
                    "exp" => neura_shader::MathFun::Exp,
                    "log" => neura_shader::MathFun::Log,
                    "tanh" => neura_shader::MathFun::Tanh,
                    "trunc" => neura_shader::MathFun::Trunc,
                    "sin" => neura_shader::MathFun::Sin,
                    "cos" => neura_shader::MathFun::Cos,
                    "pow" => neura_shader::MathFun::Pow,
                    "floor" => neura_shader::MathFun::Floor,
                    _ => unreachable!(),
                };
                assert_eq!(args.len(), fun.arity());
                let first = self.value(&args[0]);
                let mut arguments = vec![first.value];
                if let Some(second) = args.get(1) {
                    arguments.push(self.value_with_hint(second, Some(first.ty)).value);
                }
                self.emit(first.ty, |result| DeviceInstruction::Math {
                    fun,
                    arguments,
                    result,
                })
            }
            "atomic_add" | "atomic_sub" => self.atomic(name, args),
            "workgroup_uniform_load" => self.workgroup_uniform_load(args),
            "coopmat_accumulator" => {
                assert_eq!(args.len(), 1);
                let value = self.value_with_hint(&args[0], Some(self.compiler.scalar("f32")));
                let ty = self.coopmat(ir::Type::Named("coopmat_accumulator".to_owned()));
                self.emit(ty, |result| DeviceInstruction::MatrixFill {
                    value: value.value,
                    result,
                })
            }
            "coopmat_load_row" => self.matrix_load(args, MatrixLayout::RowMajor),
            "coopmat_load_column" => self.matrix_load(args, MatrixLayout::ColumnMajor),
            "coopmat_muladd" => {
                assert_eq!(args.len(), 3);
                let accumulate = self.value(&args[2]);
                let left = self.value_with_hint(&args[0], None);
                let right = self.value_with_hint(&args[1], None);
                self.emit(accumulate.ty, |result| DeviceInstruction::MatrixMulAdd {
                    left: left.value,
                    right: right.value,
                    accumulate: accumulate.value,
                    result,
                })
            }
            "workgroup_barrier" | "storage_barrier" | "atomic_store" | "coopmat_store" => {
                panic!("a device synchronization or store cannot yield a value")
            }
            _ => {
                let function = self.compiler.lower_callee(name);
                let (parameters, result) = {
                    let callee = &self.compiler.module().functions()[function as usize];
                    (
                        callee
                            .arguments
                            .iter()
                            .map(|argument| argument.ty)
                            .collect::<Vec<_>>(),
                        callee.result,
                    )
                };
                assert_eq!(
                    args.len(),
                    parameters.len(),
                    "{name} takes a fixed number of arguments"
                );
                let arguments = args
                    .iter()
                    .zip(parameters)
                    .map(|(arg, ty)| self.value_with_hint(arg, Some(ty)).value)
                    .collect();
                let ty = result.expect("a void function cannot be used as a value");
                self.emit(ty, |value| DeviceInstruction::Call {
                    function,
                    arguments,
                    result: Some(value),
                })
            }
        }
    }

    fn coopmat(&mut self, ty: ir::Type) -> TypeId {
        self.compiler.rust_type(&ty)
    }

    fn matrix_load(&mut self, args: &[ir::Expression], layout: MatrixLayout) -> Typed {
        assert_eq!(
            args.len(),
            2,
            "a device matrix load takes a place and a stride"
        );
        let pointer = self.reference(
            &args[0],
            if matches!(layout, MatrixLayout::RowMajor) {
                "a device matrix load"
            } else {
                "a device matrix transpose"
            },
        );
        let element = pointee(self, pointer.ty);
        let ty = match self.compiler.module().ty(element) {
            Type::Scalar(Scalar::F16) => {
                if matches!(layout, MatrixLayout::RowMajor) {
                    self.coopmat(ir::Type::Named("coopmat_a".to_owned()))
                } else {
                    self.coopmat(ir::Type::Named("coopmat_b".to_owned()))
                }
            }
            other => panic!(
                "a device matrix load reads {}",
                neura_shader::element_name(other)
            ),
        };
        let stride = self.value_with_hint(&args[1], Some(self.compiler.scalar("u32")));
        self.emit(ty, |result| DeviceInstruction::MatrixLoad {
            pointer: pointer.value,
            stride: stride.value,
            layout,
            result,
        })
    }

    pub(super) fn atomic(&mut self, name: &str, args: &[ir::Expression]) -> Typed {
        assert_eq!(args.len(), 2);
        let pointer = self.reference(&args[0], "a device atomic");
        let fun = match name {
            "atomic_add" => AtomicOp::Add,
            "atomic_sub" => AtomicOp::Subtract,
            other => panic!("{other} is not a device atomic"),
        };
        let ty = self.compiler.module().loaded_ty(pointer.ty);
        assert_eq!(
            self.compiler.module().ty(ty).scalar(),
            Some(Scalar::U32),
            "a device atomic counts unsigned words",
        );
        let value = self.value_with_hint(&args[1], Some(ty));
        self.emit(ty, |result| DeviceInstruction::Atomic {
            op: fun,
            pointer: pointer.value,
            value: value.value,
            result,
        })
    }
}

fn pointer_space(lower: &FunctionLower<'_>, ty: TypeId) -> neura_shader::Space {
    lower.compiler.module().pointee(ty).0
}

fn pointee(lower: &FunctionLower<'_>, ty: TypeId) -> TypeId {
    lower.compiler.module().pointee(ty).1
}
