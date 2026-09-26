use super::{FunctionLower, Symbol, Typed};
use crate::{Global, ir};
use naga::{
    BinaryOperator, Expression, Handle, LocalVariable, MathFunction, ScalarKind, Statement, Type,
    TypeInner, UnaryOperator,
};

impl FunctionLower<'_> {
    fn scalar(&self, ty: Handle<Type>) -> ScalarKind {
        let TypeInner::Scalar(scalar) = self.compiler.module.types[ty].inner else {
            panic!("a device numeric operation requires a scalar");
        };
        scalar.kind
    }

    fn constant(&mut self, value: u32) -> Typed {
        self.emit(
            Expression::Literal(naga::Literal::U32(value)),
            self.compiler.ty("u32"),
        )
    }

    fn literal(&mut self, value: u32, kind: ir::IntegerType, hint: Option<Handle<Type>>) -> Typed {
        let signed = matches!(kind, ir::IntegerType::Signed)
            || matches!(kind, ir::IntegerType::Inferred)
                && hint.is_some_and(|ty| self.compiler.ty("i32") == ty);
        if signed {
            self.emit(
                Expression::Literal(naga::Literal::I32(
                    i32::try_from(value).expect("a signed device integer fits in i32"),
                )),
                self.compiler.ty("i32"),
            )
        } else {
            self.constant(value)
        }
    }

    fn field_type(&self, base: Handle<Type>, field: &str) -> (u32, Handle<Type>) {
        match &self.compiler.module.types[base].inner {
            TypeInner::Struct { members, .. } => members
                .iter()
                .enumerate()
                .find(|(_, member)| member.name.as_deref() == Some(field))
                .map(|(index, member)| (index as u32, member.ty))
                .unwrap_or_else(|| panic!("Rust device struct has no field {field}")),
            TypeInner::Vector { size, scalar } => {
                let index = match field {
                    "x" => 0,
                    "y" => 1,
                    "z" => 2,
                    "w" => 3,
                    _ => panic!("Rust device vector has no component {field}"),
                };
                assert!(index < *size as u32, "a device vector has no {field} lane");
                let ty = match scalar.kind {
                    ScalarKind::Uint => self.compiler.ty("u32"),
                    ScalarKind::Float => self.compiler.ty("f32"),
                    other => panic!("unsupported vector lane {other:?}"),
                };
                (index, ty)
            }
            other => panic!("a field of {other:?} is not accessible"),
        }
    }

    fn element_type(&self, array: Handle<Type>) -> Handle<Type> {
        match &self.compiler.module.types[array].inner {
            TypeInner::Array { base, .. } => *base,
            TypeInner::Vector { scalar, .. } => match scalar.kind {
                ScalarKind::Uint => self.compiler.ty("u32"),
                ScalarKind::Float => self.compiler.ty("f32"),
                other => panic!("a device vector of {other:?} is not indexable"),
            },
            other => panic!("a Rust device indexes {other:?} instead of an array or vector"),
        }
    }

    fn indexed(&mut self, base: Typed, index: &ir::Expression) -> Typed {
        let ty = self.element_type(base.ty);
        if let TypeInner::Vector { size, .. } = &self.compiler.module.types[base.ty].inner {
            let size = *size as u32;
            let index = self
                .evaluate_u32(index)
                .expect("a device vector requires a compile-time lane index");
            assert!(index < size, "a vector index lies inside its lanes");
            self.emit(
                Expression::AccessIndex {
                    base: base.expr,
                    index,
                },
                ty,
            )
        } else {
            let index = self.value(index);
            self.emit(
                Expression::Access {
                    base: base.expr,
                    index: index.expr,
                },
                ty,
            )
        }
    }

    fn global(&mut self, global: Global) -> Typed {
        self.emit(Expression::GlobalVariable(global.handle), global.ty)
    }

    pub(super) fn place(&mut self, expr: &ir::Expression) -> Option<Typed> {
        use ir::Expression as E;
        match expr {
            E::Name(name) => match self.lookup(name) {
                Some(Symbol::Local(local)) => Some(local),
                Some(Symbol::Value(_) | Symbol::Array(_) | Symbol::Constant(_)) => None,
                None => self
                    .compiler
                    .globals
                    .get(name)
                    .copied()
                    .map(|global| self.global(global)),
            },
            E::Field { base, name } => {
                let base = self.place(base)?;
                let (index, ty) = self.field_type(base.ty, name);
                Some(self.emit(
                    Expression::AccessIndex {
                        base: base.expr,
                        index,
                    },
                    ty,
                ))
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
                Some(self.indexed(base, index))
            }
            _ => None,
        }
    }

    pub(super) fn value(&mut self, expr: &ir::Expression) -> Typed {
        self.value_with_hint(expr, None)
    }

    pub(super) fn value_with_hint(
        &mut self,
        expr: &ir::Expression,
        hint: Option<Handle<Type>>,
    ) -> Typed {
        use ir::Expression as E;
        if matches!(expr, E::Name(_) | E::Field { .. } | E::Index { .. })
            && let Some(place) = self.place(expr)
        {
            let loaded = self.emit(
                Expression::Load {
                    pointer: place.expr,
                },
                place.ty,
            );
            if let Some(ty) = hint {
                assert_eq!(loaded.ty, ty, "device operand has a different type");
            }
            return loaded;
        }
        let value = match expr {
            E::Integer { value, ty } => self.literal(*value, *ty, hint),
            E::Float(value) => self.emit(
                Expression::Literal(naga::Literal::F32(*value)),
                self.compiler.ty("f32"),
            ),
            E::Bool(value) => self.emit(
                Expression::Literal(naga::Literal::Bool(*value)),
                self.compiler.ty("bool"),
            ),
            E::Name(name) => match self.lookup(name) {
                Some(Symbol::Value(value)) => value,
                Some(Symbol::Constant(value)) => self.constant(value),
                Some(Symbol::Array(_)) => {
                    panic!("a scalar register array is accessed by constant index")
                }
                Some(Symbol::Local(_)) => unreachable!("a local is a device place"),
                None => panic!("Rust device name {name} is not defined"),
            },
            E::Field { base, name } => {
                let base = self.value(base);
                let (index, ty) = self.field_type(base.ty, name);
                self.emit(
                    Expression::AccessIndex {
                        base: base.expr,
                        index,
                    },
                    ty,
                )
            }
            E::Index { base, index } => {
                let base = self.value(base);
                self.indexed(base, index)
            }
            E::Unary { op, value } => {
                let arg = self.value(value);
                let op = match op {
                    ir::UnaryOperator::Negate => UnaryOperator::Negate,
                    ir::UnaryOperator::Not if self.scalar(arg.ty) == ScalarKind::Bool => {
                        UnaryOperator::LogicalNot
                    }
                    ir::UnaryOperator::Not => UnaryOperator::BitwiseNot,
                };
                self.emit(Expression::Unary { op, expr: arg.expr }, arg.ty)
            }
            E::Binary { op, left, right } => self.binary(*op, left, right),
            E::Call { name, arguments } => self.call(name, arguments),
            E::Repeat { value, length } => {
                let count = self.compiler.evaluate(length);
                let value = self.value(value);
                assert!(
                    matches!(
                        self.function.expressions[value.expr],
                        Expression::Literal(naga::Literal::F32(0.0))
                            | Expression::Literal(naga::Literal::U32(0))
                    ),
                    "a device array starts from zero"
                );
                let ty = self.compiler.array(value.ty, Some(count));
                self.emit(Expression::ZeroValue(ty), ty)
            }
            E::Cast { value, ty } => {
                let source = self.value(value);
                let target = self.compiler.rust_type(ty);
                let kind = self.scalar(target);
                self.emit(
                    Expression::As {
                        expr: source.expr,
                        kind,
                        convert: Some(4),
                    },
                    target,
                )
            }
            E::Reference(_) => panic!("a device reference is only passed to an atomic store"),
        };
        if let Some(ty) = hint {
            assert_eq!(value.ty, ty, "device operand has a different type");
        }
        value
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
            Op::Add => BinaryOperator::Add,
            Op::Subtract => BinaryOperator::Subtract,
            Op::Multiply => BinaryOperator::Multiply,
            Op::Divide => BinaryOperator::Divide,
            Op::Modulo => BinaryOperator::Modulo,
            Op::Equal => BinaryOperator::Equal,
            Op::NotEqual => BinaryOperator::NotEqual,
            Op::Less => BinaryOperator::Less,
            Op::LessEqual => BinaryOperator::LessEqual,
            Op::Greater => BinaryOperator::Greater,
            Op::GreaterEqual => BinaryOperator::GreaterEqual,
            Op::BitAnd => BinaryOperator::And,
            Op::BitOr => BinaryOperator::InclusiveOr,
            Op::BitXor => BinaryOperator::ExclusiveOr,
            Op::LogicalAnd => BinaryOperator::LogicalAnd,
            Op::LogicalOr => BinaryOperator::LogicalOr,
            Op::ShiftLeft => BinaryOperator::ShiftLeft,
            Op::ShiftRight => BinaryOperator::ShiftRight,
        };
        if matches!(op, BinaryOperator::LogicalAnd | BinaryOperator::LogicalOr) {
            let boolean = self.compiler.ty("bool");
            let left = self.value_with_hint(left, Some(boolean));
            let local = self.function.local_variables.append(
                LocalVariable {
                    name: None,
                    ty: boolean,
                    init: None,
                },
                naga::Span::UNDEFINED,
            );
            let pointer = self.emit(Expression::LocalVariable(local), boolean);
            self.push(Statement::Store {
                pointer: pointer.expr,
                value: left.expr,
            });
            let branch = self.block(|lower| {
                let right = lower.value_with_hint(right, Some(boolean));
                lower.push(Statement::Store {
                    pointer: pointer.expr,
                    value: right.expr,
                });
            });
            let (accept, reject) = if op == BinaryOperator::LogicalAnd {
                (branch, naga::Block::new())
            } else {
                (naga::Block::new(), branch)
            };
            self.push(Statement::If {
                condition: left.expr,
                accept,
                reject,
            });
            return self.emit(
                Expression::Load {
                    pointer: pointer.expr,
                },
                boolean,
            );
        }
        let left = self.value(left);
        let right = if op == BinaryOperator::Multiply
            && matches!(
                self.compiler.module.types[left.ty].inner,
                TypeInner::Vector { .. }
            ) {
            self.value(right)
        } else if matches!(op, BinaryOperator::ShiftLeft | BinaryOperator::ShiftRight) {
            self.value_with_hint(right, Some(self.compiler.ty("u32")))
        } else {
            self.value_with_hint(right, Some(left.ty))
        };
        let ty = if matches!(
            op,
            BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterEqual
                | BinaryOperator::LogicalAnd
                | BinaryOperator::LogicalOr
        ) {
            self.compiler.ty("bool")
        } else {
            left.ty
        };
        self.emit(
            Expression::Binary {
                op,
                left: left.expr,
                right: right.expr,
            },
            ty,
        )
    }

    fn call(&mut self, name: &str, args: &[ir::Expression]) -> Typed {
        match name {
            "uvec4" => {
                assert_eq!(args.len(), 4);
                let ty = self.compiler.ty("u32");
                let components = args
                    .iter()
                    .map(|arg| self.value_with_hint(arg, Some(ty)).expr)
                    .collect();
                let ty = self.compiler.ty("uvec4");
                self.emit(Expression::Compose { ty, components }, ty)
            }
            "select" => {
                assert_eq!(args.len(), 3);
                let reject = self.value(&args[0]);
                let accept = self.value_with_hint(&args[1], Some(reject.ty));
                let condition = self.value_with_hint(&args[2], Some(self.compiler.ty("bool")));
                self.emit(
                    Expression::Select {
                        condition: condition.expr,
                        accept: accept.expr,
                        reject: reject.expr,
                    },
                    reject.ty,
                )
            }
            "bitcast_u32" => {
                assert_eq!(args.len(), 1);
                let source = self.value_with_hint(&args[0], Some(self.compiler.ty("f32")));
                let ty = self.compiler.ty("u32");
                self.emit(
                    Expression::As {
                        expr: source.expr,
                        kind: ScalarKind::Uint,
                        convert: None,
                    },
                    ty,
                )
            }
            "bitcast_f32" => {
                assert_eq!(args.len(), 1);
                let source = self.value_with_hint(&args[0], Some(self.compiler.ty("u32")));
                let ty = self.compiler.ty("f32");
                self.emit(
                    Expression::As {
                        expr: source.expr,
                        kind: ScalarKind::Float,
                        convert: None,
                    },
                    ty,
                )
            }
            "f32" | "i32" | "u32" => {
                assert_eq!(args.len(), 1);
                let source = self.value(&args[0]);
                let ty = self.compiler.ty(name);
                let kind = self.scalar(ty);
                self.emit(
                    Expression::As {
                        expr: source.expr,
                        kind,
                        convert: Some(4),
                    },
                    ty,
                )
            }
            "unpack2x16float" => {
                assert_eq!(args.len(), 1);
                let source = self.value_with_hint(&args[0], Some(self.compiler.ty("u32")));
                self.emit(
                    Expression::Math {
                        fun: MathFunction::Unpack2x16float,
                        arg: source.expr,
                        arg1: None,
                        arg2: None,
                        arg3: None,
                    },
                    self.compiler.ty("fvec2"),
                )
            }
            "max" | "min" | "abs" | "sqrt" | "exp" | "log" | "tanh" | "trunc" | "sin" | "cos"
            | "pow" | "floor" => {
                let fun = match name {
                    "max" => MathFunction::Max,
                    "min" => MathFunction::Min,
                    "abs" => MathFunction::Abs,
                    "sqrt" => MathFunction::Sqrt,
                    "exp" => MathFunction::Exp,
                    "log" => MathFunction::Log,
                    "tanh" => MathFunction::Tanh,
                    "trunc" => MathFunction::Trunc,
                    "sin" => MathFunction::Sin,
                    "cos" => MathFunction::Cos,
                    "pow" => MathFunction::Pow,
                    "floor" => MathFunction::Floor,
                    _ => unreachable!(),
                };
                assert_eq!(
                    args.len(),
                    if matches!(name, "max" | "min" | "pow") {
                        2
                    } else {
                        1
                    }
                );
                let first = self.value(&args[0]);
                let second = args
                    .get(1)
                    .map(|arg| self.value_with_hint(arg, Some(first.ty)).expr);
                self.emit(
                    Expression::Math {
                        fun,
                        arg: first.expr,
                        arg1: second,
                        arg2: None,
                        arg3: None,
                    },
                    first.ty,
                )
            }
            "workgroup_barrier" | "storage_barrier" | "atomic_store" => {
                panic!("a device synchronization or atomic store cannot yield a value")
            }
            _ => {
                let function = self.compiler.lower_function(name);
                let (parameters, result) = {
                    let function = &self.compiler.module.functions[function];
                    (
                        function
                            .arguments
                            .iter()
                            .map(|arg| arg.ty)
                            .collect::<Vec<_>>(),
                        function.result.as_ref().map(|result| result.ty),
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
                    .map(|(arg, ty)| self.value_with_hint(arg, Some(ty)).expr)
                    .collect();
                let result = result.expect("a void function cannot be used as a value");
                let value = self.emit(Expression::CallResult(function), result);
                self.push(Statement::Call {
                    function,
                    arguments,
                    result: Some(value.expr),
                });
                value
            }
        }
    }
}
