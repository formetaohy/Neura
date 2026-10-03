use super::{FunctionLower, Symbol, Typed};
use crate::{DeviceInstruction, ir};
use neura_shader::{Barrier, BinaryOp, MatrixLayout};

impl FunctionLower<'_> {
    pub(super) fn statements(&mut self, statements: &[ir::Statement]) {
        use ir::{Expression as E, Statement as S};
        for statement in statements {
            match statement {
                S::Let {
                    name,
                    mutable,
                    value,
                } => {
                    if let E::Call {
                        name: intrinsic,
                        arguments,
                    } = value
                        && intrinsic == "scalar_array"
                    {
                        assert!(*mutable, "scalar registers are mutable");
                        assert_eq!(
                            arguments.len(),
                            2,
                            "scalar registers require a value and a length"
                        );
                        let initial = self.value(&arguments[0]);
                        let count = self.compiler.evaluate(&arguments[1]);
                        assert!(count > 0, "a scalar register array is not empty");
                        let mut registers = Vec::with_capacity(count as usize);
                        for index in 0..count {
                            let register = format!("{name}_{index}");
                            let local = self.declare_local(&register, initial.ty);
                            self.push(DeviceInstruction::Store {
                                pointer: local.value,
                                value: initial.value,
                            });
                            registers.push(local);
                        }
                        assert!(
                            self.scopes
                                .last_mut()
                                .expect("registers belong to a block")
                                .insert(name.clone(), Symbol::Array(registers.into()))
                                .is_none(),
                            "the Rust device local {name} is declared twice in a block"
                        );
                        continue;
                    }
                    if !mutable && let Some(value) = self.evaluate_u32(value) {
                        assert!(
                            self.scopes
                                .last_mut()
                                .expect("a local has a block")
                                .insert(name.clone(), Symbol::Constant(value))
                                .is_none(),
                            "the Rust device local {name} is declared twice in a block"
                        );
                        continue;
                    }
                    let value = self.value(value);
                    if *mutable {
                        let local = self.declare_local(name, value.ty);
                        self.push(DeviceInstruction::Store {
                            pointer: local.value,
                            value: value.value,
                        });
                    } else {
                        assert!(
                            self.scopes
                                .last_mut()
                                .expect("a local has a block")
                                .insert(name.clone(), Symbol::Value(value))
                                .is_none(),
                            "the Rust device local {name} is declared twice in a block"
                        );
                    }
                }
                S::Assign {
                    place,
                    value,
                    operator,
                } => {
                    let op = operator.map(|operator| match operator {
                        ir::BinaryOperator::Add => BinaryOp::Add,
                        ir::BinaryOperator::Subtract => BinaryOp::Subtract,
                        ir::BinaryOperator::Multiply => BinaryOp::Multiply,
                        ir::BinaryOperator::Divide => BinaryOp::Divide,
                        ir::BinaryOperator::Modulo => BinaryOp::Modulo,
                        ir::BinaryOperator::BitAnd => BinaryOp::And,
                        ir::BinaryOperator::BitOr => BinaryOp::Or,
                        ir::BinaryOperator::BitXor => BinaryOp::Xor,
                        ir::BinaryOperator::ShiftLeft => BinaryOp::ShiftLeft,
                        ir::BinaryOperator::ShiftRight => BinaryOp::ShiftRight,
                        other => panic!("{other:?} cannot update a device place"),
                    });
                    self.assignment(place, value, op);
                }
                S::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let condition =
                        self.value_with_hint(condition, Some(self.compiler.scalar("bool")));
                    let accept = self.block(|lower| lower.statements(accept));
                    let reject = self.block(|lower| lower.statements(reject));
                    self.push(DeviceInstruction::If {
                        condition: condition.value,
                        accept,
                        reject,
                    });
                }
                S::Match { selector, arms } => {
                    let selector = self.value(selector);
                    let ty = self.compiler.scalar("u32");
                    assert!(
                        selector.ty == ty || selector.ty == self.compiler.scalar("i32"),
                        "device matches use an integer selector"
                    );
                    let mut cases = Vec::new();
                    let mut default = Vec::new();
                    for arm in arms {
                        match &arm.pattern {
                            ir::Pattern::Default => {
                                default = self.block(|lower| lower.statements(&arm.body));
                            }
                            ir::Pattern::Integer(value) => {
                                let body = self.block(|lower| lower.statements(&arm.body));
                                cases.push((*value, body));
                            }
                            ir::Pattern::Constant(name) => {
                                let value = self.compiler.constant_u32(name);
                                let body = self.block(|lower| lower.statements(&arm.body));
                                cases.push((value, body));
                            }
                        }
                    }
                    assert!(
                        arms.iter()
                            .any(|arm| matches!(arm.pattern, ir::Pattern::Default)),
                        "a device match covers its default"
                    );
                    self.push(DeviceInstruction::Switch {
                        selector: selector.value,
                        cases,
                        default,
                    });
                }
                S::For {
                    name,
                    start,
                    end,
                    step,
                    unroll,
                    body,
                } => {
                    self.for_loop(name, start, end, step, *unroll, body);
                }
                S::While { condition, body } => {
                    let body = self.block(|lower| {
                        let condition =
                            lower.value_with_hint(condition, Some(lower.compiler.scalar("bool")));
                        let negated = lower.emit(lower.compiler.scalar("bool"), |result| {
                            DeviceInstruction::Unary {
                                op: neura_shader::UnaryOp::LogicalNot,
                                value: condition.value,
                                result,
                            }
                        });
                        lower.push(DeviceInstruction::If {
                            condition: negated.value,
                            accept: vec![DeviceInstruction::Break],
                            reject: Vec::new(),
                        });
                        lower.statements(body);
                    });
                    self.push(DeviceInstruction::Loop {
                        body,
                        continuing: Vec::new(),
                    });
                }
                S::Loop(body) => {
                    let body = self.block(|lower| lower.statements(body));
                    self.push(DeviceInstruction::Loop {
                        body,
                        continuing: Vec::new(),
                    });
                }
                S::Return(value) => {
                    let value = value.as_ref().map(|expr| self.value(expr).value);
                    self.push(DeviceInstruction::Return { value });
                }
                S::Break => self.push(DeviceInstruction::Break),
                S::Continue => self.push(DeviceInstruction::Continue),
                S::Block(body) => {
                    let body = self.block(|lower| lower.statements(body));
                    self.push(DeviceInstruction::Block(body));
                }
                S::Expression(expr) => self.expression_statement(expr),
            }
        }
    }

    fn assignment(&mut self, left: &ir::Expression, right: &ir::Expression, op: Option<BinaryOp>) {
        let destination = self
            .place(left)
            .expect("a device assignment has a writable place");
        let ty = self.compiler.module().loaded_ty(destination.ty);
        let value = if let Some(op) = op {
            let loaded = self.emit(ty, |result| DeviceInstruction::Load {
                pointer: destination.value,
                result,
            });
            let incoming = self.value_with_hint(right, Some(ty));
            self.emit(ty, |result| DeviceInstruction::Binary {
                op,
                left: loaded.value,
                right: incoming.value,
                result,
            })
        } else {
            self.value_with_hint(right, Some(ty))
        };
        self.push(DeviceInstruction::Store {
            pointer: destination.value,
            value: value.value,
        });
    }

    pub(super) fn workgroup_uniform_load(&mut self, arguments: &[ir::Expression]) -> Typed {
        use ir::Expression as E;
        assert_eq!(
            arguments.len(),
            1,
            "a workgroup uniform load takes one reference",
        );
        let E::Reference(reference) = &arguments[0] else {
            panic!("a workgroup uniform load takes a reference");
        };
        let pointer = self
            .place(reference)
            .expect("a workgroup uniform load refers to a workgroup value");
        let ty = self.compiler.module().loaded_ty(pointer.ty);
        self.emit(ty, |result| DeviceInstruction::WorkGroupUniformLoad {
            pointer: pointer.value,
            result,
        })
    }

    fn expression_statement(&mut self, expr: &ir::Expression) {
        use ir::Expression as E;
        if let E::Call { name, arguments } = expr {
            match name.as_str() {
                "workgroup_barrier" => {
                    assert!(arguments.is_empty());
                    self.push(DeviceInstruction::Barrier(Barrier::WorkGroup));
                    return;
                }
                "storage_barrier" => {
                    assert!(arguments.is_empty());
                    self.push(DeviceInstruction::Barrier(Barrier::Storage));
                    return;
                }
                "atomic_store" => {
                    assert_eq!(arguments.len(), 2);
                    let pointer = self.reference(&arguments[0], "an atomic store");
                    let ty = self.compiler.module().loaded_ty(pointer.ty);
                    let value = self.value_with_hint(&arguments[1], Some(ty));
                    self.push(DeviceInstruction::Store {
                        pointer: pointer.value,
                        value: value.value,
                    });
                    return;
                }
                "coopmat_store" => {
                    assert_eq!(arguments.len(), 3);
                    let pointer = self.reference(&arguments[0], "a device matrix store");
                    let stride =
                        self.value_with_hint(&arguments[1], Some(self.compiler.scalar("u32")));
                    let value = self.value(&arguments[2]);
                    self.push(DeviceInstruction::MatrixStore {
                        pointer: pointer.value,
                        value: value.value,
                        stride: stride.value,
                        layout: MatrixLayout::RowMajor,
                    });
                    return;
                }
                "atomic_add" | "atomic_sub" => {
                    self.atomic(name, arguments);
                    return;
                }
                _ => {}
            }
            if !self.compiler.returns_value(name) {
                let function = self.compiler.lower_callee(name);
                let parameters = self.compiler.module().functions()[function as usize]
                    .arguments
                    .iter()
                    .map(|argument| argument.ty)
                    .collect::<Vec<_>>();
                assert_eq!(
                    parameters.len(),
                    arguments.len(),
                    "{name} takes a fixed number of arguments"
                );
                let arguments = arguments
                    .iter()
                    .zip(parameters)
                    .map(|(argument, ty)| self.value_with_hint(argument, Some(ty)).value)
                    .collect();
                self.push(DeviceInstruction::Call {
                    function,
                    arguments,
                    result: None,
                });
                return;
            }
        }
        self.value(expr);
    }

    fn for_loop(
        &mut self,
        name: &str,
        start: &ir::Expression,
        end: &ir::Expression,
        step: &ir::Expression,
        unroll: bool,
        statements: &[ir::Statement],
    ) {
        if unroll {
            let first = self.compiler.evaluate(start);
            let limit = self.compiler.evaluate(end);
            let increment = self.compiler.evaluate(step);
            assert!(increment > 0, "a compile-time loop advances");
            for index in (first..limit).step_by(increment as usize) {
                let body = self.block(|lower| {
                    lower
                        .scopes
                        .last_mut()
                        .expect("an unrolled loop has scope")
                        .insert(name.to_owned(), Symbol::Constant(index));
                    lower.statements(statements);
                });
                self.push(DeviceInstruction::Block(body));
            }
            return;
        }
        let initial = self.value(start);
        assert_eq!(
            initial.ty,
            self.compiler.scalar("u32"),
            "a device index is unsigned"
        );
        self.scopes.push(Default::default());
        let position = self.declare_local(name, initial.ty);
        self.push(DeviceInstruction::Store {
            pointer: position.value,
            value: initial.value,
        });
        let loaded = self.compiler.module().loaded_ty(position.ty);
        let body = self.block(|lower| {
            let current = lower.emit(loaded, |result| DeviceInstruction::Load {
                pointer: position.value,
                result,
            });
            let limit = lower.value_with_hint(end, Some(loaded));
            let condition = lower.emit(lower.compiler.scalar("bool"), |result| {
                DeviceInstruction::Binary {
                    op: BinaryOp::GreaterEqual,
                    left: current.value,
                    right: limit.value,
                    result,
                }
            });
            lower.push(DeviceInstruction::If {
                condition: condition.value,
                accept: vec![DeviceInstruction::Break],
                reject: Vec::new(),
            });
            lower.statements(statements);
        });
        let continuing = self.block(|lower| {
            let current = lower.emit(loaded, |result| DeviceInstruction::Load {
                pointer: position.value,
                result,
            });
            let increment = lower.value_with_hint(step, Some(loaded));
            let next = lower.emit(loaded, |result| DeviceInstruction::Binary {
                op: BinaryOp::Add,
                left: current.value,
                right: increment.value,
                result,
            });
            lower.push(DeviceInstruction::Store {
                pointer: position.value,
                value: next.value,
            });
        });
        self.push(DeviceInstruction::Loop { body, continuing });
        self.scopes.pop();
    }
}
