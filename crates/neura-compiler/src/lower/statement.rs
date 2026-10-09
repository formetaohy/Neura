use super::{FunctionLower, Symbol, Typed};
use crate::DeviceInstruction;
use crate::ast;
use crate::device::{Intrinsic, LoopForm};
use neura_shader::{Barrier, BinaryOp, MatrixLayout};

impl FunctionLower<'_> {
    pub(super) fn statements(&mut self, statements: &[ast::Statement]) {
        use crate::ast::{Expression as E, Statement as S};
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
                        && Intrinsic::of(intrinsic) == Some(Intrinsic::ScalarArray)
                    {
                        assert!(*mutable, "scalar registers are mutable");
                        let arguments = self.intrinsic_arguments(Intrinsic::ScalarArray, arguments);
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
                        ast::BinaryOperator::Add => BinaryOp::Add,
                        ast::BinaryOperator::Subtract => BinaryOp::Subtract,
                        ast::BinaryOperator::Multiply => BinaryOp::Multiply,
                        ast::BinaryOperator::Divide => BinaryOp::Divide,
                        ast::BinaryOperator::Modulo => BinaryOp::Modulo,
                        ast::BinaryOperator::BitAnd => BinaryOp::And,
                        ast::BinaryOperator::BitOr => BinaryOp::Or,
                        ast::BinaryOperator::BitXor => BinaryOp::Xor,
                        ast::BinaryOperator::ShiftLeft => BinaryOp::ShiftLeft,
                        ast::BinaryOperator::ShiftRight => BinaryOp::ShiftRight,
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
                            ast::Pattern::Default => {
                                default = self.block(|lower| lower.statements(&arm.body));
                            }
                            ast::Pattern::Integer(value) => {
                                let body = self.block(|lower| lower.statements(&arm.body));
                                cases.push((*value, body));
                            }
                            ast::Pattern::Constant(name) => {
                                let value = self.compiler.constant_u32(name);
                                let body = self.block(|lower| lower.statements(&arm.body));
                                cases.push((value, body));
                            }
                        }
                    }
                    assert!(
                        arms.iter()
                            .any(|arm| matches!(arm.pattern, ast::Pattern::Default)),
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
                    iterator,
                    body,
                } => {
                    self.for_loop(name, iterator, body);
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

    fn assignment(
        &mut self,
        left: &ast::Expression,
        right: &ast::Expression,
        op: Option<BinaryOp>,
    ) {
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

    pub(super) fn workgroup_uniform_load(&mut self, arguments: &[ast::Expression]) -> Typed {
        use crate::ast::Expression as E;
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

    fn expression_statement(&mut self, expr: &ast::Expression) {
        use crate::ast::Expression as E;
        if let E::Call { name, arguments } = expr {
            match Intrinsic::of(name) {
                Some(intrinsic) => {
                    let arguments = self.intrinsic_arguments(intrinsic, arguments);
                    match intrinsic {
                        Intrinsic::WorkgroupBarrier => {
                            self.push(DeviceInstruction::Barrier(Barrier::WorkGroup));
                            return;
                        }
                        Intrinsic::StorageBarrier => {
                            self.push(DeviceInstruction::Barrier(Barrier::Storage));
                            return;
                        }
                        Intrinsic::AtomicStore => {
                            let pointer = self.reference(&arguments[0], "an atomic store");
                            let ty = self.compiler.module().loaded_ty(pointer.ty);
                            let value = self.value_with_hint(&arguments[1], Some(ty));
                            self.push(DeviceInstruction::Store {
                                pointer: pointer.value,
                                value: value.value,
                            });
                            return;
                        }
                        Intrinsic::CoopmatStore => {
                            let pointer = self.reference(&arguments[0], "a device matrix store");
                            let stride = self
                                .value_with_hint(&arguments[1], Some(self.compiler.scalar("u32")));
                            let value = self.value(&arguments[2]);
                            self.push(DeviceInstruction::MatrixStore {
                                pointer: pointer.value,
                                value: value.value,
                                stride: stride.value,
                                layout: MatrixLayout::RowMajor,
                            });
                            return;
                        }
                        Intrinsic::AtomicAdd | Intrinsic::AtomicSub => {
                            self.atomic(intrinsic, arguments);
                            return;
                        }
                        _ => {}
                    }
                }
                None if !self.compiler.returns_value(name) => {
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
                None => {}
            }
        }
        self.value(expr);
    }

    fn for_loop(&mut self, name: &str, iterator: &ast::Expression, statements: &[ast::Statement]) {
        let ast::Expression::Call {
            name: loop_form,
            arguments,
        } = iterator
        else {
            panic!("a device loop walks a declared loop form");
        };
        let form = LoopForm::of(loop_form)
            .unwrap_or_else(|| panic!("the device loop form {loop_form} is not declared"));
        assert_eq!(
            arguments.len(),
            3,
            "the device loop form {} takes a start, an end and a step",
            form.name(),
        );
        let start = &arguments[0];
        let end = &arguments[1];
        let step = &arguments[2];
        if form.unroll() {
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
