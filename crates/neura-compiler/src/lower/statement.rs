use super::{FunctionLower, Symbol, Typed};
use crate::ir;
use naga::{BinaryOperator, Expression, Statement, SwitchCase, SwitchValue};

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
                            let handle = self.function.local_variables.append(
                                naga::LocalVariable {
                                    name: Some(format!("{name}_{index}")),
                                    ty: initial.ty,
                                    init: None,
                                },
                                naga::Span::UNDEFINED,
                            );
                            let pointer = self.emit(Expression::LocalVariable(handle), initial.ty);
                            self.push(Statement::Store {
                                pointer: pointer.expr,
                                value: initial.expr,
                            });
                            registers.push(pointer);
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
                        self.declare(name, value);
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
                        ir::BinaryOperator::Add => BinaryOperator::Add,
                        ir::BinaryOperator::Subtract => BinaryOperator::Subtract,
                        ir::BinaryOperator::Multiply => BinaryOperator::Multiply,
                        ir::BinaryOperator::Divide => BinaryOperator::Divide,
                        ir::BinaryOperator::Modulo => BinaryOperator::Modulo,
                        ir::BinaryOperator::BitAnd => BinaryOperator::And,
                        ir::BinaryOperator::BitOr => BinaryOperator::InclusiveOr,
                        ir::BinaryOperator::BitXor => BinaryOperator::ExclusiveOr,
                        ir::BinaryOperator::ShiftLeft => BinaryOperator::ShiftLeft,
                        ir::BinaryOperator::ShiftRight => BinaryOperator::ShiftRight,
                        other => panic!("{other:?} cannot update a device place"),
                    });
                    self.assignment(place, value, op);
                }
                S::If {
                    condition,
                    accept,
                    reject,
                } => {
                    let condition = self.value_with_hint(condition, Some(self.compiler.ty("bool")));
                    let accept = self.block(|lower| lower.statements(accept));
                    let reject = self.block(|lower| lower.statements(reject));
                    self.push(Statement::If {
                        condition: condition.expr,
                        accept,
                        reject,
                    });
                }
                S::Match { selector, arms } => {
                    let selector = self.value(selector);
                    assert!(
                        selector.ty == self.compiler.ty("u32")
                            || selector.ty == self.compiler.ty("i32"),
                        "device matches use an integer selector"
                    );
                    let mut cases = Vec::new();
                    for arm in arms {
                        let value = match &arm.pattern {
                            ir::Pattern::Default => SwitchValue::Default,
                            ir::Pattern::Integer(value) => SwitchValue::U32(*value),
                            ir::Pattern::Constant(name) => SwitchValue::U32(
                                *self
                                    .compiler
                                    .constants
                                    .get(name)
                                    .unwrap_or_else(|| panic!("unknown device case {name}")),
                            ),
                        };
                        let body = self.block(|lower| lower.statements(&arm.body));
                        cases.push(SwitchCase {
                            value,
                            body,
                            fall_through: false,
                        });
                    }
                    assert!(
                        cases.iter().any(|case| case.value == SwitchValue::Default),
                        "a device match covers its default"
                    );
                    self.push(Statement::Switch {
                        selector: selector.expr,
                        cases,
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
                            lower.value_with_hint(condition, Some(lower.compiler.ty("bool")));
                        let negated = lower.emit(
                            Expression::Unary {
                                op: naga::UnaryOperator::LogicalNot,
                                expr: condition.expr,
                            },
                            lower.compiler.ty("bool"),
                        );
                        lower.push(Statement::If {
                            condition: negated.expr,
                            accept: naga::Block::from_vec(vec![Statement::Break]),
                            reject: naga::Block::new(),
                        });
                        lower.statements(body);
                    });
                    self.push(Statement::Loop {
                        body,
                        continuing: naga::Block::new(),
                        break_if: None,
                    });
                }
                S::Loop(body) => {
                    let body = self.block(|lower| lower.statements(body));
                    self.push(Statement::Loop {
                        body,
                        continuing: naga::Block::new(),
                        break_if: None,
                    });
                }
                S::Return(value) => {
                    let value = value.as_ref().map(|expr| self.value(expr).expr);
                    self.push(Statement::Return { value });
                }
                S::Break => self.push(Statement::Break),
                S::Continue => self.push(Statement::Continue),
                S::Block(body) => {
                    let body = self.block(|lower| lower.statements(body));
                    self.push(Statement::Block(body));
                }
                S::Expression(expr) => self.expression_statement(expr),
            }
        }
    }

    fn declare(&mut self, name: &str, value: Typed) -> Typed {
        let handle = self.function.local_variables.append(
            naga::LocalVariable {
                name: Some(name.to_owned()),
                ty: value.ty,
                init: None,
            },
            naga::Span::UNDEFINED,
        );
        let pointer = self.emit(Expression::LocalVariable(handle), value.ty);
        assert!(
            self.scopes
                .last_mut()
                .expect("a local is declared in a block")
                .insert(name.to_owned(), Symbol::Local(pointer))
                .is_none(),
            "the Rust device local {name} is declared twice in a block"
        );
        self.push(Statement::Store {
            pointer: pointer.expr,
            value: value.expr,
        });
        pointer
    }

    fn assignment(
        &mut self,
        left: &ir::Expression,
        right: &ir::Expression,
        op: Option<BinaryOperator>,
    ) {
        let destination = self
            .place(left)
            .expect("a device assignment has a writable place");
        let value = if let Some(op) = op {
            let old = self.emit(
                Expression::Load {
                    pointer: destination.expr,
                },
                destination.ty,
            );
            let incoming = self.value_with_hint(right, Some(destination.ty));
            self.emit(
                Expression::Binary {
                    op,
                    left: old.expr,
                    right: incoming.expr,
                },
                destination.ty,
            )
        } else {
            self.value_with_hint(right, Some(destination.ty))
        };
        self.push(Statement::Store {
            pointer: destination.expr,
            value: value.expr,
        });
    }

    fn expression_statement(&mut self, expr: &ir::Expression) {
        use ir::Expression as E;
        if let E::Call { name, arguments } = expr {
            match name.as_str() {
                "workgroup_barrier" => {
                    assert!(arguments.is_empty());
                    self.push(Statement::ControlBarrier(naga::Barrier::WORK_GROUP));
                    return;
                }
                "storage_barrier" => {
                    assert!(arguments.is_empty());
                    self.push(Statement::ControlBarrier(naga::Barrier::STORAGE));
                    return;
                }
                "atomic_store" => {
                    assert_eq!(arguments.len(), 2);
                    let E::Reference(reference) = &arguments[0] else {
                        panic!("an atomic store takes a reference");
                    };
                    let pointer = self
                        .place(reference)
                        .expect("an atomic store refers to a buffer");
                    let value = self.value_with_hint(&arguments[1], Some(self.compiler.ty("u32")));
                    self.push(Statement::Store {
                        pointer: pointer.expr,
                        value: value.expr,
                    });
                    return;
                }
                _ => {}
            }
            if self
                .compiler
                .functions
                .get(name)
                .is_some_and(|function| function.result.is_none())
            {
                let function = self.compiler.lower_function(name);
                let parameters = self.compiler.module.functions[function]
                    .arguments
                    .iter()
                    .map(|arg| arg.ty)
                    .collect::<Vec<_>>();
                assert_eq!(
                    parameters.len(),
                    arguments.len(),
                    "{name} takes a fixed number of arguments"
                );
                let arguments = arguments
                    .iter()
                    .zip(parameters)
                    .map(|(argument, ty)| self.value_with_hint(argument, Some(ty)).expr)
                    .collect();
                self.push(Statement::Call {
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
                self.push(Statement::Block(body));
            }
            return;
        }
        let initial = self.value(start);
        assert_eq!(
            initial.ty,
            self.compiler.ty("u32"),
            "a device index is unsigned"
        );
        self.scopes.push(Default::default());
        let position = self.declare(name, initial);
        let body = self.block(|lower| {
            let current = lower.emit(
                Expression::Load {
                    pointer: position.expr,
                },
                position.ty,
            );
            let limit = lower.value_with_hint(end, Some(position.ty));
            let condition = lower.emit(
                Expression::Binary {
                    op: BinaryOperator::GreaterEqual,
                    left: current.expr,
                    right: limit.expr,
                },
                lower.compiler.ty("bool"),
            );
            lower.push(Statement::If {
                condition: condition.expr,
                accept: naga::Block::from_vec(vec![Statement::Break]),
                reject: naga::Block::new(),
            });
            lower.statements(statements);
        });
        let continuing = self.block(|lower| {
            let current = lower.emit(
                Expression::Load {
                    pointer: position.expr,
                },
                position.ty,
            );
            let increment = lower.value_with_hint(step, Some(position.ty));
            let next = lower.emit(
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left: current.expr,
                    right: increment.expr,
                },
                position.ty,
            );
            lower.push(Statement::Store {
                pointer: position.expr,
                value: next.expr,
            });
        });
        self.push(Statement::Loop {
            body,
            continuing,
            break_if: None,
        });
        self.scopes.pop();
    }
}
