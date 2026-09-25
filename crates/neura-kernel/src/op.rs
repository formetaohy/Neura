use neura_compiler::{Compiler, ir};
use neura_op::{OPS, Role};

pub fn define(compiler: &mut Compiler) {
    super::op_device::define(compiler);
    for definition in OPS {
        compiler.insert_case(
            "op_apply",
            ir::Arm {
                pattern: ir::Pattern::Integer(definition.code),
                body: vec![ir::Statement::Return(Some(definition.apply_expression()))],
            },
        );
        for (slot, partial) in definition.partials.iter().enumerate() {
            let Some(formula) = partial.formula() else {
                continue;
            };
            let mut body = partial
                .roles()
                .iter()
                .map(|role| {
                    let source = match role {
                        Role::Other => "other",
                        Role::Operand | Role::Result => "primary",
                    };
                    ir::Statement::Let {
                        name: role.name().to_owned(),
                        mutable: false,
                        value: ir::Expression::call(
                            "fetch",
                            vec![
                                ir::Expression::name(source),
                                ir::Expression::call(
                                    "read_address",
                                    vec![
                                        ir::Expression::name("at"),
                                        ir::Expression::field(
                                            ir::Expression::name(source),
                                            "strides",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    }
                })
                .collect::<Vec<_>>();
            body.push(ir::Statement::Assign {
                place: ir::Expression::name("result"),
                value: formula,
                operator: None,
            });
            compiler.insert_case(
                "run_partial",
                ir::Arm {
                    pattern: ir::Pattern::Integer(definition.code * 2 + slot as u32),
                    body,
                },
            );
        }
    }
}
