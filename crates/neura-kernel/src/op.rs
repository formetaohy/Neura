use neura_compiler::{Compiler, ir};
use neura_pointwise::{OPS, Role};

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
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

#[neura_compiler::module]
mod device {
    fn op_apply(kind: u32, op: u32, a: f32, b: f32) -> f32 {
        match op {
            _ => {
                refuse(kind, refusal::OP, op);
                return 0.0;
            }
        }
    }

    fn run_partial(task: Task, lid: u32) {
        let primary = values[select(task.a, task.out, task.a == NO_VALUE)];
        let other = values[select(task.b, task.out, task.b == NO_VALUE)];
        let gradient = values[task.c];
        let output = values[task.out];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = walked_at(task.geometry, index, dims);
            let g = fetch(gradient, read_address(at, gradient.strides));
            let mut result = 0.0;
            match task.op * 2u32 + task.slot {
                _ => refuse(task.kind, refusal::PARTIAL, task.op * 2u32 + task.slot),
            }
            publish(output, index, chained(task, at, result));
        }
    }
}
