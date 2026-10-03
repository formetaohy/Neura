use neura_compiler::{Compiler, ast};
use neura_pointwise::{OPS, Role};

pub(crate) fn install(compiler: &mut Compiler) {
    device::define(compiler);
    for definition in OPS {
        compiler.insert_case(
            "op_apply",
            ast::Arm {
                pattern: ast::Pattern::Integer(definition.code),
                body: vec![ast::Statement::Return(Some(definition.apply_expression()))],
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
                    ast::Statement::Let {
                        name: role.name().to_owned(),
                        mutable: false,
                        value: ast::Expression::call(
                            "fetch",
                            vec![
                                ast::Expression::name(source),
                                ast::Expression::call(
                                    "read_address",
                                    vec![
                                        ast::Expression::name("at"),
                                        ast::Expression::field(
                                            ast::Expression::name(source),
                                            "strides",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    }
                })
                .collect::<Vec<_>>();
            body.push(ast::Statement::Assign {
                place: ast::Expression::name("result"),
                value: formula,
                operator: None,
            });
            compiler.insert_case(
                "run_partial",
                ast::Arm {
                    pattern: ast::Pattern::Integer(definition.code * 2 + slot as u32),
                    body,
                },
            );
        }
    }
}

#[neura_compiler::module]
mod device {
    fn sigmoid(x: f32) -> f32 {
        return 1.0 / (1.0 + exp(-x));
    }

    fn gelu(x: f32) -> f32 {
        let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1.0 + tanh(inner));
    }

    fn gelu_grad(x: f32) -> f32 {
        let inner = 0.7978845608 * (x + 0.044715 * x * x * x);
        let slope = tanh(inner);
        return 0.5 * (1.0 + slope)
            + 0.5 * x * (1.0 - slope * slope) * 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
    }

    fn chain_operand(step: Step, at: uvec4) -> f32 {
        if step.operand == NO_VALUE {
            return 0.0;
        }
        let source = values[step.operand];
        return fetch(source, read_address(at, source.strides));
    }

    fn chained(task: Task, at: uvec4, carried: f32) -> f32 {
        let mut result = carried;
        for step in stride(0u32, task.steps, 1u32) {
            let record = steps[task.chain + step];
            let operand = chain_operand(record, at);
            let swapped = record.swapped == 1u32;
            result = op_apply(
                task.kind,
                record.op,
                select(result, operand, swapped),
                select(operand, result, swapped),
            );
        }
        return result;
    }

    fn opened(task: Task, at: uvec4, carried: f32) -> f32 {
        let mut result = carried;
        for step in stride(0u32, task.prelude_steps, 1u32) {
            let record = steps[task.prelude + step];
            let operand = chain_operand(record, at);
            let swapped = record.swapped == 1u32;
            result = op_apply(
                task.kind,
                record.op,
                select(result, operand, swapped),
                select(operand, result, swapped),
            );
        }
        return result;
    }

    fn read_frame(task: Task, source: Value, at: uvec4) -> f32 {
        let carried = fetch(source, read_address(at, source.strides));
        if task.prelude_steps == 0u32 {
            return carried;
        }
        return opened(task, at, carried);
    }

    fn read_flat(task: Task, source: Value, index: u32) -> f32 {
        if task.prelude_steps == 0u32 {
            return fetch(source, index);
        }
        let at = coordinates(index, source.dims);
        return opened(task, at, fetch(source, read_address(at, source.strides)));
    }

    fn run_binary(task: Task, lid: u32) {
        let left = values[task.a];
        let right = values[task.b];
        let output = values[task.out];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = walked_at(task.geometry, index, dims);
            let a = fetch(left, read_address(at, left.strides));
            let b = fetch(right, read_address(at, right.strides));
            publish(
                output,
                index,
                chained(task, at, op_apply(task.kind, task.op, a, b)),
            );
        }
    }

    fn run_unary(task: Task, lid: u32) {
        let source = values[task.a];
        let output = values[task.out];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = walked_at(task.geometry, index, dims);
            let a = fetch(source, read_address(at, source.strides));
            publish(
                output,
                index,
                chained(task, at, op_apply(task.kind, task.op, a, 0.0)),
            );
        }
    }

    fn run_fill(task: Task, lid: u32) {
        let output = values[task.out];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            publish(
                output,
                index,
                chained(task, walked_at(task.geometry, index, dims), task.param),
            );
        }
    }

    fn run_broadcast(task: Task, lid: u32) {
        let output = values[task.out];
        let source = values[task.a];
        let dims = output.dims;
        for index in stride(task.first + lid, task.first + task.count, WORKGROUP_SIZE) {
            let at = walked_at(task.geometry, index, dims);
            publish(
                output,
                index,
                chained(task, at, fetch(source, read_address(at, source.strides))),
            );
        }
    }

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
