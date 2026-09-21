use neura_abi::op::{OPS, Partial, Role};
use std::fmt::Write as _;

pub fn fragment() -> String {
    let mut source = String::from(
        "fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

",
    );
    source.push_str(
        "fn op_apply(kind: u32, op: u32, a: f32, b: f32) -> f32 {
    switch (op) {
",
    );
    for definition in OPS {
        writeln!(
            source,
            "        case {}: {{ return {}; }}",
            definition.constant, definition.apply,
        )
        .unwrap();
    }
    source.push_str(
        "        default: { refuse(kind, op); }
    }
    return 0.0;
}

",
    );
    source.push_str(
        "fn run_partial(task: Task, lid: u32) {
    let primary = values[select(task.a, task.out, task.a == NO_VALUE)];
    let other = values[select(task.b, task.out, task.b == NO_VALUE)];
    let gradient = values[task.c];
    let output = values[task.out];
    let dims = output.dims;
    for (var index = task.first + lid; index < task.first + task.count; index = index + WORKGROUP_SIZE) {
        let at = coordinates(index, dims);
        let g = fetch(gradient.base, read_address(at, gradient.strides));
        var result = 0.0;
        switch (task.op * 2u + task.slot) {
",
    );
    for definition in OPS {
        for (slot, partial) in definition.partials.iter().enumerate() {
            let Some(expression) = partial.formula() else {
                continue;
            };
            writeln!(
                source,
                "            case {}u: {{ {}result = {expression}; }}",
                definition.code * 2 + slot as u32,
                roles(*partial),
            )
            .unwrap();
        }
    }
    source.push_str(
        "            default: { refuse(task.kind, task.op * 2u + task.slot); }
        }
        publish(output.base, index, chained(task, at, result));
    }
}
",
    );
    source
}

fn roles(partial: Partial) -> String {
    let mut source = String::new();
    for role in partial.roles() {
        let record = match role {
            Role::Other => "other",
            Role::Operand | Role::Result => "primary",
        };
        write!(
            source,
            "let {} = fetch({record}.base, read_address(at, {record}.strides)); ",
            role.name(),
        )
        .unwrap();
    }
    source
}
