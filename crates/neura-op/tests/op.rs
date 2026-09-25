use neura_ir::Expression;
use neura_op::{self as op, OPS, Role};

fn refuses(action: impl FnOnce() + std::panic::UnwindSafe) -> bool {
    std::panic::catch_unwind(action).is_err()
}

#[test]
fn every_pointwise_op_is_declared_once() {
    assert_eq!(op::COUNT as usize, OPS.len());
    for (code, entry) in OPS.iter().enumerate() {
        assert_eq!(
            entry.code, code as u32,
            "the {} op leaves a hole",
            entry.name
        );
        let resolved = op::of(entry.code);
        assert_eq!(resolved.code, entry.code);
        assert_eq!(resolved.family, entry.family);
        assert_eq!(resolved.name, entry.name);
        assert_eq!(op::name(entry.code), entry.name);
        assert_eq!(op::kind(entry.code), entry.family.kind());
        assert!(mentions(&entry.apply_expression(), "a"));
        assert_eq!(
            entry.partials.len() as u32,
            entry.family.operands(),
            "the {} op declares a partial for every operand it reads",
            entry.name,
        );
        for slot in 0..entry.family.operands() {
            let _ = entry.partial(slot);
        }
        assert!(refuses(|| {
            let _ = entry.partial(entry.family.operands());
        }));
    }
    let mut names = OPS.iter().map(|entry| entry.name).collect::<Vec<_>>();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), OPS.len(), "two ops share a name");
    assert_eq!(op::name(op::ADD), "add");
    assert_eq!(op::name(op::RELU), "relu");
    assert_eq!(op::name(op::SIGMOID), "sigmoid");
    assert!(refuses(|| {
        let _ = op::of(op::COUNT);
    }));
    assert!(refuses(|| {
        let _ = op::of(op::NONE);
    }));
}

#[test]
fn every_partial_reads_exactly_the_roles_it_names() {
    for op in OPS {
        for slot in 0..op.family.operands() {
            let partial = op.partial(slot);
            let roles = partial.roles();
            assert!(
                !(roles.contains(&Role::Operand) && roles.contains(&Role::Result)),
                "the {} partial {slot} differentiates an operand and its own result at once",
                op.name,
            );
            assert!(
                !roles.contains(&Role::Result) || roles.len() == 1,
                "the {} partial {slot} reads its own result next to an operand",
                op.name,
            );
            if op.family == op::Family::Unary {
                assert!(
                    !roles.contains(&Role::Other),
                    "the {} partial {slot} reads a second operand its op never has",
                    op.name,
                );
            }
            let Some(formula) = partial.formula() else {
                assert!(
                    roles.is_empty(),
                    "the {} partial {slot} reads a role it never asks to be handed",
                    op.name,
                );
                continue;
            };
            assert!(
                mentions(&formula, "g"),
                "the {} partial {slot} ignores the gradient it descends from",
                op.name,
            );
            for role in roles {
                assert!(
                    mentions(&formula, role.name()),
                    "the {} partial {slot} asks for {} it never reads",
                    op.name,
                    role.name(),
                );
            }
        }
    }
}

fn mentions(expr: &Expression, name: &str) -> bool {
    match expr {
        Expression::Name(value) => value == name,
        Expression::Field { base, .. }
        | Expression::Unary { value: base, .. }
        | Expression::Cast { value: base, .. }
        | Expression::Reference(base) => mentions(base, name),
        Expression::Index { base, index }
        | Expression::Binary {
            left: base,
            right: index,
            ..
        }
        | Expression::Repeat {
            value: base,
            length: index,
        } => mentions(base, name) || mentions(index, name),
        Expression::Call { arguments, .. } => arguments.iter().any(|arg| mentions(arg, name)),
        Expression::Integer { .. } | Expression::Float(_) | Expression::Bool(_) => false,
    }
}
