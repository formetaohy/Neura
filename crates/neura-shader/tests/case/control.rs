use neura_shader::{
    Access, Address, Argument, BinaryOp, Binding, BuiltIn, Constant, Function, Global, Instruction,
    Language, Module, Scalar, Space,
};

#[derive(Clone, Copy)]
enum Placement {
    InACase,
    UnderACaseBranch,
    InANestedLoop,
    Nowhere,
}

fn control(placement: Placement) -> Module {
    let mut module = Module::new(64);
    let u32_ty = module.scalar(Scalar::U32);
    let bool_ty = module.scalar(Scalar::Bool);
    let table_ty = module.array(u32_ty, None);
    let out = module.add_global(Global {
        name: "out".to_owned(),
        ty: table_ty,
        space: Space::Storage,
        binding: Some(Binding {
            group: 0,
            binding: 0,
        }),
        access: Access::ReadWrite,
        coherent: false,
    });
    let table_pointer = module.pointer(Space::Storage, table_ty);
    let cell_pointer = module.pointer(Space::Storage, u32_ty);
    let lid = module.define(u32_ty);
    let base = module.define(table_pointer);
    let cell = module.define(cell_pointer);
    let condition = module.define(bool_ty);
    let zero = module.constant(Constant::U32(0));
    let one = module.constant(Constant::U32(1));
    let store = Instruction::Store {
        pointer: cell,
        value: one,
    };
    let nested = vec![Instruction::Loop {
        body: vec![Instruction::Break],
        continuing: Vec::new(),
    }];
    let arm = match placement {
        Placement::InANestedLoop => nested,
        _ => vec![store.clone()],
    };
    let first = match placement {
        Placement::InACase => vec![Instruction::Break],
        Placement::UnderACaseBranch => vec![Instruction::If {
            condition,
            accept: vec![Instruction::Break],
            reject: Vec::new(),
        }],
        _ => arm.clone(),
    };
    let second = match placement {
        Placement::Nowhere => vec![store.clone()],
        _ => arm,
    };
    let entry = module.declare(Function {
        name: "control".to_owned(),
        arguments: vec![Argument {
            name: "lid".to_owned(),
            ty: u32_ty,
            builtin: Some(BuiltIn::LocalInvocationIndex),
        }],
        result: None,
        locals: Vec::new(),
        body: vec![
            Instruction::Argument {
                index: 0,
                result: lid,
            },
            Instruction::Address {
                address: Address::Global(out),
                result: base,
            },
            Instruction::Access {
                base,
                index: lid,
                result: cell,
            },
            Instruction::Binary {
                op: BinaryOp::Equal,
                left: lid,
                right: zero,
                result: condition,
            },
            Instruction::Loop {
                body: vec![Instruction::Switch {
                    selector: lid,
                    cases: vec![(0, first), (1, second)],
                    default: vec![store.clone()],
                }],
                continuing: Vec::new(),
            },
            Instruction::Return { value: None },
        ],
    });
    module.set_entry(entry);
    module.verify();
    module
}

#[test]
fn a_text_writer_chains_the_cases_a_break_leaves_a_loop_through() {
    for placement in [Placement::InACase, Placement::UnderACaseBranch] {
        let module = control(placement);
        let hlsl = neura_shader::text::write(&module, Language::HLSL);
        assert!(!hlsl.contains("switch ("), "{hlsl}");
        assert!(hlsl.contains("else if"), "{hlsl}");
        let loop_at = hlsl.find("while (true)").expect("a device loop");
        let break_at = hlsl.find("break;").expect("a device break");
        assert!(loop_at < break_at, "{hlsl}");
        let msl = neura_shader::text::write(&module, Language::MSL);
        assert!(!msl.contains("switch ("), "{msl}");
        assert!(msl.contains("else if"), "{msl}");
        let loop_at = msl.find("while (true)").expect("a device loop");
        let break_at = msl.find("break;").expect("a device break");
        assert!(loop_at < break_at, "{msl}");
    }
}

#[test]
fn a_text_writer_keeps_the_cases_of_a_switch_no_break_leaves() {
    for placement in [Placement::InANestedLoop, Placement::Nowhere] {
        let module = control(placement);
        let hlsl = neura_shader::text::write(&module, Language::HLSL);
        assert!(hlsl.contains("switch ("), "{hlsl}");
        assert!(!hlsl.contains("else if"), "{hlsl}");
        let msl = neura_shader::text::write(&module, Language::MSL);
        assert!(msl.contains("switch ("), "{msl}");
        assert!(!msl.contains("else if"), "{msl}");
    }
}
