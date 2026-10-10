use neura_shader::{
    Access, Address, Argument, BinaryOp, Binding, BuiltIn, Constant, Function, Global, Instruction,
    Language, Module, Scalar, Space, ValueId,
};
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};

const OP_CONSTANT: u16 = 43;
const OP_SELECT: u16 = 169;
const OP_I_EQUAL: u16 = 170;

struct Choosing {
    module: Module,
    condition: ValueId,
    chosen: ValueId,
}

fn choosing() -> Choosing {
    let mut module = Module::new(2);
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
    let chosen = module.define(u32_ty);
    let accept = module.constant(Constant::U32(7));
    let reject = module.constant(Constant::U32(9));
    let zero = module.constant(Constant::U32(0));
    let entry = module.declare(Function {
        name: "choosing".to_owned(),
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
            Instruction::Select {
                condition,
                accept,
                reject,
                result: chosen,
            },
            Instruction::Store {
                pointer: cell,
                value: chosen,
            },
            Instruction::Return { value: None },
        ],
    });
    module.set_entry(entry);
    module.verify();
    Choosing {
        module,
        condition,
        chosen,
    }
}

fn conditioned_on_a_vector() -> Module {
    let mut module = Module::new(2);
    let u32_ty = module.scalar(Scalar::U32);
    let bools = module.vector(Scalar::Bool, 2);
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
    let entry = module.declare(Function {
        name: "entry".to_owned(),
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
            Instruction::Store {
                pointer: cell,
                value: lid,
            },
            Instruction::Return { value: None },
        ],
    });
    module.set_entry(entry);
    let flag = module.define(bools);
    let picked = module.define(u32_ty);
    let accept = module.constant(Constant::U32(7));
    let reject = module.constant(Constant::U32(9));
    module.declare(Function {
        name: "picking".to_owned(),
        arguments: vec![Argument {
            name: "flag".to_owned(),
            ty: bools,
            builtin: None,
        }],
        result: Some(u32_ty),
        locals: Vec::new(),
        body: vec![
            Instruction::Argument {
                index: 0,
                result: flag,
            },
            Instruction::Select {
                condition: flag,
                accept,
                reject,
                result: picked,
            },
            Instruction::Return {
                value: Some(picked),
            },
        ],
    });
    module
}

fn instructions(words: &[u32]) -> Vec<(u16, usize)> {
    let mut found = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let header = words[at];
        let count = ((header >> 16) as usize).max(1);
        found.push(((header & 0xffff) as u16, at));
        at += count;
    }
    found
}

#[test]
fn every_backend_selects_the_branch_its_condition_accepts() {
    let choosing = choosing();
    let expected = format!(
        "d_v{} = (d_v{} ? 7u : 9u);",
        choosing.chosen.index(),
        choosing.condition.index(),
    );
    let hlsl = neura_shader::text::write(&choosing.module, Language::HLSL);
    assert!(
        hlsl.contains(&expected),
        "the HLSL of a select hands its condition the rejected branch:\n{hlsl}",
    );
    let msl = neura_shader::text::write(&choosing.module, Language::MSL);
    assert!(
        msl.contains(&expected),
        "the MSL of a select hands its condition the rejected branch:\n{msl}",
    );
    assert!(
        !msl.contains("metal::select"),
        "the MSL of a select calls a dialect function whose argument order is the reverse of the device select:\n{msl}",
    );
}

#[test]
fn the_vulkan_device_program_selects_the_accepted_branch_first() {
    let choosing = choosing();
    let words = neura_shader::spirv::write(&choosing.module);
    let mut constants = HashMap::new();
    for (opcode, at) in instructions(&words) {
        if opcode == OP_CONSTANT {
            constants.insert(words[at + 3], words[at + 2]);
        }
    }
    let accept = constants[&7];
    let reject = constants[&9];
    let condition = instructions(&words)
        .into_iter()
        .find_map(|(opcode, at)| (opcode == OP_I_EQUAL).then_some(words[at + 2]))
        .expect("a device comparison conditions the select");
    let select = instructions(&words)
        .into_iter()
        .find_map(|(opcode, at)| (opcode == OP_SELECT).then_some(at))
        .expect("a select reaches the Vulkan device program");
    assert_eq!(
        [words[select + 3], words[select + 4], words[select + 5]],
        [condition, accept, reject],
        "the Vulkan device program reads its select operands in another order",
    );
}

#[test]
fn the_device_ir_conditions_a_select_on_one_scalar() {
    let module = conditioned_on_a_vector();
    let refused = catch_unwind(AssertUnwindSafe(|| module.verify()))
        .expect_err("a select is conditioned on one scalar");
    let message = refused
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "a refusal without a message".to_owned());
    assert!(
        message.contains("a device select") && message.contains("boolx2"),
        "the device IR conditions a select on {message}",
    );
}
