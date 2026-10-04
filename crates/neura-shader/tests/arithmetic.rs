use neura_shader::{
    Access, Address, Argument, BinaryOp, Binding, BuiltIn, Constant, Function, Global, Instruction,
    IntegerArithmetic, Module, Scalar, Space,
};

fn arithmetic(op: BinaryOp, scalar: Scalar) -> Module {
    let mut module = Module::new(2);
    let arithmetic_ty = module.scalar(scalar);
    let table_ty = module.array(arithmetic_ty, None);
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
    let cell_pointer = module.pointer(Space::Storage, arithmetic_ty);
    let (left, right) = match scalar {
        Scalar::I32 => (
            module.constant(Constant::I32(-7)),
            module.constant(Constant::I32(3)),
        ),
        _ => (
            module.constant(Constant::U32(0xffff_fff9)),
            module.constant(Constant::U32(3)),
        ),
    };
    let mut body = Vec::new();
    let lid_ty = module.scalar(Scalar::U32);
    let lid = module.define(lid_ty);
    body.push(Instruction::Argument {
        index: 0,
        result: lid,
    });
    let base = module.define(table_pointer);
    body.push(Instruction::Address {
        address: Address::Global(out),
        result: base,
    });
    let cell = module.define(cell_pointer);
    body.push(Instruction::Access {
        base,
        index: lid,
        result: cell,
    });
    let result = module.define(arithmetic_ty);
    body.push(Instruction::Binary {
        op,
        left,
        right,
        result,
    });
    body.push(Instruction::Store {
        pointer: cell,
        value: result,
    });
    body.push(Instruction::Return { value: None });
    let entry = module.declare(Function {
        name: "arithmetic".to_owned(),
        arguments: vec![Argument {
            name: "lid".to_owned(),
            ty: lid_ty,
            builtin: Some(BuiltIn::LocalInvocationIndex),
        }],
        result: None,
        locals: Vec::new(),
        body,
    });
    module.set_entry(entry);
    module.verify();
    module
}

fn opcodes(words: &[u32]) -> Vec<u16> {
    let mut opcodes = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let word = words[at];
        opcodes.push((word & 0xffff) as u16);
        at += ((word >> 16) as usize).max(1);
    }
    opcodes
}

#[test]
fn the_integer_arithmetic_of_the_ir_truncates_and_follows_the_dividend() {
    assert_eq!(
        BinaryOp::Divide.integer_arithmetic(Scalar::I32),
        Some(IntegerArithmetic::TruncatedQuotient),
    );
    assert_eq!(
        BinaryOp::Divide.integer_arithmetic(Scalar::U32),
        Some(IntegerArithmetic::UnsignedQuotient),
    );
    assert_eq!(
        BinaryOp::Modulo.integer_arithmetic(Scalar::I32),
        Some(IntegerArithmetic::TruncatedRemainder),
    );
    assert_eq!(
        BinaryOp::Modulo.integer_arithmetic(Scalar::U32),
        Some(IntegerArithmetic::UnsignedRemainder),
    );
    assert_eq!(BinaryOp::Modulo.integer_arithmetic(Scalar::F32), None);
    assert_eq!(BinaryOp::Add.integer_arithmetic(Scalar::I32), None);
}

#[test]
fn spirv_takes_a_signed_remainder_without_a_remainder_instruction() {
    let words = neura_shader::spirv::write(&arithmetic(BinaryOp::Modulo, Scalar::I32));
    let opcodes = opcodes(&words);
    for expected in [135u16, 132, 130] {
        assert!(
            opcodes.contains(&expected),
            "a signed remainder on the Vulkan environment is a quotient, a product and a difference, and the program holds no {expected}",
        );
    }
    for forbidden in [137u16, 138, 139, 140, 141] {
        assert!(
            !opcodes.contains(&forbidden),
            "the Vulkan environment leaves the integer remainders to the implementation when an operand is negative, and the program holds {forbidden}",
        );
    }
}

#[test]
fn spirv_keeps_an_unsigned_remainder_unsigned() {
    let words = neura_shader::spirv::write(&arithmetic(BinaryOp::Modulo, Scalar::U32));
    let opcodes = opcodes(&words);
    assert!(opcodes.contains(&137));
    assert!(!opcodes.contains(&135));
}

#[test]
fn the_text_writers_take_the_remainder_of_the_dividend() {
    let module = arithmetic(BinaryOp::Modulo, Scalar::I32);
    let hlsl = neura_shader::hlsl::write(&module);
    assert!(hlsl.contains("(-7 % 3)"), "{hlsl}");
    assert!(!hlsl.contains("fmod"), "{hlsl}");
    let msl = neura_shader::msl::write(&module);
    assert!(msl.contains("(-7 % 3)"), "{msl}");
    assert!(!msl.contains("fmod"), "{msl}");
}
