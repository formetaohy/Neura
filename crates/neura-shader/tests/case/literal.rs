use neura_shader::{
    Access, Address, Argument, Binding, BuiltIn, Constant, Function, Global, Instruction, Language,
    Module, Scalar, Space,
};

const CONSTANTS: [f32; 10] = [
    0.0,
    -0.0,
    0.5,
    0.7978846,
    4.4715e-2,
    1.0e-45,
    3.4028235e38,
    1.2345678e7,
    12345.678,
    1.0e-8,
];

const SPELLINGS: [&str; 10] = [
    "0.0f",
    "-0.0f",
    "5e-1f",
    "7.978846e-1f",
    "4.4715e-2f",
    "1e-45f",
    "3.4028235e38f",
    "1.2345678e7f",
    "1.2345678e4f",
    "1e-8f",
];

fn module_of(constants: &[f32]) -> Module {
    let mut module = Module::new(1);
    let number = module.scalar(Scalar::F32);
    let table = module.array(number, None);
    let out = module.add_global(Global {
        name: "out".to_owned(),
        ty: table,
        space: Space::Storage,
        binding: Some(Binding {
            group: 0,
            binding: 0,
        }),
        access: Access::ReadWrite,
        coherent: false,
    });
    let table_pointer = module.pointer(Space::Storage, table);
    let cell_pointer = module.pointer(Space::Storage, number);
    let lid_type = module.scalar(Scalar::U32);
    let mut body = Vec::new();
    let lid = module.define(lid_type);
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
    for constant in constants {
        let value = module.constant(Constant::F32(*constant));
        body.push(Instruction::Store {
            pointer: cell,
            value,
        });
    }
    body.push(Instruction::Return { value: None });
    let entry = module.declare(Function {
        name: "spell".to_owned(),
        arguments: vec![Argument {
            name: "lid".to_owned(),
            ty: lid_type,
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

fn floats(text: &str) -> Vec<String> {
    let bytes = text.as_bytes();
    let word = |byte: u8| byte.is_ascii_alphanumeric() || byte == b'_';
    let mut found = Vec::new();
    let mut at = 0;
    while at < bytes.len() {
        let starts = bytes[at].is_ascii_digit() && (at == 0 || !word(bytes[at - 1]))
            || bytes[at] == b'-'
                && at + 1 < bytes.len()
                && bytes[at + 1].is_ascii_digit()
                && (at == 0 || !word(bytes[at - 1]));
        if !starts {
            at += 1;
            continue;
        }
        let mut cursor = at + usize::from(bytes[at] == b'-');
        while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
            cursor += 1;
        }
        let mut fractional = false;
        if cursor < bytes.len() && bytes[cursor] == b'.' {
            fractional = true;
            cursor += 1;
            while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                cursor += 1;
            }
        }
        if cursor < bytes.len() && matches!(bytes[cursor], b'e' | b'E') {
            let mut exponent = cursor + 1;
            if exponent < bytes.len() && matches!(bytes[exponent], b'+' | b'-') {
                exponent += 1;
            }
            if exponent < bytes.len() && bytes[exponent].is_ascii_digit() {
                fractional = true;
                cursor = exponent;
                while cursor < bytes.len() && bytes[cursor].is_ascii_digit() {
                    cursor += 1;
                }
            }
        }
        if !fractional {
            at = cursor;
            continue;
        }
        let suffix = if cursor < bytes.len() {
            bytes[cursor] as char
        } else {
            '\0'
        };
        found.push(format!("{}{suffix}", &text[at..cursor]));
        at = cursor + 1;
    }
    found
}

#[test]
fn both_text_writers_spell_every_f32_constant_as_a_float_literal() {
    let module = module_of(&CONSTANTS);
    for written in [
        neura_shader::text::write(&module, Language::HLSL),
        neura_shader::text::write(&module, Language::MSL),
    ] {
        let spelled = floats(&written);
        assert_eq!(
            spelled.len(),
            SPELLINGS.len(),
            "every constant reaches the shader once: {spelled:?}"
        );
        for (found, expected) in spelled.iter().zip(SPELLINGS) {
            assert_eq!(
                found, expected,
                "an f32 constant is spelled as the target's single precision literal"
            );
        }
    }
}
