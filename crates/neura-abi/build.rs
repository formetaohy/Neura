use naga::front::wgsl::parse_str;
use naga::{
    Expression, Handle, Literal, Module, Scalar, ScalarKind, StructMember, Type, TypeInner,
    VectorSize,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

const ABI: &str = "abi/program.wgsl";
const CONSTANTS: &str = "constants.rs";
const RECORDS: &str = "records.rs";
const RECORD_SUFFIX: &str = "Record";

fn main() {
    println!("cargo:rerun-if-changed={ABI}");
    let source = fs::read_to_string(ABI).expect("the abi source is readable");
    let module = parse_str(&source)
        .unwrap_or_else(|error| panic!("{}", error.emit_to_string_with_path(&source, ABI)));
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    fs::write(out.join(CONSTANTS), constants(&module)).expect("the constants are writable");
    fs::write(out.join(RECORDS), records(&module)).expect("the records are writable");
}

fn constants(module: &Module) -> String {
    let mut values = BTreeMap::new();
    for (_, constant) in module.constants.iter() {
        let name = constant
            .name
            .clone()
            .unwrap_or_else(|| panic!("the abi source declares an unnamed constant"));
        let value = match module.global_expressions[constant.init] {
            Expression::Literal(Literal::U32(value)) => value,
            ref other => panic!("the abi constant {name} does not fold to a u32: {other:?}"),
        };
        values.insert(name, value);
    }
    let mut output = String::new();
    for (name, value) in values {
        writeln!(output, "pub const {name}: u32 = {value};").unwrap();
    }
    output
}

fn records(module: &Module) -> String {
    let mut records = BTreeMap::new();
    for (_, ty) in module.types.iter() {
        let TypeInner::Struct { members, span } = &ty.inner else {
            continue;
        };
        let name = ty
            .name
            .clone()
            .unwrap_or_else(|| panic!("the abi source declares an unnamed struct"));
        records.insert(name, (members.clone(), *span));
    }
    let mut output = String::new();
    for (name, (members, span)) in records {
        emit_record(module, &mut output, &name, &members, span);
    }
    output
}

fn emit_record(
    module: &Module,
    output: &mut String,
    name: &str,
    members: &[StructMember],
    span: u32,
) {
    writeln!(output, "#[repr(C)]").unwrap();
    writeln!(
        output,
        "#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod)]"
    )
    .unwrap();
    writeln!(output, "pub struct {name}{RECORD_SUFFIX} {{").unwrap();
    let mut offset = 0u32;
    let mut pads = 0;
    let mut fields = Vec::new();
    for member in members {
        let field = member
            .name
            .as_deref()
            .unwrap_or_else(|| panic!("a field of {name} is unnamed"));
        assert!(
            offset <= member.offset,
            "field {field} of {name} cannot be laid out at {} bytes: the rust field order puts it at {offset}",
            member.offset,
        );
        if offset < member.offset {
            let gap = member.offset - offset;
            writeln!(output, "    _wgsl_pad{pads}: [u8; {gap}],").unwrap();
            pads += 1;
            offset += gap;
        }
        let rust = rust_type(module, member.ty, name, field);
        writeln!(output, "    pub {field}: {rust},").unwrap();
        fields.push((field.to_owned(), rust));
        offset += rust_size(module, member.ty, name, field);
    }
    assert!(
        offset <= span,
        "the rust fields of {name} outrun the {span} byte shader struct",
    );
    if offset < span {
        writeln!(output, "    _wgsl_pad{pads}: [u8; {}],", span - offset).unwrap();
    }
    writeln!(output, "}}").unwrap();
    writeln!(output).unwrap();
    emit_constructor(output, name, &fields, pads);
}

fn emit_constructor(output: &mut String, name: &str, fields: &[(String, String)], pads: usize) {
    writeln!(output, "#[derive(Clone, Copy, Debug, PartialEq)]").unwrap();
    writeln!(output, "pub struct {name}Fields {{").unwrap();
    for (field, rust) in fields {
        writeln!(output, "    pub {field}: {rust},").unwrap();
    }
    writeln!(output, "}}").unwrap();
    writeln!(output).unwrap();
    writeln!(output, "impl {name}{RECORD_SUFFIX} {{").unwrap();
    writeln!(output, "    pub fn of(fields: {name}Fields) -> Self {{").unwrap();
    writeln!(output, "        Self {{").unwrap();
    for (field, _) in fields {
        writeln!(output, "            {field}: fields.{field},").unwrap();
    }
    for pad in 0..pads {
        writeln!(output, "            _wgsl_pad{pad}: [0; _],").unwrap();
    }
    writeln!(output, "        }}").unwrap();
    writeln!(output, "    }}").unwrap();
    writeln!(output, "}}").unwrap();
    writeln!(output).unwrap();
    writeln!(
        output,
        "unsafe impl bytemuck::Zeroable for {name}{RECORD_SUFFIX} {{"
    )
    .unwrap();
    writeln!(output, "    fn zeroed() -> Self {{").unwrap();
    writeln!(output, "        Self::of({name}Fields {{").unwrap();
    for (field, rust) in fields {
        writeln!(output, "            {field}: {},", zero_literal(rust)).unwrap();
    }
    writeln!(output, "        }})").unwrap();
    writeln!(output, "    }}").unwrap();
    writeln!(output, "}}").unwrap();
    writeln!(output).unwrap();
}

fn zero_literal(rust: &str) -> &'static str {
    match rust {
        "[u32; 4]" => "[0; 4]",
        "f32" => "0.0",
        _ => "0",
    }
}

fn rust_type(module: &Module, ty: Handle<Type>, owner: &str, field: &str) -> String {
    match &module.types[ty].inner {
        TypeInner::Scalar(Scalar {
            kind: ScalarKind::Uint,
            width: 4,
        }) => "u32".to_owned(),
        TypeInner::Scalar(Scalar {
            kind: ScalarKind::Sint,
            width: 4,
        }) => "i32".to_owned(),
        TypeInner::Scalar(Scalar {
            kind: ScalarKind::Float,
            width: 4,
        }) => "f32".to_owned(),
        TypeInner::Vector {
            size: VectorSize::Quad,
            scalar:
                Scalar {
                    kind: ScalarKind::Uint,
                    width: 4,
                },
        } => "[u32; 4]".to_owned(),
        ref other => panic!("field {field} of {owner} holds an unsupported shader type {other:?}"),
    }
}

fn rust_size(module: &Module, ty: Handle<Type>, owner: &str, field: &str) -> u32 {
    match &module.types[ty].inner {
        TypeInner::Scalar(_) => 4,
        TypeInner::Vector {
            size: VectorSize::Quad,
            ..
        } => 16,
        ref other => panic!("field {field} of {owner} holds an unsupported shader type {other:?}"),
    }
}
