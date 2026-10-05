use crate::hlsl::source::Source;
use crate::instruction::leaves_a_loop;
use crate::{
    Address, AtomicOp, Barrier, Constant, Function, Instruction, MathFun, Module, Scalar, Space,
    Target, Type, TypeId, UnaryOp, ValueId,
};
use std::collections::{HashMap, HashSet};

pub fn write(module: &Module) -> String {
    Target::HLSL.supports(module.requirements(), "the D3D12 device program");
    Writer::new(module).run()
}

pub fn symbol(identifier: &str) -> String {
    format!("d_{identifier}")
}

fn scalar_name(scalar: Scalar) -> &'static str {
    match scalar {
        Scalar::U32 => "uint",
        Scalar::I32 => "int",
        Scalar::F32 => "float",
        Scalar::F16 => "half",
        Scalar::Bool => "bool",
    }
}

struct Writer<'m> {
    module: &'m Module,
    out: Source,
    declarations: HashMap<ValueId, &'m Instruction>,
    names: HashMap<ValueId, String>,
    loops: Vec<String>,
}

impl<'m> Writer<'m> {
    fn new(module: &'m Module) -> Self {
        let mut declarations = HashMap::new();
        for function in module.functions() {
            collect(&function.body, &mut declarations);
        }
        Self {
            module,
            out: Source::new(),
            declarations,
            names: HashMap::new(),
            loops: Vec::new(),
        }
    }

    fn run(mut self) -> String {
        self.layouts();
        self.structures();
        self.resources();
        for (index, function) in self.module.functions().iter().enumerate() {
            if index as u32 == self.module.entry_index() {
                continue;
            }
            let prototype = format!("{} {}(", self.result_name(function), symbol(&function.name));
            let parameters = self.parameters(function);
            self.out.line(format!("{prototype}{parameters});"));
        }
        self.out.line("");
        for index in 0..self.module.functions().len() as u32 {
            if index == self.module.entry_index() {
                continue;
            }
            self.function(index);
        }
        self.entry();
        self.out.finish()
    }

    fn result_name(&self, function: &Function) -> String {
        match function.result {
            Some(ty) => self.value_type(ty),
            None => "void".to_owned(),
        }
    }

    fn parameters(&self, function: &Function) -> String {
        function
            .arguments
            .iter()
            .map(|argument| {
                format!(
                    "{} {}",
                    self.value_type(argument.ty),
                    symbol(&argument.name)
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn value_type(&self, ty: TypeId) -> String {
        match self.module.ty(ty) {
            Type::Scalar(scalar) => scalar_name(*scalar).to_owned(),
            Type::Vector { scalar, length } => format!("{}{length}", scalar_name(*scalar)),
            Type::Array { element, count } => {
                let count = count.expect("a device array holds a fixed number of elements");
                format!("{}[{count}]", self.value_type(*element))
            }
            Type::Struct { name, .. } => name.clone(),
            Type::Atomic(scalar) => scalar_name(*scalar).to_owned(),
            Type::Pointer { .. } => panic!("an HLSL shader carries no device pointer type"),
            Type::CooperativeMatrix { .. } => {
                panic!("an HLSL shader cannot carry a cooperative matrix")
            }
        }
    }

    fn element_type(&self, ty: TypeId) -> TypeId {
        match self.module.ty(ty) {
            Type::Array { element, .. } => *element,
            _ => ty,
        }
    }

    fn layouts(&self) {
        for ty in self.module.type_ids() {
            let Type::Struct {
                name: structure,
                members,
                span,
            } = self.module.ty(ty)
            else {
                continue;
            };
            let mut offset = 0u32;
            let mut alignment = 1u32;
            for member in members {
                let align = self.module.alignment(member.ty).clamp(1, 16);
                offset = offset.div_ceil(align) * align;
                assert_eq!(
                    offset, member.offset,
                    "the device struct {structure} member {} sits at {} while HLSL packs it at {offset}",
                    member.name, member.offset
                );
                offset += self.module.size(member.ty);
                alignment = alignment.max(align);
            }
            offset = offset.div_ceil(alignment) * alignment;
            assert_eq!(
                offset, *span,
                "the device struct {structure} spans {span} bytes while HLSL packs it into {offset}"
            );
        }
    }

    fn structures(&mut self) {
        for ty in self.module.type_ids() {
            let Type::Struct {
                name: structure,
                members,
                ..
            } = self.module.ty(ty)
            else {
                continue;
            };
            let declared = structure.clone();
            self.out.open(format!("struct {declared}"));
            for member in members {
                let ty = self.value_type(member.ty);
                let member = symbol(&member.name);
                self.out.line(format!("{ty} {member};"));
            }
            self.out.close("};");
            self.out.line("");
        }
    }

    fn resources(&mut self) {
        for global in self.module.globals() {
            let global_name = symbol(&global.name);
            match global.space {
                Space::Storage => {
                    let declared = self.value_type(self.element_type(global.ty));
                    let binding = global.binding.expect("a storage resource is bound");
                    if global.access.writable() {
                        self.out.line(format!(
                            "globallycoherent RWStructuredBuffer<{declared}> {global_name} : register(u{});",
                            binding.binding
                        ));
                    } else {
                        self.out.line(format!(
                            "StructuredBuffer<{declared}> {global_name} : register(t{});",
                            binding.binding
                        ));
                    }
                }
                Space::WorkGroup => match self.module.ty(global.ty) {
                    Type::Array {
                        element,
                        count: Some(count),
                    } => {
                        let declared = self.value_type(*element);
                        self.out
                            .line(format!("groupshared {declared} {global_name}[{count}];"));
                    }
                    other => panic!(
                        "the HLSL workgroup variable {} is {}",
                        global.name,
                        crate::element_name(other)
                    ),
                },
                Space::Function => panic!("a device module global lives in function memory"),
            }
        }
        self.out.line("");
    }

    fn function(&mut self, index: u32) {
        let function = &self.module.functions()[index as usize];
        let result = self.result_name(function);
        let parameters = self.parameters(function);
        self.out
            .open(format!("{result} {}({parameters})", symbol(&function.name)));
        self.locals(function);
        self.prepare(function);
        self.block(&function.body);
        self.out.close("}");
        self.out.line("");
    }

    fn entry(&mut self) {
        let entry = self.module.entry();
        let parameters = entry
            .arguments
            .iter()
            .map(|argument| {
                let semantic = match argument.builtin.expect("an entry argument is a builtin") {
                    crate::BuiltIn::LocalInvocationIndex => "SV_GroupIndex",
                    crate::BuiltIn::WorkGroupId => "SV_GroupID",
                    crate::BuiltIn::GlobalInvocationId => "SV_DispatchThreadID",
                };
                format!(
                    "{} {} : {semantic}",
                    self.value_type(argument.ty),
                    symbol(&argument.name)
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        self.out.line(format!(
            "[numthreads({}, 1, 1)]",
            self.module.workgroup_size()
        ));
        self.out
            .open(format!("void {}({parameters})", symbol(&entry.name)));
        self.locals(entry);
        self.prepare(entry);
        self.block(&entry.body);
        self.out.close("}");
    }

    fn locals(&mut self, function: &Function) {
        let names = local_names(self.module, function);
        for (local, name) in function.locals.iter().zip(&names) {
            let declared = self.value_type(local.ty);
            self.out.line(format!("{declared} {name};"));
        }
        self.declare_values(&function.body);
    }

    fn prepare(&mut self, function: &'m Function) {
        self.names.clear();
        for (position, argument) in function.arguments.iter().enumerate() {
            if let Some(value) = argument_value(&function.body, position) {
                self.names.insert(value, symbol(&argument.name));
            }
        }
        let names = local_names(self.module, function);
        addresses(self.module, &names, &function.body, &mut self.names);
    }

    fn declare_values(&mut self, body: &[Instruction]) {
        for instruction in body {
            if let Some(result) = instruction.result()
                && self.hoisted(instruction)
            {
                let declared = self.value_type(self.module.value_ty(result));
                self.out.line(format!("{declared} d_v{};", result.index()));
            }
            match instruction {
                Instruction::If { accept, reject, .. } => {
                    self.declare_values(accept);
                    self.declare_values(reject);
                }
                Instruction::Switch { cases, default, .. } => {
                    for (_, block) in cases {
                        self.declare_values(block);
                    }
                    self.declare_values(default);
                }
                Instruction::Loop { body, continuing } => {
                    self.declare_values(body);
                    self.declare_values(continuing);
                }
                Instruction::Block(block) => self.declare_values(block),
                _ => {}
            }
        }
    }

    fn hoisted(&self, instruction: &Instruction) -> bool {
        match instruction {
            Instruction::Argument { .. } | Instruction::Address { .. } => false,
            Instruction::Access { result, .. } | Instruction::AccessIndex { result, .. } => {
                !matches!(
                    self.module.ty(self.module.value_ty(*result)),
                    Type::Pointer { .. }
                )
            }
            other => other.result().is_some(),
        }
    }

    fn block(&mut self, body: &[Instruction]) {
        for instruction in body {
            self.instruction(instruction);
        }
    }

    fn instruction(&mut self, instruction: &Instruction) {
        match instruction {
            Instruction::Argument { .. } | Instruction::Address { .. } => {}
            Instruction::Access { result, .. } | Instruction::AccessIndex { result, .. } => {
                if self.hoisted(instruction) {
                    let target = self.target(*result);
                    let text = self.place(instruction);
                    self.out.line(format!("{target} = {text};"));
                }
            }
            Instruction::Load { pointer, result } => {
                let target = self.target(*result);
                let text = self.place_value(*pointer);
                self.out.line(format!("{target} = {text};"));
            }
            Instruction::Store { pointer, value } => {
                let text = self.place_value(*pointer);
                let value = self.value(*value);
                self.out.line(format!("{text} = {value};"));
            }
            Instruction::Unary { op, value, result } => {
                let target = self.target(*result);
                let value = self.value(*value);
                let text = match op {
                    UnaryOp::Negate => format!("(-{value})"),
                    UnaryOp::LogicalNot => format!("(!{value})"),
                    UnaryOp::BitwiseNot => format!("(~{value})"),
                };
                self.out.line(format!("{target} = {text};"));
            }
            Instruction::Binary {
                op,
                left,
                right,
                result,
            } => {
                let target = self.target(*result);
                let left = self.value(*left);
                let right = self.value(*right);
                let text = format!("({left} {} {right})", op.name());
                self.out.line(format!("{target} = {text};"));
            }
            Instruction::Select {
                condition,
                accept,
                reject,
                result,
            } => {
                let target = self.target(*result);
                let condition = self.value(*condition);
                let accept = self.value(*accept);
                let reject = self.value(*reject);
                self.out
                    .line(format!("{target} = ({condition} ? {accept} : {reject});"));
            }
            Instruction::Convert { value, result } => {
                let target = self.target(*result);
                let declared = self.value_type(self.module.value_ty(*result));
                let value = self.value(*value);
                self.out.line(format!("{target} = ({declared})({value});"));
            }
            Instruction::Bitcast { value, result } => {
                let target = self.target(*result);
                let value = self.value(*value);
                let conversion = match self.scalar(*result) {
                    Scalar::F32 => "asfloat",
                    Scalar::U32 => "asuint",
                    Scalar::I32 => "asint",
                    other => panic!("an HLSL bitcast cannot reach the {}", other.name()),
                };
                self.out.line(format!("{target} = {conversion}({value});"));
            }
            Instruction::Math {
                fun,
                arguments,
                result,
            } => {
                let target = self.target(*result);
                let text = if *fun == MathFun::UnpackHalf2x16 {
                    let value = self.value(arguments[0]);
                    format!("float2(f16tof32({value}), f16tof32(({value}) >> 16))")
                } else {
                    let arguments = arguments
                        .iter()
                        .map(|argument| self.value(*argument))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{}({arguments})", fun.name())
                };
                self.out.line(format!("{target} = {text};"));
            }
            Instruction::Compose {
                constituents,
                result,
            } => {
                let target = self.target(*result);
                let declared = self.value_type(self.module.value_ty(*result));
                let constituents = constituents
                    .iter()
                    .map(|constituent| self.value(*constituent))
                    .collect::<Vec<_>>()
                    .join(", ");
                self.out
                    .line(format!("{target} = {declared}({constituents});"));
            }
            Instruction::Call {
                function,
                arguments,
                result,
            } => {
                let callee = symbol(&self.module.functions()[*function as usize].name);
                let arguments = arguments
                    .iter()
                    .map(|argument| self.value(*argument))
                    .collect::<Vec<_>>()
                    .join(", ");
                match result {
                    Some(result) => {
                        let target = self.target(*result);
                        self.out.line(format!("{target} = {callee}({arguments});"));
                    }
                    None => self.out.line(format!("{callee}({arguments});")),
                }
            }
            Instruction::Atomic {
                op,
                pointer,
                value,
                result,
            } => {
                let target = self.target(*result);
                let cell = self.place_value(*pointer);
                let value = self.value(*value);
                let call = match op {
                    AtomicOp::Add => format!("InterlockedAdd({cell}, {value}, {target})"),
                    AtomicOp::Subtract => {
                        format!("InterlockedAdd({cell}, (0 - {value}), {target})")
                    }
                    AtomicOp::Min => format!("InterlockedMin({cell}, {value}, {target})"),
                    AtomicOp::Max => format!("InterlockedMax({cell}, {value}, {target})"),
                    AtomicOp::And => format!("InterlockedAnd({cell}, {value}, {target})"),
                    AtomicOp::Or => format!("InterlockedOr({cell}, {value}, {target})"),
                    AtomicOp::Xor => format!("InterlockedXor({cell}, {value}, {target})"),
                    AtomicOp::Exchange => {
                        format!("InterlockedExchange({cell}, {value}, {target})")
                    }
                };
                self.out.line(format!("{call};"));
            }
            Instruction::WorkGroupUniformLoad { pointer, result } => {
                self.barrier(Barrier::WorkGroup);
                let target = self.target(*result);
                let text = self.place_value(*pointer);
                self.out.line(format!("{target} = {text};"));
            }
            Instruction::Barrier(barrier) => self.barrier(*barrier),
            Instruction::If {
                condition,
                accept,
                reject,
            } => {
                let condition = self.value(*condition);
                self.out.open(format!("if ({condition})"));
                self.block(accept);
                self.out.close("}");
                if !reject.is_empty() {
                    self.out.open("else");
                    self.block(reject);
                    self.out.close("}");
                }
            }
            Instruction::Switch {
                selector,
                cases,
                default,
            } => {
                let selector = self.value(*selector);
                if cases.iter().any(|(_, body)| leaves_a_loop(body)) || leaves_a_loop(default) {
                    for (index, (value, body)) in cases.iter().enumerate() {
                        let branch = if index == 0 { "if" } else { "else if" };
                        self.out.open(format!("{branch} ({selector} == {value})"));
                        self.block(body);
                        self.out.close("}");
                    }
                    if !default.is_empty() {
                        if cases.is_empty() {
                            self.block(default);
                        } else {
                            self.out.open("else");
                            self.block(default);
                            self.out.close("}");
                        }
                    }
                } else {
                    self.out.open(format!("switch ({selector})"));
                    for (value, body) in cases {
                        self.out.open(format!("case {value}:"));
                        self.block(body);
                        if !terminates(body) {
                            self.out.line("break;");
                        }
                        self.out.close("}");
                    }
                    self.out.open("default:");
                    self.block(default);
                    if !terminates(default) {
                        self.out.line("break;");
                    }
                    self.out.close("}");
                    self.out.close("}");
                }
            }
            Instruction::Loop { body, continuing } => {
                let step = self.step(continuing);
                self.loops.push(step);
                self.out.open("while (true)");
                self.block(body);
                let step = self.loops.pop().expect("a device loop is open");
                self.out.raw(&step);
                self.out.close("}");
            }
            Instruction::Break => self.out.line("break;"),
            Instruction::Continue => {
                let step = self.loops.last().cloned().unwrap_or_default();
                if step.is_empty() {
                    self.out.line("continue;");
                } else {
                    self.out.line("{");
                    self.out.raw(&step);
                    self.out.line("    continue;");
                    self.out.line("}");
                }
            }
            Instruction::Return { value } => match value {
                Some(value) => {
                    let value = self.value(*value);
                    self.out.line(format!("return {value};"));
                }
                None => self.out.line("return;"),
            },
            Instruction::Block(body) => self.block(body),
            Instruction::MatrixFill { .. }
            | Instruction::MatrixLoad { .. }
            | Instruction::MatrixStore { .. }
            | Instruction::MatrixMulAdd { .. }
            | Instruction::MatrixLength { .. }
            | Instruction::MatrixExtract { .. }
            | Instruction::MatrixInsert { .. } => {
                panic!("an HLSL shader cannot carry a cooperative matrix")
            }
        }
    }

    fn step(&mut self, continuing: &[Instruction]) -> String {
        let out = std::mem::replace(&mut self.out, Source::new());
        self.block(continuing);
        std::mem::replace(&mut self.out, out).finish()
    }

    fn barrier(&mut self, barrier: Barrier) {
        match barrier {
            Barrier::WorkGroup => self.out.line("GroupMemoryBarrierWithGroupSync();"),
            Barrier::Storage => self.out.line("DeviceMemoryBarrierWithGroupSync();"),
        }
    }

    fn target(&self, value: ValueId) -> String {
        format!("d_v{}", value.index())
    }

    fn scalar(&self, value: ValueId) -> Scalar {
        let ty = self.module.value_ty(value);
        match self.module.ty(ty) {
            Type::Scalar(scalar) => *scalar,
            Type::Vector { scalar, .. } => *scalar,
            other => panic!(
                "the device value {} is {}",
                value.index(),
                crate::element_name(other)
            ),
        }
    }

    fn value(&self, value: ValueId) -> String {
        if let Some(text) = self.names.get(&value) {
            return text.clone();
        }
        match self.module.constant_of(value) {
            Some(Constant::U32(number)) => format!("{number}u"),
            Some(Constant::I32(number)) => format!("{number}"),
            Some(Constant::F32(number)) => {
                assert!(
                    number.is_finite(),
                    "a device program carries the non-finite literal {number}"
                );
                if number == number.trunc() && number.abs() < 1e7 {
                    format!("{number:.1}")
                } else if number == 0.0 {
                    "0.0".to_owned()
                } else {
                    format!("{number:e}")
                }
            }
            Some(Constant::Bool(flag)) => format!("{flag}"),
            Some(Constant::Zero(ty)) => {
                let declared = self.value_type(ty);
                format!("({declared})0")
            }
            None => format!("d_v{}", value.index()),
        }
    }

    fn place(&self, instruction: &Instruction) -> String {
        match instruction {
            Instruction::Access { base, index, .. } => {
                let base = self.place_value(*base);
                let index = self.value(*index);
                format!("{base}[{index}]")
            }
            Instruction::AccessIndex { base, index, .. } => {
                let text = self.place_value(*base);
                let pointee = self.module.ty(self.module.value_ty(*base));
                let member = match pointee {
                    Type::Pointer { base: pointed, .. } => match self.module.ty(*pointed) {
                        Type::Struct { members, .. } => {
                            Some(symbol(&members[*index as usize].name))
                        }
                        _ => None,
                    },
                    Type::Struct { members, .. } => Some(symbol(&members[*index as usize].name)),
                    _ => None,
                };
                match member {
                    Some(member) => format!("{text}.{member}"),
                    None => format!("{text}[{index}]"),
                }
            }
            other => panic!("the device instruction {other:?} is not a place"),
        }
    }

    fn place_value(&self, value: ValueId) -> String {
        match self.declarations.get(&value) {
            Some(instruction @ (Instruction::Access { .. } | Instruction::AccessIndex { .. })) => {
                self.place(instruction)
            }
            Some(Instruction::Address {
                address: Address::Global(index),
                ..
            }) => {
                let global = self.module.global(*index);
                let named = symbol(&global.name);
                if matches!(self.module.ty(global.ty), Type::Array { .. }) {
                    named
                } else {
                    format!("{named}[0]")
                }
            }
            _ => self.value(value),
        }
    }
}

fn argument_value(body: &[Instruction], index: usize) -> Option<ValueId> {
    body.iter().find_map(|instruction| match instruction {
        Instruction::Argument {
            index: present,
            result,
        } if *present as usize == index => Some(*result),
        _ => None,
    })
}

fn terminates(body: &[Instruction]) -> bool {
    match body.last() {
        Some(Instruction::Return { .. } | Instruction::Break | Instruction::Continue) => true,
        Some(Instruction::If { accept, reject, .. }) => terminates(accept) && terminates(reject),
        Some(Instruction::Switch { cases, default, .. }) => {
            cases.iter().all(|(_, body)| terminates(body)) && terminates(default)
        }
        _ => false,
    }
}

fn collect<'m>(body: &'m [Instruction], declarations: &mut HashMap<ValueId, &'m Instruction>) {
    for instruction in body {
        if let Some(result) = instruction.result() {
            declarations.insert(result, instruction);
        }
        match instruction {
            Instruction::If { accept, reject, .. } => {
                collect(accept, declarations);
                collect(reject, declarations);
            }
            Instruction::Switch { cases, default, .. } => {
                for (_, body) in cases {
                    collect(body, declarations);
                }
                collect(default, declarations);
            }
            Instruction::Loop { body, continuing } => {
                collect(body, declarations);
                collect(continuing, declarations);
            }
            Instruction::Block(body) => collect(body, declarations),
            _ => {}
        }
    }
}

fn addresses<'m>(
    module: &'m Module,
    locals: &[String],
    body: &'m [Instruction],
    names: &mut HashMap<ValueId, String>,
) {
    for instruction in body {
        match instruction {
            Instruction::Address {
                address: Address::Local(index),
                result,
            } => {
                names.insert(*result, locals[*index as usize].clone());
            }
            Instruction::Address {
                address: Address::Global(index),
                result,
            } => {
                names.insert(*result, symbol(&module.global(*index).name));
            }
            Instruction::If { accept, reject, .. } => {
                addresses(module, locals, accept, names);
                addresses(module, locals, reject, names);
            }
            Instruction::Switch { cases, default, .. } => {
                for (_, block) in cases {
                    addresses(module, locals, block, names);
                }
                addresses(module, locals, default, names);
            }
            Instruction::Loop { body, continuing } => {
                addresses(module, locals, body, names);
                addresses(module, locals, continuing, names);
            }
            Instruction::Block(block) => addresses(module, locals, block, names),
            _ => {}
        }
    }
}

fn local_names(module: &Module, function: &Function) -> Vec<String> {
    let mut used = function
        .arguments
        .iter()
        .map(|argument| symbol(&argument.name))
        .chain(module.globals().iter().map(|global| symbol(&global.name)))
        .collect::<HashSet<_>>();
    function
        .locals
        .iter()
        .enumerate()
        .map(|(index, local)| {
            let base = symbol(&local.name);
            let mut name = base.clone();
            let mut count = 0;
            while !used.insert(name.clone()) {
                count += 1;
                name = format!("{base}_{index}_{count}");
            }
            name
        })
        .collect()
}
