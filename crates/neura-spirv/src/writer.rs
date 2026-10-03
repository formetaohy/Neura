use crate::body::FunctionWriter;
use crate::op::*;
use neura_shader_ir::{
    BuiltIn, Constant, Instruction as DeviceInstruction, Module, Scalar, Space, Target, Type,
    TypeId, ValueId,
};
use std::collections::{BTreeMap, HashMap, HashSet};

pub fn write(module: &Module) -> Vec<u32> {
    Target::SPIRV.supports(module.requirements(), "the Vulkan device program");
    Writer::new(module).run()
}

pub(crate) struct Global {
    pub(crate) id: u32,
    pub(crate) pointer: u32,
    pub(crate) wrapped: bool,
}

pub(crate) struct Writer<'m> {
    pub(crate) module: &'m Module,
    next: u32,
    pub(crate) version: u32,
    pub(crate) vulkan_memory_model: bool,
    types: HashMap<TypeId, u32>,
    pub(crate) values: HashMap<ValueId, u32>,
    pub(crate) value_types: HashMap<ValueId, u32>,
    pub(crate) declared: HashMap<ValueId, &'m DeviceInstruction>,
    pub(crate) arguments: Vec<Vec<ValueId>>,
    pub(crate) locals: Vec<Vec<(u32, u32)>>,
    pub(crate) function_ids: Vec<u32>,
    pub(crate) globals: Vec<Global>,
    builtins: BTreeMap<u32, u32>,
    used: Vec<HashSet<u32>>,
    literals: HashMap<(u32, u64), u32>,
    glsl: Option<u32>,
    void_type: Option<u32>,
    defined: HashSet<u32>,
    function_types: HashMap<(u32, Vec<u32>), u32>,
    header: Vec<Instruction>,
    debug: Vec<Instruction>,
    annotations: Vec<Instruction>,
    declarations: Vec<Instruction>,
    definitions: Vec<Instruction>,
}

impl<'m> Writer<'m> {
    fn new(module: &'m Module) -> Self {
        let cooperative = module.requirements().cooperative_matrix;
        Self {
            module,
            next: 1,
            version: if cooperative {
                VERSION_1_6
            } else {
                VERSION_1_3
            },
            vulkan_memory_model: cooperative,
            types: HashMap::new(),
            values: HashMap::new(),
            value_types: HashMap::new(),
            declared: HashMap::new(),
            arguments: vec![Vec::new(); module.functions().len()],
            locals: vec![Vec::new(); module.functions().len()],
            function_ids: Vec::new(),
            globals: Vec::new(),
            builtins: BTreeMap::new(),
            used: vec![HashSet::new(); module.functions().len()],
            literals: HashMap::new(),
            glsl: None,
            void_type: None,
            defined: HashSet::new(),
            function_types: HashMap::new(),
            header: Vec::new(),
            debug: Vec::new(),
            annotations: Vec::new(),
            declarations: Vec::new(),
            definitions: Vec::new(),
        }
    }

    pub(crate) fn id(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }

    pub(crate) fn function_type(&mut self, result: Option<u32>, parameters: &[u32]) -> u32 {
        let void = self.void();
        let key = (result.unwrap_or(void), parameters.to_vec());
        if let Some(id) = self.function_types.get(&key) {
            return *id;
        }
        let id = self.id();
        let mut instruction = Instruction::defined(TYPE_FUNCTION, id).operand(key.0);
        for parameter in parameters {
            instruction = instruction.operand(*parameter);
        }
        self.declare(instruction);
        self.function_types.insert(key, id);
        id
    }

    pub(crate) fn void(&mut self) -> u32 {
        if let Some(id) = self.void_type {
            return id;
        }
        let id = self.id();
        self.declare(Instruction::defined(TYPE_VOID, id));
        self.void_type = Some(id);
        id
    }

    fn run(mut self) -> Vec<u32> {
        self.intern_types();
        self.declare_globals();
        self.collect();
        self.intern_constants();
        self.declare_builtins();
        self.usage();
        self.capabilities();
        self.entry_point();
        for index in 0..self.module.functions().len() as u32 {
            FunctionWriter::new(&mut self, index).run();
        }
        self.assemble()
    }

    fn intern_types(&mut self) {
        let types = self.module.type_ids().collect::<Vec<_>>();
        for ty in types {
            self.ty(ty);
        }
    }

    pub(crate) fn ty(&mut self, ty: TypeId) -> u32 {
        if let Some(id) = self.types.get(&ty) {
            return *id;
        }
        let id = match self.module.ty(ty).clone() {
            Type::Scalar(scalar) => self.scalar_type(scalar),
            Type::Vector { scalar, length } => {
                let component = self.scalar_type(scalar);
                let id = self.id();
                self.declare(
                    Instruction::defined(TYPE_VECTOR, id)
                        .operand(component)
                        .word(length),
                );
                id
            }
            Type::Array { element, count } => {
                let element = self.ty(element);
                let id = self.id();
                let instruction = match count {
                    Some(count) => {
                        let count = self.integer(Scalar::U32, count);
                        Instruction::defined(TYPE_ARRAY, id)
                            .operand(element)
                            .operand(count)
                    }
                    None => Instruction::defined(TYPE_RUNTIME_ARRAY, id).operand(element),
                };
                self.declare(instruction);
                id
            }
            Type::Struct { members, .. } => {
                let mut instruction = Instruction::defined(TYPE_STRUCT, self.id());
                for member in &members {
                    let member = self.ty(member.ty);
                    instruction = instruction.operand(member);
                }
                let id = instruction.operands[0].id();
                self.declare(instruction);
                id
            }
            Type::Atomic(scalar) => self.scalar_type(scalar),
            Type::Pointer { space, base } => {
                let base = self.ty(base);
                let id = self.id();
                self.declare(
                    Instruction::defined(TYPE_POINTER, id)
                        .word(storage_class(space))
                        .operand(base),
                );
                id
            }
            Type::CooperativeMatrix {
                scalar,
                rows,
                columns,
                usage,
            } => {
                let component = self.scalar_type(scalar);
                let scope = self.integer(Scalar::U32, scope::SUBGROUP);
                let rows = self.integer(Scalar::U32, rows);
                let columns = self.integer(Scalar::U32, columns);
                let usage = self.integer(Scalar::U32, usage.code());
                let id = self.id();
                self.declare(
                    Instruction::defined(COOPERATIVE_MATRIX_TYPE, id)
                        .operand(component)
                        .operand(scope)
                        .operand(rows)
                        .operand(columns)
                        .operand(usage),
                );
                id
            }
        };
        self.types.insert(ty, id);
        id
    }

    fn scalar_type(&mut self, scalar: Scalar) -> u32 {
        if let Some(ty) = self.module.scalar_type(scalar)
            && let Some(id) = self.types.get(&ty)
        {
            return *id;
        }
        let id = self.id();
        let instruction = match scalar {
            Scalar::Bool => Instruction::defined(TYPE_BOOL, id),
            Scalar::U32 | Scalar::I32 => Instruction::defined(TYPE_INT, id)
                .word(32u32)
                .word(u32::from(scalar.signed())),
            Scalar::F32 | Scalar::F16 => Instruction::defined(TYPE_FLOAT, id)
                .word(if scalar == Scalar::F32 { 32u32 } else { 16u32 }),
        };
        self.declare(instruction);
        if let Some(ty) = self.module.scalar_type(scalar) {
            self.types.insert(ty, id);
        }
        id
    }

    pub(crate) fn integer(&mut self, scalar: Scalar, value: u32) -> u32 {
        let ty = self.scalar_type(scalar);
        self.literal(ty, u64::from(value), |instruction| instruction.word(value))
    }

    pub(crate) fn float(&mut self, value: f32) -> u32 {
        let ty = self.scalar_type(Scalar::F32);
        self.literal(ty, u64::from(value.to_bits()), |instruction| {
            instruction.word(value.to_bits())
        })
    }

    pub(crate) fn boolean(&mut self, value: bool) -> u32 {
        let ty = self.scalar_type(Scalar::Bool);
        if let Some(id) = self.literals.get(&(ty, u64::from(value))) {
            return *id;
        }
        let id = self.id();
        let opcode = if value { CONSTANT_TRUE } else { CONSTANT_FALSE };
        self.declare(Instruction::result(opcode, ty, id));
        self.literals.insert((ty, u64::from(value)), id);
        id
    }

    fn literal(
        &mut self,
        ty: u32,
        bits: u64,
        build: impl FnOnce(Instruction) -> Instruction,
    ) -> u32 {
        if let Some(id) = self.literals.get(&(ty, bits)) {
            return *id;
        }
        let id = self.id();
        self.declare(build(Instruction::result(CONSTANT, ty, id)));
        self.literals.insert((ty, bits), id);
        id
    }

    fn zero(&mut self, ty: TypeId) -> u32 {
        let spirv = self.ty(ty);
        let key = (spirv, u64::MAX);
        if let Some(id) = self.literals.get(&key) {
            return *id;
        }
        let id = self.id();
        let instruction = match self.module.ty(ty).clone() {
            Type::CooperativeMatrix { scalar, .. } => {
                let component = match scalar {
                    Scalar::F32 => self.float(0.0),
                    Scalar::F16 | Scalar::U32 | Scalar::I32 => self.integer(scalar, 0),
                    Scalar::Bool => panic!("a cooperative matrix holds no boolean"),
                };
                Instruction::result(CONSTANT_COMPOSITE, spirv, id).operand(component)
            }
            _ => Instruction::result(CONSTANT_NULL, spirv, id),
        };
        self.declare(instruction);
        self.literals.insert(key, id);
        id
    }

    fn declare_globals(&mut self) {
        let mut decorated = Decoration::default();
        for index in 0..self.module.globals().len() as u32 {
            let global = self.module.global(index).clone();
            let (pointee, wrapper) = match global.space {
                Space::WorkGroup => (self.ty(global.ty), false),
                Space::Storage => self.storage_pointee(&global, &mut decorated),
                Space::Function => panic!("a module global cannot live in function memory"),
            };
            let _ = wrapper;
            let pointer = self.id();
            self.declare(
                Instruction::defined(TYPE_POINTER, pointer)
                    .word(storage_class(global.space))
                    .operand(pointee),
            );
            let id = self.id();
            self.declare(
                Instruction::result(VARIABLE, pointer, id).word(storage_class(global.space)),
            );
            self.debug.push(
                Instruction::plain(NAME)
                    .operand(Operand::Id(id))
                    .operand(Operand::Literal(global.name.clone())),
            );
            if let Some(binding) = global.binding {
                self.annotations.push(
                    Instruction::plain(DECORATE)
                        .operand(Operand::Id(id))
                        .word(decoration::DESCRIPTOR_SET)
                        .word(binding.group),
                );
                self.annotations.push(
                    Instruction::plain(DECORATE)
                        .operand(Operand::Id(id))
                        .word(decoration::BINDING)
                        .word(binding.binding),
                );
                if !global.access.writable() {
                    self.annotations.push(
                        Instruction::plain(DECORATE)
                            .operand(Operand::Id(id))
                            .word(decoration::NON_WRITABLE),
                    );
                }
                if global.coherent && !self.vulkan_memory_model {
                    self.annotations.push(
                        Instruction::plain(DECORATE)
                            .operand(Operand::Id(id))
                            .word(decoration::COHERENT),
                    );
                }
            }
            self.globals.push(Global {
                id,
                pointer,
                wrapped: wrapper,
            });
        }
        self.assert_workgroup_left_undecorated(&decorated);
    }

    fn storage_pointee(
        &mut self,
        global: &neura_shader_ir::Global,
        decorated: &mut Decoration,
    ) -> (u32, bool) {
        match self.module.ty(global.ty).clone() {
            Type::Array {
                count: None,
                element,
            } => {
                let array = self.ty(global.ty);
                let wrapper = self.id();
                self.declare(Instruction::defined(TYPE_STRUCT, wrapper).operand(array));
                decorated.block(&mut self.annotations, wrapper);
                decorated.offset(&mut self.annotations, wrapper, 0, 0);
                decorated.stride(&mut self.annotations, array, self.module.stride(element));
                self.layout(element, decorated);
                (wrapper, true)
            }
            Type::Struct { members, .. } => {
                let pointee = self.ty(global.ty);
                decorated.block(&mut self.annotations, pointee);
                let mut pending = Vec::new();
                for (position, member) in members.iter().enumerate() {
                    decorated.offset(
                        &mut self.annotations,
                        pointee,
                        position as u32,
                        member.offset,
                    );
                    pending.push(member.ty);
                }
                for ty in pending {
                    self.layout(ty, decorated);
                }
                (pointee, false)
            }
            other => panic!(
                "the device storage global {} holds {}",
                global.name,
                neura_shader_ir::element_name(&other)
            ),
        }
    }

    fn layout(&mut self, ty: TypeId, decorated: &mut Decoration) {
        match self.module.ty(ty).clone() {
            Type::Array { element, .. } => {
                let array = self.types[&ty];
                decorated.stride(&mut self.annotations, array, self.module.stride(element));
                self.layout(element, decorated);
            }
            Type::Struct { members, .. } => {
                let id = self.types[&ty];
                for (position, member) in members.iter().enumerate() {
                    decorated.offset(&mut self.annotations, id, position as u32, member.offset);
                }
                for member in &members {
                    self.layout(member.ty, decorated);
                }
            }
            _ => {}
        }
    }

    fn assert_workgroup_left_undecorated(&self, decorated: &Decoration) {
        for index in 0..self.module.globals().len() as u32 {
            let global = self.module.global(index);
            if global.space == Space::Storage {
                continue;
            }
            let mut pending = vec![global.ty];
            while let Some(ty) = pending.pop() {
                if let Some(id) = self.types.get(&ty) {
                    assert!(
                        !decorated.types.contains(id),
                        "the workgroup memory of {} reuses a decorated storage type",
                        global.name
                    );
                }
                match self.module.ty(ty) {
                    Type::Array { element, .. } => pending.push(*element),
                    Type::Struct { members, .. } => {
                        pending.extend(members.iter().map(|member| member.ty));
                    }
                    _ => {}
                }
            }
        }
    }

    fn collect(&mut self) {
        let module = self.module;
        for index in 0..module.functions().len() as u32 {
            let id = self.id();
            self.function_ids.push(id);
            let function = &module.functions()[index as usize];
            let mut locals = Vec::new();
            for local in &function.locals {
                let base = self.ty(local.ty);
                let pointer = self.id();
                self.declare(
                    Instruction::defined(TYPE_POINTER, pointer)
                        .word(storage::FUNCTION)
                        .operand(base),
                );
                locals.push((pointer, self.id()));
            }
            self.locals[index as usize] = locals;
            self.collect_block(index, &function.body);
        }
    }

    fn collect_value(&mut self, function: u32, instruction: &DeviceInstruction) -> u32 {
        match instruction {
            DeviceInstruction::Argument { index, .. } => {
                let ty = self.module.functions()[function as usize].arguments[*index as usize].ty;
                self.ty(ty)
            }
            DeviceInstruction::Address {
                address: neura_shader_ir::Address::Local(index),
                ..
            } => self.locals[function as usize][*index as usize].0,
            DeviceInstruction::Address {
                address: neura_shader_ir::Address::Global(index),
                ..
            } => self.globals[*index as usize].pointer,
            _ => {
                let ty = self.module.value_ty(instruction.result().expect("a value"));
                self.ty(ty)
            }
        }
    }

    fn collect_block(&mut self, function: u32, body: &'m [DeviceInstruction]) {
        for instruction in body {
            if let Some(result) = instruction.result() {
                let ty = self.collect_value(function, instruction);
                let id = match instruction {
                    DeviceInstruction::Address {
                        address: neura_shader_ir::Address::Local(index),
                        ..
                    } => self.locals[function as usize][*index as usize].1,
                    DeviceInstruction::Address {
                        address: neura_shader_ir::Address::Global(index),
                        ..
                    } => self.globals[*index as usize].id,
                    _ => self.id(),
                };
                self.values.insert(result, id);
                self.value_types.insert(result, ty);
                self.declared.insert(result, instruction);
                if matches!(instruction, DeviceInstruction::Argument { .. }) {
                    self.arguments[function as usize].push(result);
                }
            }
            match instruction {
                DeviceInstruction::If { accept, reject, .. } => {
                    self.collect_block(function, accept);
                    self.collect_block(function, reject);
                }
                DeviceInstruction::Switch { cases, default, .. } => {
                    for (_, body) in cases {
                        self.collect_block(function, body);
                    }
                    self.collect_block(function, default);
                }
                DeviceInstruction::Loop { body, continuing } => {
                    self.collect_block(function, body);
                    self.collect_block(function, continuing);
                }
                DeviceInstruction::Block(body) => self.collect_block(function, body),
                _ => {}
            }
        }
    }

    fn intern_constants(&mut self) {
        let values = self
            .module
            .value_ids()
            .filter_map(|value| {
                self.module
                    .constant_of(value)
                    .map(|constant| (value, constant))
            })
            .collect::<Vec<_>>();
        for (value, constant) in values {
            let id = match constant {
                Constant::U32(number) => self.integer(Scalar::U32, number),
                Constant::I32(number) => self.integer(Scalar::I32, number as u32),
                Constant::F32(number) => self.float(number),
                Constant::Bool(flag) => self.boolean(flag),
                Constant::Zero(ty) => self.zero(ty),
            };
            self.values.insert(value, id);
        }
    }

    fn declare_builtins(&mut self) {
        let entry = self.module.entry_index() as usize;
        let arguments = self.module.entry().arguments.clone();
        for (position, argument) in arguments.iter().enumerate() {
            let builtin = argument.builtin.unwrap_or_else(|| {
                panic!(
                    "the device entry argument {} is not a builtin",
                    argument.name
                )
            });
            let code = builtin_code(builtin);
            if self.builtins.contains_key(&code) {
                continue;
            }
            let base = self.value_types[&self.arguments[entry][position]];
            let pointer = self.id();
            self.declare(
                Instruction::defined(TYPE_POINTER, pointer)
                    .word(storage::INPUT)
                    .operand(base),
            );
            let id = self.id();
            self.declare(Instruction::result(VARIABLE, pointer, id).word(storage::INPUT));
            self.debug.push(
                Instruction::plain(NAME)
                    .operand(Operand::Id(id))
                    .operand(Operand::Literal(argument.name.clone())),
            );
            self.annotations.push(
                Instruction::plain(DECORATE)
                    .operand(Operand::Id(id))
                    .word(decoration::BUILT_IN)
                    .word(code),
            );
            self.builtins.insert(code, id);
        }
    }

    pub(crate) fn builtin(&self, builtin: BuiltIn) -> u32 {
        self.builtins[&builtin_code(builtin)]
    }

    fn usage(&mut self) {
        for (index, function) in self.module.functions().iter().enumerate() {
            let mut used = HashSet::new();
            collect_globals(&function.body, &mut used);
            self.used[index] = used;
        }
        loop {
            let mut grown = false;
            for index in 0..self.module.functions().len() {
                let mut calls = Vec::new();
                gather_calls(&self.module.functions()[index].body, &mut calls);
                for callee in calls {
                    let callee = self.used[callee as usize].clone();
                    for global in callee {
                        grown |= self.used[index].insert(global);
                    }
                }
            }
            if !grown {
                break;
            }
        }
    }

    fn capabilities(&mut self) {
        self.header
            .push(Instruction::plain(CAPABILITY).word(capability::SHADER));
        if self.module.requirements().float16 {
            self.header
                .push(Instruction::plain(CAPABILITY).word(capability::FLOAT16));
        }
        if self.module.requirements().cooperative_matrix {
            self.header
                .push(Instruction::plain(CAPABILITY).word(capability::COOPERATIVE_MATRIX_KHR));
        }
        if self.vulkan_memory_model {
            self.header
                .push(Instruction::plain(CAPABILITY).word(capability::VULKAN_MEMORY_MODEL));
            self.header.push(
                Instruction::plain(CAPABILITY).word(capability::VULKAN_MEMORY_MODEL_DEVICE_SCOPE),
            );
        }
        if self.module.requirements().cooperative_matrix {
            self.header.push(
                Instruction::plain(EXTENSION)
                    .operand(Operand::Literal("SPV_KHR_cooperative_matrix".to_owned())),
            );
        }
        if self
            .declared
            .values()
            .any(|instruction| matches!(instruction, DeviceInstruction::Math { .. }))
        {
            let id = self.id();
            self.header.push(
                Instruction::defined(EXT_INST_IMPORT, id)
                    .operand(Operand::Literal("GLSL.std.450".to_owned())),
            );
            self.glsl = Some(id);
        }
        self.header.push(
            Instruction::plain(MEMORY_MODEL)
                .word(ADDRESSING_MODEL_LOGICAL)
                .word(if self.vulkan_memory_model {
                    MEMORY_MODEL_VULKAN
                } else {
                    MEMORY_MODEL_GLSL450
                }),
        );
    }

    pub(crate) fn glsl(&self) -> u32 {
        self.glsl
            .expect("the device program uses no GLSL instruction")
    }

    fn entry_point(&mut self) {
        let entry = self.module.entry_index();
        let function = self.function_ids[entry as usize];
        let mut operands = vec![
            Operand::Id(function),
            Operand::Literal(self.module.entry().name.clone()),
        ];
        for builtin in self.builtins.values() {
            operands.push(Operand::Id(*builtin));
        }
        if self.version >= VERSION_1_4 {
            let mut used = self.used[entry as usize]
                .iter()
                .copied()
                .collect::<Vec<_>>();
            used.sort_unstable();
            for index in used {
                operands.push(Operand::Id(self.globals[index as usize].id));
            }
        }
        self.header.push(
            Instruction::plain(ENTRY_POINT)
                .word(EXECUTION_MODEL_GL_COMPUTE)
                .operands(operands),
        );
        self.header.push(
            Instruction::plain(EXECUTION_MODE)
                .operand(Operand::Id(function))
                .word(EXECUTION_MODE_LOCAL_SIZE)
                .word(self.module.workgroup_size())
                .word(1u32)
                .word(1u32),
        );
        self.debug.push(
            Instruction::plain(NAME)
                .operand(Operand::Id(function))
                .operand(Operand::Literal(self.module.entry().name.clone())),
        );
    }

    pub(crate) fn push(&mut self, instruction: Instruction) {
        self.defined(&instruction);
        self.definitions.push(instruction);
    }

    pub(crate) fn declare(&mut self, instruction: Instruction) {
        self.defined(&instruction);
        self.declarations.push(instruction);
    }

    fn defined(&mut self, instruction: &Instruction) {
        if let Some(id) = instruction.defines() {
            assert!(
                self.defined.insert(id),
                "the device program defines the SPIR-V id {id} twice: {instruction:?}"
            );
        }
    }

    fn assemble(self) -> Vec<u32> {
        let mut words = vec![MAGIC, self.version, 13, self.next, 0];
        for instruction in self
            .header
            .iter()
            .chain(&self.debug)
            .chain(&self.annotations)
            .chain(&self.declarations)
            .chain(&self.definitions)
        {
            instruction.encode(&mut words);
        }
        words
    }
}

#[derive(Default)]
struct Decoration {
    blocks: HashSet<u32>,
    offsets: HashSet<(u32, u32)>,
    strides: HashSet<u32>,
    types: HashSet<u32>,
}

impl Decoration {
    fn block(&mut self, annotations: &mut Vec<Instruction>, id: u32) {
        if self.blocks.insert(id) {
            annotations.push(
                Instruction::plain(DECORATE)
                    .operand(Operand::Id(id))
                    .word(decoration::BLOCK),
            );
            self.types.insert(id);
        }
    }

    fn offset(&mut self, annotations: &mut Vec<Instruction>, id: u32, member: u32, offset: u32) {
        if self.offsets.insert((id, member)) {
            annotations.push(
                Instruction::plain(MEMBER_DECORATE)
                    .operand(Operand::Id(id))
                    .word(member)
                    .word(decoration::OFFSET)
                    .word(offset),
            );
            self.types.insert(id);
        }
    }

    fn stride(&mut self, annotations: &mut Vec<Instruction>, id: u32, stride: u32) {
        if self.strides.insert(id) {
            annotations.push(
                Instruction::plain(DECORATE)
                    .operand(Operand::Id(id))
                    .word(decoration::ARRAY_STRIDE)
                    .word(stride),
            );
            self.types.insert(id);
        }
    }
}

pub(crate) fn storage_class(space: Space) -> u32 {
    match space {
        Space::Storage => storage::STORAGE_BUFFER,
        Space::WorkGroup => storage::WORKGROUP,
        Space::Function => storage::FUNCTION,
    }
}

fn builtin_code(builtin: BuiltIn) -> u32 {
    match builtin {
        BuiltIn::LocalInvocationIndex => built_in::LOCAL_INVOCATION_INDEX,
        BuiltIn::WorkGroupId => built_in::WORKGROUP_ID,
        BuiltIn::GlobalInvocationId => built_in::GLOBAL_INVOCATION_ID,
    }
}

fn collect_globals(block: &[DeviceInstruction], used: &mut HashSet<u32>) {
    for instruction in block {
        if let DeviceInstruction::Address {
            address: neura_shader_ir::Address::Global(index),
            ..
        } = instruction
        {
            used.insert(*index);
        }
        walk(instruction, &mut |inner| collect_globals(inner, used));
    }
}

fn gather_calls(block: &[DeviceInstruction], calls: &mut Vec<u32>) {
    for instruction in block {
        if let DeviceInstruction::Call { function, .. } = instruction {
            calls.push(*function);
        }
        walk(instruction, &mut |inner| gather_calls(inner, calls));
    }
}

fn walk(instruction: &DeviceInstruction, nested: &mut impl FnMut(&[DeviceInstruction])) {
    match instruction {
        DeviceInstruction::If { accept, reject, .. } => {
            nested(accept);
            nested(reject);
        }
        DeviceInstruction::Switch { cases, default, .. } => {
            for (_, body) in cases {
                nested(body);
            }
            nested(default);
        }
        DeviceInstruction::Loop { body, continuing } => {
            nested(body);
            nested(continuing);
        }
        DeviceInstruction::Block(body) => nested(body),
        _ => {}
    }
}
