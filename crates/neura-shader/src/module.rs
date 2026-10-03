use crate::instruction::{Block, BuiltIn, Constant};
use crate::ty::MatrixUse;
use crate::ty::{Member, Scalar, Type, TypeId, ValueId};
use crate::validate::Verifier;
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Space {
    Storage,
    WorkGroup,
    Function,
}

impl Space {
    pub const fn name(self) -> &'static str {
        match self {
            Self::Storage => "storage",
            Self::WorkGroup => "workgroup",
            Self::Function => "function",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Access {
    Read,
    ReadWrite,
}

impl Access {
    pub const fn writable(self) -> bool {
        matches!(self, Self::ReadWrite)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Binding {
    pub group: u32,
    pub binding: u32,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Global {
    pub name: String,
    pub ty: TypeId,
    pub space: Space,
    pub binding: Option<Binding>,
    pub access: Access,
    pub coherent: bool,
}

#[derive(Clone, PartialEq, Hash, Debug)]
pub struct Argument {
    pub name: String,
    pub ty: TypeId,
    pub builtin: Option<BuiltIn>,
}

#[derive(Clone, PartialEq, Hash, Debug)]
pub struct Local {
    pub name: String,
    pub ty: TypeId,
}

#[derive(Clone, PartialEq, Hash, Debug)]
pub struct Function {
    pub name: String,
    pub arguments: Vec<Argument>,
    pub result: Option<TypeId>,
    pub locals: Vec<Local>,
    pub body: Block,
}

impl Function {
    pub fn returns_value(&self) -> bool {
        self.result.is_some()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Target {
    pub float16: bool,
    pub cooperative_matrix: bool,
}

impl Target {
    pub const SPIRV: Self = Self {
        float16: true,
        cooperative_matrix: true,
    };

    pub const HLSL: Self = Self {
        float16: false,
        cooperative_matrix: false,
    };

    pub const MSL: Self = Self {
        float16: true,
        cooperative_matrix: true,
    };

    pub fn supports(&self, requirements: Requirements, label: &str) {
        assert!(
            !requirements.float16 || self.float16,
            "{label} carries 16-bit floating point numbers this backend cannot express"
        );
        assert!(
            !requirements.cooperative_matrix || self.cooperative_matrix,
            "{label} carries cooperative matrices this backend cannot express"
        );
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Requirements {
    pub float16: bool,
    pub cooperative_matrix: bool,
}

pub struct Module {
    types: Vec<Type>,
    values: Vec<TypeId>,
    constants: Vec<Option<Constant>>,
    globals: Vec<Global>,
    functions: Vec<Function>,
    entry: u32,
    workgroup: u32,
    interned: HashMap<Type, TypeId>,
    literals: HashMap<Constant, ValueId>,
}

impl Module {
    pub fn new(workgroup: u32) -> Self {
        assert!(workgroup > 0, "a device program runs no thread");
        Self {
            types: Vec::new(),
            values: Vec::new(),
            constants: Vec::new(),
            globals: Vec::new(),
            functions: Vec::new(),
            entry: 0,
            workgroup,
            interned: HashMap::new(),
            literals: HashMap::new(),
        }
    }

    pub fn set_workgroup(&mut self, size: u32) {
        assert!(size > 0, "a device program runs no thread");
        self.workgroup = size;
    }

    pub fn set_entry(&mut self, function: u32) {
        assert!(
            (function as usize) < self.functions.len(),
            "the device entry is not a declared function"
        );
        self.entry = function;
    }

    pub fn declare(&mut self, function: Function) -> u32 {
        let index = self.functions.len() as u32;
        self.functions.push(function);
        index
    }

    pub fn function(&mut self, index: u32) -> &mut Function {
        &mut self.functions[index as usize]
    }

    pub fn entry(&self) -> &Function {
        &self.functions[self.entry as usize]
    }

    pub fn entry_index(&self) -> u32 {
        self.entry
    }

    pub fn functions(&self) -> &[Function] {
        &self.functions
    }

    pub fn workgroup_size(&self) -> u32 {
        self.workgroup
    }

    pub fn add_global(&mut self, global: Global) -> u32 {
        if let Some(binding) = global.binding {
            assert_eq!(binding.group, 0, "device bindings live in group zero");
            assert!(
                !self
                    .globals
                    .iter()
                    .any(|present| present.name == global.name),
                "the device global {} is declared twice",
                global.name
            );
        }
        let index = self.globals.len() as u32;
        self.globals.push(global);
        index
    }

    pub fn globals(&self) -> &[Global] {
        &self.globals
    }

    pub fn global(&self, index: u32) -> &Global {
        &self.globals[index as usize]
    }

    pub fn define(&mut self, ty: TypeId) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(ty);
        self.constants.push(None);
        id
    }

    pub fn constant(&mut self, constant: Constant) -> ValueId {
        if let Some(value) = self.literals.get(&constant) {
            return *value;
        }
        let ty = match constant {
            Constant::U32(_) => self.scalar(Scalar::U32),
            Constant::I32(_) => self.scalar(Scalar::I32),
            Constant::F32(_) => self.scalar(Scalar::F32),
            Constant::Bool(_) => self.scalar(Scalar::Bool),
            Constant::Zero(ty) => ty,
        };
        let value = self.define(ty);
        self.constants[value.0 as usize] = Some(constant);
        self.literals.insert(constant, value);
        value
    }

    pub fn constant_of(&self, value: ValueId) -> Option<Constant> {
        self.constants[value.0 as usize]
    }

    pub fn ty(&self, id: TypeId) -> &Type {
        &self.types[id.0 as usize]
    }

    pub fn types(&self) -> &[Type] {
        &self.types
    }

    pub fn type_ids(&self) -> impl ExactSizeIterator<Item = TypeId> {
        (0..self.types.len() as u32).map(TypeId)
    }

    pub fn value_ids(&self) -> impl ExactSizeIterator<Item = ValueId> {
        (0..self.values.len() as u32).map(ValueId)
    }

    pub fn value_ty(&self, value: ValueId) -> TypeId {
        *self
            .values
            .get(value.0 as usize)
            .unwrap_or_else(|| panic!("device value {} is not defined", value.0))
    }

    pub fn values(&self) -> &[TypeId] {
        &self.values
    }

    pub fn scalar(&mut self, scalar: Scalar) -> TypeId {
        if let Some(id) = self.scalar_type(scalar) {
            return id;
        }
        self.intern(Type::Scalar(scalar))
    }

    pub fn scalar_type(&self, scalar: Scalar) -> Option<TypeId> {
        self.interned.get(&Type::Scalar(scalar)).copied()
    }

    pub fn scalar_id(&self, scalar: Scalar) -> TypeId {
        self.scalar_type(scalar)
            .unwrap_or_else(|| panic!("the device module declares no {}", scalar.name()))
    }

    pub fn vector(&mut self, scalar: Scalar, length: u32) -> TypeId {
        assert!(
            (2..=4).contains(&length),
            "a device vector carries between two and four lanes"
        );
        self.intern(Type::Vector { scalar, length })
    }

    pub fn array(&mut self, element: TypeId, count: Option<u32>) -> TypeId {
        if let Some(count) = count {
            assert!(count > 0, "a device array holds no element");
        }
        self.intern(Type::Array { element, count })
    }

    pub fn structure(&mut self, name: &str, members: Vec<Member>, span: u32) -> TypeId {
        self.intern(Type::Struct {
            name: name.to_owned(),
            members,
            span,
        })
    }

    pub fn atomic(&mut self, scalar: Scalar) -> TypeId {
        self.intern(Type::Atomic(scalar))
    }

    pub fn pointer(&mut self, space: Space, base: TypeId) -> TypeId {
        self.intern(Type::Pointer { space, base })
    }

    pub fn cooperative_matrix(
        &mut self,
        scalar: Scalar,
        rows: u32,
        columns: u32,
        usage: MatrixUse,
    ) -> TypeId {
        self.intern(Type::CooperativeMatrix {
            scalar,
            rows,
            columns,
            usage,
        })
    }

    fn intern(&mut self, ty: Type) -> TypeId {
        if let Some(id) = self.interned.get(&ty) {
            return *id;
        }
        let id = TypeId(self.types.len() as u32);
        self.types.push(ty.clone());
        self.interned.insert(ty, id);
        id
    }

    pub fn size(&self, ty: TypeId) -> u32 {
        match self.ty(ty) {
            Type::Scalar(scalar) => scalar.bytes(),
            Type::Vector { scalar, length } => scalar.bytes() * length,
            Type::Array { element, count } => {
                let count = count.expect("a runtime sized array has no byte size");
                self.stride(*element) * count
            }
            Type::Struct { span, .. } => *span,
            Type::Atomic(scalar) => scalar.bytes(),
            Type::Pointer { .. } => Scalar::U32.bytes(),
            Type::CooperativeMatrix { .. } => {
                panic!("a cooperative matrix has no storage size")
            }
        }
    }

    pub fn stride(&self, element: TypeId) -> u32 {
        let size = self.size(element);
        let alignment = self.alignment(element);
        size.div_ceil(alignment) * alignment
    }

    pub fn alignment(&self, ty: TypeId) -> u32 {
        match self.ty(ty) {
            Type::Scalar(scalar) => scalar.bytes(),
            Type::Vector { scalar, length } => scalar.bytes() * length,
            Type::Array { element, .. } => self.alignment(*element),
            Type::Struct { members, .. } => members
                .iter()
                .map(|member| self.alignment(member.ty))
                .max()
                .unwrap_or(1),
            Type::Atomic(scalar) => scalar.bytes(),
            Type::Pointer { .. } => Scalar::U32.bytes(),
            Type::CooperativeMatrix { .. } => 1,
        }
    }

    pub fn element(&self, ty: TypeId) -> TypeId {
        match self.ty(ty) {
            Type::Array { element, .. } => *element,
            other => panic!("{} is not a device array", element_name(other)),
        }
    }

    pub fn pointee(&self, ty: TypeId) -> (Space, TypeId) {
        self.ty(ty)
            .pointer()
            .unwrap_or_else(|| panic!("{} is not a device pointer", element_name(self.ty(ty))))
    }

    pub fn loaded_ty(&self, pointer_ty: TypeId) -> TypeId {
        let (_, base) = self.pointee(pointer_ty);
        match self.ty(base) {
            Type::Atomic(scalar) => self.scalar_type(*scalar).expect("an atomic scalar exists"),
            _ => base,
        }
    }

    pub fn requirements(&self) -> Requirements {
        let mut requirements = Requirements::default();
        for ty in &self.types {
            match ty {
                Type::Scalar(Scalar::F16) => requirements.float16 = true,
                Type::Vector {
                    scalar: Scalar::F16,
                    ..
                } => requirements.float16 = true,
                Type::CooperativeMatrix { .. } => requirements.cooperative_matrix = true,
                _ => {}
            }
        }
        requirements
    }

    pub fn verify(&self) {
        Verifier::new(self).run();
    }

    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.workgroup.hash(&mut hasher);
        for ty in &self.types {
            ty.hash(&mut hasher);
        }
        for value in &self.values {
            value.hash(&mut hasher);
        }
        for constant in &self.constants {
            constant.hash(&mut hasher);
        }
        for global in &self.globals {
            global.hash(&mut hasher);
        }
        for function in &self.functions {
            function.hash(&mut hasher);
        }
        self.entry.hash(&mut hasher);
        hasher.finish()
    }

    pub fn bindings(&self) -> Vec<Binding> {
        self.globals
            .iter()
            .filter_map(|global| global.binding)
            .collect()
    }
}

pub fn element_name(ty: &Type) -> String {
    match ty {
        Type::Scalar(scalar) => scalar.name().to_owned(),
        Type::Vector { scalar, length } => format!("{}x{length}", scalar.name()),
        Type::Array { .. } => "an array".to_owned(),
        Type::Struct { name, .. } => format!("struct {name}"),
        Type::Atomic(scalar) => format!("atomic {}", scalar.name()),
        Type::Pointer { space, .. } => format!("a {} pointer", space.name()),
        Type::CooperativeMatrix {
            scalar,
            rows,
            columns,
            usage,
        } => format!("a {rows}x{columns} {} {}", scalar.name(), usage.name()),
    }
}
