extern crate self as neura_compiler;

mod lower;
mod program;
mod resource;
pub use neura_abi as abi;
pub use neura_ir as ir;
mod uniform;
mod workgroup;

use naga::{
    AddressSpace, ArraySize, GlobalVariable, Handle, MemoryDecorations, Module, Scalar,
    StorageAccess, StructMember, Type, TypeInner, VectorSize,
};
use neura_abi::{FieldType, RecordLayout};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::num::NonZeroU32;

pub use neura_macro::{kernel, module};
pub use program::{
    Backend, BindingKind, BindingSpec, ComputeProgram, METAL_SIZE_BUFFER_SLOT, ShaderBinding,
    ShaderTranslation, describe, reflect,
};
pub use resource::{DynamicRead, Read, ReadWrite, Uvec3};

#[derive(Clone, Copy)]
struct Global {
    handle: Handle<GlobalVariable>,
    ty: Handle<Type>,
}

pub struct Compiler {
    module: Module,
    types: HashMap<String, Handle<Type>>,
    functions: BTreeMap<String, ir::Function>,
    lowered: HashMap<String, Handle<naga::Function>>,
    pending: HashSet<String>,
    globals: HashMap<String, Global>,
    constants: HashMap<String, u32>,
    bindings: Vec<BindingSpec>,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    pub fn new() -> Self {
        let mut compiler = Self {
            module: Module::default(),
            types: HashMap::new(),
            functions: BTreeMap::new(),
            lowered: HashMap::new(),
            pending: HashSet::new(),
            globals: HashMap::new(),
            constants: HashMap::new(),
            bindings: Vec::new(),
        };
        for (name, inner) in [
            ("u32", TypeInner::Scalar(Scalar::U32)),
            ("i32", TypeInner::Scalar(Scalar::I32)),
            ("f32", TypeInner::Scalar(Scalar::F32)),
            ("bool", TypeInner::Scalar(Scalar::BOOL)),
            (
                "uvec3",
                TypeInner::Vector {
                    size: VectorSize::Tri,
                    scalar: Scalar::U32,
                },
            ),
            (
                "uvec4",
                TypeInner::Vector {
                    size: VectorSize::Quad,
                    scalar: Scalar::U32,
                },
            ),
            (
                "fvec2",
                TypeInner::Vector {
                    size: VectorSize::Bi,
                    scalar: Scalar::F32,
                },
            ),
            ("AtomicU32", TypeInner::Atomic(Scalar::U32)),
        ] {
            compiler.types.insert(
                name.to_owned(),
                compiler
                    .module
                    .types
                    .insert(Type { name: None, inner }, naga::Span::UNDEFINED),
            );
        }
        compiler
    }

    fn ty(&self, name: &str) -> Handle<Type> {
        *self
            .types
            .get(name)
            .unwrap_or_else(|| panic!("the Rust device type {name} is not declared"))
    }

    pub fn record(&mut self, layout: RecordLayout) {
        assert!(
            !self.types.contains_key(layout.name),
            "a device type is declared twice"
        );
        let members = layout
            .fields
            .iter()
            .map(|field| StructMember {
                name: Some(field.name.to_owned()),
                ty: self.ty(match field.ty {
                    FieldType::U32 => "u32",
                    FieldType::I32 => "i32",
                    FieldType::F32 => "f32",
                    FieldType::U32x4 => "uvec4",
                }),
                binding: None,
                offset: field.offset,
            })
            .collect();
        let handle = self.module.types.insert(
            Type {
                name: Some(layout.name.to_owned()),
                inner: TypeInner::Struct {
                    members,
                    span: layout.size,
                },
            },
            naga::Span::UNDEFINED,
        );
        self.types.insert(layout.name.to_owned(), handle);
    }

    fn array(&mut self, element: Handle<Type>, length: Option<u32>) -> Handle<Type> {
        let stride = self.size(element);
        let size = length.map_or(ArraySize::Dynamic, |count| {
            ArraySize::Constant(NonZeroU32::new(count).expect("an array is not empty"))
        });
        self.module.types.insert(
            Type {
                name: None,
                inner: TypeInner::Array {
                    base: element,
                    size,
                    stride,
                },
            },
            naga::Span::UNDEFINED,
        )
    }

    fn size(&self, ty: Handle<Type>) -> u32 {
        match &self.module.types[ty].inner {
            TypeInner::Scalar(scalar) | TypeInner::Atomic(scalar) => u32::from(scalar.width),
            TypeInner::Vector { size, scalar } => *size as u32 * u32::from(scalar.width),
            TypeInner::Array { size, stride, .. } => match size {
                ArraySize::Constant(count) => count.get() * stride,
                ArraySize::Dynamic => panic!("a runtime array cannot be an array element"),
                ArraySize::Pending(_) => panic!("an unresolved array has no byte size"),
            },
            TypeInner::Struct { span, .. } => *span,
            other => panic!("{other:?} cannot be held by a storage buffer"),
        }
    }

    pub fn storage_array(&mut self, name: &str, element: &str, spec: BindingSpec) {
        let ty = self.ty(element);
        let array = self.array(ty, None);
        self.storage(name, array, spec);
    }

    pub fn storage_record(&mut self, name: &str, record: &str, spec: BindingSpec) {
        let ty = self.ty(record);
        self.storage(name, ty, spec);
    }

    fn storage(&mut self, name: &str, ty: Handle<Type>, spec: BindingSpec) {
        assert_eq!(
            spec.binding as usize,
            self.bindings.len(),
            "storage slots are dense"
        );
        let access = match spec.kind {
            BindingKind::ReadOnlyStorage => StorageAccess::LOAD,
            BindingKind::ReadWriteStorage => {
                StorageAccess::LOAD | StorageAccess::STORE | StorageAccess::ATOMIC
            }
        };
        let handle = self.module.global_variables.append(
            GlobalVariable {
                name: Some(name.to_owned()),
                space: AddressSpace::Storage { access },
                binding: Some(naga::ResourceBinding {
                    group: 0,
                    binding: spec.binding,
                }),
                ty,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        assert!(
            self.globals
                .insert(name.to_owned(), Global { handle, ty })
                .is_none()
        );
        self.bindings.push(spec);
    }

    pub fn workgroup(&mut self, name: &str, element: &str, count: u32) {
        let ty = self.ty(element);
        let array = self.array(ty, Some(count));
        let handle = self.module.global_variables.append(
            GlobalVariable {
                name: Some(name.to_owned()),
                space: AddressSpace::WorkGroup,
                binding: None,
                ty: array,
                init: None,
                memory_decorations: MemoryDecorations::empty(),
            },
            naga::Span::UNDEFINED,
        );
        assert!(
            self.globals
                .insert(name.to_owned(), Global { handle, ty: array })
                .is_none()
        );
    }

    pub fn workgroup_bytes(&self) -> u64 {
        self.globals
            .values()
            .filter(|global| {
                matches!(
                    self.module.global_variables[global.handle].space,
                    AddressSpace::WorkGroup
                )
            })
            .map(|global| u64::from(self.size(global.ty)))
            .sum()
    }

    pub fn constant(&mut self, name: &str, value: u32) {
        assert!(
            self.constants.insert(name.to_owned(), value).is_none(),
            "{name} is defined twice"
        );
    }

    pub fn function(&mut self, function: ir::Function) {
        let name = function.name.clone();
        assert!(
            self.functions.insert(name.clone(), function).is_none(),
            "the Rust device function {name} is declared twice"
        );
    }

    pub fn select(&mut self, source: &str, name: &str) {
        let mut function = self
            .functions
            .get(source)
            .unwrap_or_else(|| panic!("Rust device function {source} does not exist"))
            .clone();
        function.name = name.to_owned();
        self.function(function);
    }

    pub fn insert_case(&mut self, name: &str, arm: ir::Arm) {
        let function = self
            .functions
            .get_mut(name)
            .unwrap_or_else(|| panic!("device dispatcher {name} does not exist"));
        assert_eq!(
            ir::match_count(&function.body),
            1,
            "device dispatcher {name} has exactly one match"
        );
        let cases = ir::match_cases(&mut function.body).expect("a device dispatcher has one match");
        assert!(
            !cases.iter().any(|present| present.pattern == arm.pattern),
            "device dispatcher {name} has a duplicate case"
        );
        let fallback = cases
            .iter()
            .position(|present| matches!(present.pattern, ir::Pattern::Default))
            .unwrap_or(cases.len());
        cases.insert(fallback, arm);
    }

    pub fn retain_cases(&mut self, name: &str, cases: &[String]) {
        let function = self
            .functions
            .get_mut(name)
            .unwrap_or_else(|| panic!("Rust device dispatcher {name} does not exist"));
        let mut found = false;
        for statement in &mut function.body {
            if let ir::Statement::Match { arms, .. } = statement {
                assert!(!found, "a specialized dispatcher has one Rust match");
                found = true;
                arms.retain(|arm| match &arm.pattern {
                    ir::Pattern::Default => true,
                    ir::Pattern::Constant(path) => cases.contains(path),
                    _ => panic!("a device dispatcher requires qualified Rust constants"),
                });
            }
        }
        assert!(found, "{name} does not dispatch a Rust match");
    }

    pub fn specialize(&mut self, template: &str, name: &str, constants: &[(&str, u32)]) {
        let mut function = self
            .functions
            .get(template)
            .unwrap_or_else(|| panic!("the Rust device template {template} does not exist"))
            .clone();
        function.name = name.to_owned();
        let suffix = name
            .rsplit_once('_')
            .expect("a specialized name has a suffix")
            .1;
        let constants = constants
            .iter()
            .map(|(name, value)| (name.to_string(), *value))
            .collect::<HashMap<_, _>>();
        for statement in &mut function.body {
            statement.specialize(&constants, suffix);
        }
        self.function(function);
    }

    pub fn finish(mut self, label: &str, entry: &str, workgroup: u32) -> ComputeProgram {
        let function = self
            .functions
            .get(entry)
            .unwrap_or_else(|| panic!("{label} does not define the Rust device entry {entry}"))
            .clone();
        let lowered = lower::FunctionLower::new(&mut self, &function, true).lower(&function.body);
        self.module.entry_points.push(naga::EntryPoint {
            name: entry.to_owned(),
            stage: naga::ShaderStage::Compute,
            early_depth_test: None,
            workgroup_size: [workgroup, 1, 1],
            workgroup_size_overrides: None,
            function: lowered,
            mesh_info: None,
            task_payload: None,
            incoming_ray_payload: None,
        });
        ComputeProgram::new(label, self.module, entry, &self.bindings)
    }

    fn lower_function(&mut self, name: &str) -> Handle<naga::Function> {
        if let Some(handle) = self.lowered.get(name) {
            return *handle;
        }
        assert!(
            self.pending.insert(name.to_owned()),
            "recursive Rust device functions are not supported: {name}"
        );
        let function = self
            .functions
            .get(name)
            .unwrap_or_else(|| panic!("Rust device function {name} is not declared"))
            .clone();
        let lowered = lower::FunctionLower::new(self, &function, false).lower(&function.body);
        let handle = self.module.functions.append(lowered, naga::Span::UNDEFINED);
        self.lowered.insert(name.to_owned(), handle);
        self.pending.remove(name);
        handle
    }
}
