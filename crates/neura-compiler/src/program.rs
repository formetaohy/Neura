use neura_shader_ir::{Module, Target, element_name};
use std::fmt::Write as _;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub const METAL_SIZE_BUFFER_SLOT: u8 = 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Vulkan,
    Metal,
    Dx12,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingKind {
    ReadOnlyStorage,
    ReadWriteStorage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindingSpec {
    pub binding: u32,
    pub kind: BindingKind,
}

impl BindingSpec {
    pub const fn storage(binding: u32) -> Self {
        Self {
            binding,
            kind: BindingKind::ReadOnlyStorage,
        }
    }

    pub const fn writable_storage(binding: u32) -> Self {
        Self {
            binding,
            kind: BindingKind::ReadWriteStorage,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShaderBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
    pub kind: BindingKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShaderTranslation {
    Spirv(Vec<u32>),
    Hlsl {
        source: String,
        entry: String,
    },
    Msl {
        source: String,
        entry: String,
        size_bindings: Vec<u32>,
    },
}

pub fn reflect(module: &Module) -> Vec<ShaderBinding> {
    let mut bindings = module
        .globals()
        .iter()
        .filter_map(|global| {
            let binding = global.binding?;
            Some(ShaderBinding {
                group: binding.group,
                binding: binding.binding,
                name: global.name.clone(),
                kind: if global.access.writable() {
                    BindingKind::ReadWriteStorage
                } else {
                    BindingKind::ReadOnlyStorage
                },
            })
        })
        .collect::<Vec<_>>();
    bindings.sort_by_key(|binding| (binding.group, binding.binding));
    bindings
}

pub fn describe(bindings: &[ShaderBinding]) -> String {
    let mut out = String::new();
    for binding in bindings {
        writeln!(
            out,
            "group {} binding {} {} {:?}",
            binding.group, binding.binding, binding.name, binding.kind,
        )
        .expect("a string accepts binding descriptions");
    }
    out
}

struct Compiled {
    label: String,
    module: Module,
    entry: String,
    workgroup: u32,
    bindings: Vec<BindingSpec>,
    reflected: Vec<ShaderBinding>,
    spirv: Vec<u32>,
}

#[derive(Clone)]
pub struct ComputeProgram {
    compiled: Arc<Compiled>,
}

impl PartialEq for ComputeProgram {
    fn eq(&self, other: &Self) -> bool {
        self.compiled.spirv == other.compiled.spirv
            && self.compiled.bindings == other.compiled.bindings
    }
}

impl Eq for ComputeProgram {}

impl Hash for ComputeProgram {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.compiled.spirv.hash(state);
        self.compiled.bindings.hash(state);
    }
}

impl std::fmt::Debug for ComputeProgram {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct(&self.compiled.label)
            .field("entry", &self.compiled.entry)
            .field("bindings", &self.compiled.bindings)
            .field("instructions", &self.compiled.spirv.len())
            .finish()
    }
}

impl ComputeProgram {
    pub fn new(label: &str, module: Module, entry: &str, bindings: &[BindingSpec]) -> Self {
        assert!(
            !bindings.is_empty() && bindings.len() <= METAL_SIZE_BUFFER_SLOT as usize,
            "a compute program binds between one and 30 storage buffers"
        );
        for (index, binding) in bindings.iter().enumerate() {
            assert_eq!(
                binding.binding as usize, index,
                "storage bindings are dense"
            );
        }
        assert_eq!(
            module.entry().name,
            entry,
            "{label} carries the device entry {} instead of {entry}",
            module.entry().name
        );
        assert!(
            module.entry().result.is_none(),
            "{label} declares an entry that returns a value"
        );
        let workgroup = module.workgroup_size();
        let reflected = reflect(&module);
        assert_eq!(
            reflected.len(),
            bindings.len(),
            "{label} declares different storage resources from its bindings:\n{}",
            describe(&reflected)
        );
        for (declared, spec) in reflected.iter().zip(bindings) {
            assert_eq!(declared.group, 0, "compute resources use group zero");
            assert_eq!(
                declared.binding, spec.binding,
                "a storage binding is missing"
            );
            assert_eq!(
                declared.kind, spec.kind,
                "binding {} has the wrong access",
                spec.binding
            );
        }
        for global in module.globals() {
            if global.space == neura_shader_ir::Space::WorkGroup {
                assert!(
                    matches!(
                        module.ty(global.ty),
                        neura_shader_ir::Type::Array { count: Some(_), .. }
                    ),
                    "{label} declares the workgroup variable {} as a {}",
                    global.name,
                    element_name(module.ty(global.ty))
                );
            }
        }
        let spirv = neura_spirv::write(&module);
        Self {
            compiled: Arc::new(Compiled {
                label: label.to_owned(),
                module,
                entry: entry.to_owned(),
                workgroup,
                bindings: bindings.to_vec(),
                reflected,
                spirv,
            }),
        }
    }

    pub fn label(&self) -> &str {
        &self.compiled.label
    }

    pub fn entry(&self) -> &str {
        &self.compiled.entry
    }

    pub fn workgroup_size(&self) -> u32 {
        self.compiled.workgroup
    }

    pub fn bindings(&self) -> &[BindingSpec] {
        &self.compiled.bindings
    }

    pub fn reflected(&self) -> &[ShaderBinding] {
        &self.compiled.reflected
    }

    pub fn spirv(&self) -> &[u32] {
        &self.compiled.spirv
    }

    pub fn module(&self) -> &Module {
        &self.compiled.module
    }

    pub fn requirements(&self) -> neura_shader_ir::Requirements {
        self.compiled.module.requirements()
    }

    pub fn translate(&self, backend: Backend) -> ShaderTranslation {
        let compiled = &self.compiled;
        match backend {
            Backend::Vulkan => {
                Target::SPIRV.supports(compiled.module.requirements(), compiled.label.as_str());
                ShaderTranslation::Spirv(compiled.spirv.clone())
            }
            Backend::Dx12 => {
                let source = neura_hlsl::write(&compiled.module);
                ShaderTranslation::Hlsl {
                    entry: neura_hlsl::symbol(&compiled.entry),
                    source,
                }
            }
            Backend::Metal => {
                let source = neura_msl::write(&compiled.module);
                ShaderTranslation::Msl {
                    entry: neura_msl::symbol(&compiled.entry),
                    source,
                    size_bindings: Vec::new(),
                }
            }
        }
    }
}
