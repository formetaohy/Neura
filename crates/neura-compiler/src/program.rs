use crate::workgroup;
use naga::back::{hlsl, msl, spv};
use naga::valid::{Capabilities, ModuleInfo, ValidationFlags, Validator};
use naga::{AddressSpace, Module, ShaderStage, StorageAccess};
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
    pub dynamic_offset: bool,
}

impl BindingSpec {
    pub const fn storage(binding: u32) -> Self {
        Self {
            binding,
            kind: BindingKind::ReadOnlyStorage,
            dynamic_offset: false,
        }
    }

    pub const fn writable_storage(binding: u32) -> Self {
        Self {
            binding,
            kind: BindingKind::ReadWriteStorage,
            dynamic_offset: false,
        }
    }

    pub const fn dynamic_storage(binding: u32) -> Self {
        Self {
            binding,
            kind: BindingKind::ReadOnlyStorage,
            dynamic_offset: true,
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

struct Compiled {
    label: String,
    module: Module,
    info: ModuleInfo,
    entry: String,
    index: usize,
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

pub fn reflect(module: &Module) -> Vec<ShaderBinding> {
    let mut bindings = module
        .global_variables
        .iter()
        .filter_map(|(_, variable)| {
            variable.binding.map(|binding| {
                let AddressSpace::Storage { access } = variable.space else {
                    panic!("device binding {} is not storage", binding.binding);
                };
                ShaderBinding {
                    group: binding.group,
                    binding: binding.binding,
                    name: variable
                        .name
                        .clone()
                        .expect("every device binding has a name"),
                    kind: if access.contains(StorageAccess::STORE) {
                        BindingKind::ReadWriteStorage
                    } else {
                        BindingKind::ReadOnlyStorage
                    },
                }
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
        let index = module
            .entry_points
            .iter()
            .position(|point| point.name == entry && point.stage == ShaderStage::Compute)
            .unwrap_or_else(|| panic!("{label} has no compute entry named {entry}"));
        let workgroup = module.entry_points[index].workgroup_size[0];
        assert!(workgroup > 0, "a compute workgroup is not empty");
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
        let info = Validator::new(
            ValidationFlags::all(),
            Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        )
        .validate(&module)
        .unwrap_or_else(|error| panic!("{label} contains an invalid device program: {error:#?}"));
        crate::uniform::verify(&module, index);
        let mut options = spv::Options {
            fake_missing_bindings: false,
            ..Default::default()
        };
        for spec in bindings {
            options.binding_map.insert(
                naga::ResourceBinding {
                    group: 0,
                    binding: spec.binding,
                },
                spv::BindingInfo {
                    descriptor_set: 0,
                    binding: spec.binding,
                    binding_array_size: None,
                },
            );
        }
        let spirv = workgroup::isolate(
            spv::write_vec(
                &module,
                &info,
                &options,
                Some(&spv::PipelineOptions {
                    shader_stage: ShaderStage::Compute,
                    entry_point: entry.to_owned(),
                }),
            )
            .unwrap_or_else(|error| panic!("{label} failed SPIR-V compilation: {error}")),
        );
        Self {
            compiled: Arc::new(Compiled {
                label: label.to_owned(),
                module,
                info,
                entry: entry.to_owned(),
                index,
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

    pub fn translate(&self, backend: Backend) -> ShaderTranslation {
        let compiled = &self.compiled;
        match backend {
            Backend::Vulkan => ShaderTranslation::Spirv(compiled.spirv.clone()),
            Backend::Dx12 => {
                let mut options = hlsl::Options {
                    shader_model: hlsl::ShaderModel::V6_0,
                    fake_missing_bindings: false,
                    ..Default::default()
                };
                for spec in &compiled.bindings {
                    options.binding_map.insert(
                        naga::ResourceBinding {
                            group: 0,
                            binding: spec.binding,
                        },
                        hlsl::BindTarget {
                            register: spec.binding,
                            ..Default::default()
                        },
                    );
                }
                let pipeline = hlsl::PipelineOptions {
                    entry_point: Some((ShaderStage::Compute, compiled.entry.clone())),
                };
                let mut source = String::new();
                let reflection = hlsl::Writer::new(&mut source, &options, &pipeline)
                    .write(&compiled.module, &compiled.info, None)
                    .unwrap_or_else(|error| {
                        panic!("{} failed HLSL compilation: {error}", compiled.label)
                    });
                let entry = reflection.entry_point_names[compiled.index]
                    .as_ref()
                    .unwrap_or_else(|error| panic!("{} has no HLSL entry: {error}", compiled.label))
                    .clone();
                ShaderTranslation::Hlsl { source, entry }
            }
            Backend::Metal => {
                let mut resources = msl::EntryPointResources {
                    sizes_buffer: Some(METAL_SIZE_BUFFER_SLOT),
                    ..Default::default()
                };
                let size_bindings = compiled
                    .module
                    .global_variables
                    .iter()
                    .filter(|(_, variable)| {
                        compiled.module.types[variable.ty]
                            .inner
                            .needs_host_buffer_byte_size(&compiled.module.types)
                    })
                    .map(|(_, variable)| {
                        let binding = variable.binding.expect("a runtime-sized buffer is bound");
                        assert_eq!(binding.group, 0);
                        binding.binding
                    })
                    .collect::<Vec<_>>();
                for spec in &compiled.bindings {
                    resources.resources.insert(
                        naga::ResourceBinding {
                            group: 0,
                            binding: spec.binding,
                        },
                        msl::BindTarget {
                            buffer: Some(spec.binding as u8),
                            mutable: spec.kind == BindingKind::ReadWriteStorage,
                            ..Default::default()
                        },
                    );
                }
                let mut options = msl::Options {
                    lang_version: (2, 3),
                    fake_missing_bindings: false,
                    ..Default::default()
                };
                options
                    .per_entry_point_map
                    .insert(compiled.entry.clone(), resources);
                let (source, reflection) = msl::write_string(
                    &compiled.module,
                    &compiled.info,
                    &options,
                    &msl::PipelineOptions {
                        entry_point: Some((ShaderStage::Compute, compiled.entry.clone())),
                        ..Default::default()
                    },
                )
                .unwrap_or_else(|error| {
                    panic!("{} failed MSL compilation: {error}", compiled.label)
                });
                let entry = reflection.entry_point_names[compiled.index]
                    .as_ref()
                    .unwrap_or_else(|error| panic!("{} has no MSL entry: {error}", compiled.label))
                    .clone();
                ShaderTranslation::Msl {
                    source,
                    entry,
                    size_bindings,
                }
            }
        }
    }
}
