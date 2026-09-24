use crate::buffer::{BufferBinding, GpuBuffer};
use crate::capability::{Backend, BufferUsages};
use crate::context::Device;
use crate::native::{NativeGroup, NativePipeline};
use std::sync::Arc;

pub const METAL_SIZE_BUFFER_SLOT: u8 = 30;

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

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct ComputeProgram {
    label: String,
    source: Arc<str>,
    entry: String,
    bindings: Vec<BindingSpec>,
}

impl std::fmt::Debug for ComputeProgram {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        out.debug_struct(&self.label)
            .field("entry", &self.entry)
            .field("bindings", &self.bindings)
            .field("bytes", &self.source.len())
            .finish()
    }
}

impl ComputeProgram {
    pub fn new(
        label: &str,
        source: impl Into<Arc<str>>,
        entry: &str,
        bindings: &[BindingSpec],
    ) -> Self {
        assert!(
            !bindings.is_empty() && bindings.len() <= METAL_SIZE_BUFFER_SLOT as usize,
            "a compute program binds between one and 30 storage buffers"
        );
        assert!(!entry.is_empty(), "a compute program needs an entry point");
        for (index, binding) in bindings.iter().enumerate() {
            assert_eq!(
                binding.binding as usize, index,
                "storage bindings must be densely numbered"
            );
        }
        Self {
            label: label.to_owned(),
            source: source.into(),
            entry: entry.to_owned(),
            bindings: bindings.to_vec(),
        }
    }

    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn entry(&self) -> &str {
        &self.entry
    }

    pub fn bindings(&self) -> &[BindingSpec] {
        &self.bindings
    }

    pub fn translate(&self, backend: Backend) -> ShaderTranslation {
        match backend {
            Backend::Vulkan => ShaderTranslation::Spirv(crate::native::shader::spirv(self)),
            Backend::Dx12 => {
                let (source, entry) = crate::native::shader::hlsl(self);
                ShaderTranslation::Hlsl { source, entry }
            }
            Backend::Metal => {
                let (source, entry, size_bindings) = crate::native::shader::msl(self);
                ShaderTranslation::Msl {
                    source,
                    entry,
                    size_bindings,
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Binding<'a> {
    pub index: u32,
    pub buffer: BufferBinding<'a>,
}

#[derive(Clone)]
pub(crate) struct BoundBuffer {
    pub(crate) buffer: GpuBuffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    pub(crate) dynamic: bool,
}

pub(crate) struct Slot {
    pub(crate) native: NativePipeline,
    pub(crate) program: Arc<ComputeProgram>,
    pub(crate) device: Device,
}

#[derive(Clone)]
pub struct PipelineHandle {
    pub(crate) slot: Arc<Slot>,
}

#[derive(Clone)]
pub struct BindGroup {
    pub(crate) native: NativeGroup,
    pub(crate) buffers: Vec<BoundBuffer>,
    pub(crate) slot: Arc<Slot>,
}

impl PipelineHandle {
    pub(crate) fn new(device: &Device, program: Arc<ComputeProgram>) -> Self {
        assert!(
            program.bindings().len() as u32 <= device.limits().max_storage_buffers_per_shader_stage
        );
        let native = device.native().create_pipeline(&program);
        Self {
            slot: Arc::new(Slot {
                device: device.clone(),
                program,
                native,
            }),
        }
    }

    pub fn label(&self) -> &str {
        self.slot.program.label()
    }

    pub fn bind_group(&self, entries: &[Binding<'_>]) -> BindGroup {
        let specs = self.slot.program.bindings();
        assert_eq!(
            entries.len(),
            specs.len(),
            "a group must bind exactly the program's buffers"
        );
        let buffers = entries
            .iter()
            .zip(specs)
            .map(|(entry, spec)| {
                assert_eq!(
                    entry.index, spec.binding,
                    "a binding occupies the wrong slot"
                );
                let buffer = entry.buffer.buffer;
                assert!(
                    buffer.device().same(&self.slot.device),
                    "a bind group cannot reference a different device"
                );
                assert!(
                    buffer.usage().contains(BufferUsages::STORAGE),
                    "a storage binding requires shader storage"
                );
                assert!(
                    entry.buffer.offset.is_multiple_of(
                        self.slot
                            .device
                            .limits()
                            .min_storage_buffer_offset_alignment
                    ),
                    "a storage binding is misaligned"
                );
                assert!(
                    entry.buffer.size <= self.slot.device.limits().max_storage_buffer_binding_size,
                    "a storage binding exceeds the device's maximum range"
                );
                BoundBuffer {
                    buffer: buffer.clone(),
                    offset: entry.buffer.offset,
                    size: entry.buffer.size,
                    dynamic: spec.dynamic_offset,
                }
            })
            .collect::<Vec<_>>();
        let native = self
            .slot
            .device
            .native()
            .create_group(&self.slot.native, &buffers);
        BindGroup {
            slot: self.slot.clone(),
            buffers,
            native,
        }
    }

    pub fn is_compiled(&self) -> bool {
        self.slot.native.is_compiled()
    }

    pub fn compile(&self) {
        self.slot.native.compile(&self.slot.program);
    }
}
