use std::sync::{Arc, OnceLock};
use wgpu::{
    BindGroup, BindGroupEntry, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry,
    BindingType, BufferBindingType, ComputePipeline as WgpuComputePipeline,
    ComputePipelineDescriptor, Device, PipelineLayout, PipelineLayoutDescriptor,
    ShaderModuleDescriptor, ShaderSource, ShaderStages,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BindingKind {
    ReadOnlyStorage,
    ReadWriteStorage,
}

impl BindingKind {
    fn entry(self, binding: u32, dynamic_offset: bool) -> BindGroupLayoutEntry {
        BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::COMPUTE,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage {
                    read_only: self == Self::ReadOnlyStorage,
                },
                has_dynamic_offset: dynamic_offset,
                min_binding_size: None,
            },
            count: None,
        }
    }
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
        assert!(!bindings.is_empty(), "a program binds something");
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
}

struct Slot {
    device: Device,
    program: Arc<ComputeProgram>,
    layout: PipelineLayout,
    group: BindGroupLayout,
    compiled: OnceLock<WgpuComputePipeline>,
}

#[derive(Clone)]
pub struct PipelineHandle {
    slot: Arc<Slot>,
}

impl PipelineHandle {
    pub(crate) fn new(device: &Device, program: Arc<ComputeProgram>) -> Self {
        let entries = program
            .bindings()
            .iter()
            .map(|spec| spec.kind.entry(spec.binding, spec.dynamic_offset))
            .collect::<Vec<_>>();
        let group = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(program.label()),
            entries: &entries,
        });
        let layouts = [Some(&group)];
        let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(program.label()),
            bind_group_layouts: &layouts,
            immediate_size: 0,
        });
        Self {
            slot: Arc::new(Slot {
                device: device.clone(),
                program,
                layout,
                group,
                compiled: OnceLock::new(),
            }),
        }
    }

    pub fn label(&self) -> &str {
        self.slot.program.label()
    }

    pub fn is_compiled(&self) -> bool {
        self.slot.compiled.get().is_some()
    }

    pub fn bind_group(&self, entries: &[BindGroupEntry<'_>]) -> BindGroup {
        self.slot
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(self.slot.program.label()),
                layout: &self.slot.group,
                entries,
            })
    }

    pub fn pipeline(&self) -> &WgpuComputePipeline {
        let slot = &self.slot;
        slot.compiled.get_or_init(|| {
            let module = slot.device.create_shader_module(ShaderModuleDescriptor {
                label: Some(slot.program.label()),
                source: ShaderSource::Wgsl(slot.program.source().into()),
            });
            slot.device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some(slot.program.label()),
                    layout: Some(&slot.layout),
                    module: &module,
                    entry_point: Some(slot.program.entry()),
                    compilation_options: Default::default(),
                    cache: None,
                })
        })
    }

    pub fn compile(&self) {
        self.pipeline();
    }
}
