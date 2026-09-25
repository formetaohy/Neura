use crate::buffer::{BufferBinding, GpuBuffer};
use crate::capability::BufferUsages;
use crate::context::Device;
use crate::native::{NativeGroup, NativePipeline};
pub use neura_compiler::{
    BindingKind, BindingSpec, ComputeProgram, METAL_SIZE_BUFFER_SLOT, ShaderTranslation,
};
use std::sync::Arc;

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
