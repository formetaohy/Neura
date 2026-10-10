use crate::buffer::{BufferBinding, GpuBuffer};
use crate::capability::BufferUsages;
use crate::context::Device;
use crate::native::{NativeGroup, NativePipeline};
use neura_shader::ComputeProgram;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Clone, Copy)]
pub struct Binding<'a> {
    pub index: u32,
    pub buffer: BufferBinding<'a>,
}

#[derive(Clone)]
pub(crate) struct BoundBuffer {
    pub(crate) buffer: GpuBuffer,
    pub(crate) offset: u64,
    #[cfg(vulkan_backend)]
    pub(crate) size: u64,
}

pub(crate) struct Slot {
    pub(crate) native: NativePipeline,
    pub(crate) program: Arc<ComputeProgram>,
    pub(crate) device: Device,
    pub(crate) dispatched: AtomicBool,
}

#[derive(Clone)]
pub struct PipelineHandle {
    pub(crate) slot: Arc<Slot>,
}

#[derive(Clone)]
pub struct BindGroup {
    pub(crate) native: NativeGroup,
    #[cfg(any(dx12_backend, metal_backend))]
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
                dispatched: AtomicBool::new(false),
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
                    #[cfg(vulkan_backend)]
                    size: entry.buffer.size,
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
            #[cfg(any(dx12_backend, metal_backend))]
            buffers,
            native,
        }
    }

    pub fn is_compiled(&self) -> bool {
        self.slot.native.is_compiled()
    }

    pub fn compile(&self) {
        self.slot
            .device
            .native()
            .compile(&self.slot.native, &self.slot.program);
    }

    pub(crate) fn first_dispatch(&self) -> bool {
        !self.slot.dispatched.swap(true, Ordering::Relaxed)
    }
}
