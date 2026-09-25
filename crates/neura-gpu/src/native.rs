#[cfg(target_os = "windows")]
pub(crate) mod dx12;
#[cfg(target_os = "macos")]
pub(crate) mod metal;
#[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
pub(crate) mod vulkan;

use crate::capability::{AdapterInfo, Backends, BufferUsages, Limits};
use crate::context::{GpuRequest, GpuUnavailable};
use crate::pipeline::{BoundBuffer, ComputeProgram};
use crate::submission::{Command, Write};
use std::sync::Arc;
use std::time::Duration;

pub(crate) enum NativeDevice {
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    Vulkan(Arc<vulkan::Device>),
    #[cfg(target_os = "windows")]
    Dx12(Arc<dx12::Device>),
    #[cfg(target_os = "macos")]
    Metal(Arc<metal::Device>),
}

#[derive(Clone)]
pub(crate) enum NativeBuffer {
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    Vulkan(vulkan::Buffer),
    #[cfg(target_os = "windows")]
    Dx12(dx12::Buffer),
    #[cfg(target_os = "macos")]
    Metal(metal::Buffer),
}

pub(crate) enum NativePipeline {
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    Vulkan(vulkan::Pipeline),
    #[cfg(target_os = "windows")]
    Dx12(dx12::Pipeline),
    #[cfg(target_os = "macos")]
    Metal(metal::Pipeline),
}

#[derive(Clone)]
pub(crate) enum NativeGroup {
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    Vulkan(Arc<vulkan::Group>),
    #[cfg(target_os = "windows")]
    Dx12,
    #[cfg(target_os = "macos")]
    Metal,
}

pub(crate) fn open(
    request: &GpuRequest,
) -> Result<(NativeDevice, AdapterInfo, Limits), GpuUnavailable> {
    let mut reasons = Vec::new();
    let mut unsupported = None;
    #[cfg(target_os = "windows")]
    if request.backends.contains(Backends::DX12) {
        match dx12::Device::open(request.power_preference) {
            Ok((device, info, limits)) if limits.supports(&Limits::BASELINE) => {
                return Ok((NativeDevice::Dx12(device), info, limits));
            }
            Ok((_, info, limits)) => unsupported = Some((info, limits)),
            Err(error) => reasons.push(format!("D3D12: {error}")),
        }
    }
    #[cfg(target_os = "macos")]
    if request.backends.contains(Backends::METAL) {
        match metal::Device::open(request.power_preference) {
            Ok((device, info, limits)) if limits.supports(&Limits::BASELINE) => {
                return Ok((NativeDevice::Metal(device), info, limits));
            }
            Ok((_, info, limits)) => unsupported = Some((info, limits)),
            Err(error) => reasons.push(format!("Metal: {error}")),
        }
    }
    #[cfg(any(target_os = "windows", target_os = "linux", target_os = "macos"))]
    if request.backends.contains(Backends::VULKAN) {
        match vulkan::Device::open(request.power_preference) {
            Ok((device, info, limits)) if limits.supports(&Limits::BASELINE) => {
                return Ok((NativeDevice::Vulkan(device), info, limits));
            }
            Ok((_, info, limits)) => unsupported = Some((info, limits)),
            Err(error) => reasons.push(format!("Vulkan: {error}")),
        }
    }
    if let Some((info, limits)) = unsupported {
        Err(GpuUnavailable::UnsupportedLimits { info, limits })
    } else {
        Err(GpuUnavailable::NoAdapter {
            reason: if reasons.is_empty() {
                "no requested compute backend is available on this platform".to_owned()
            } else {
                reasons.join("; ")
            },
        })
    }
}

impl NativePipeline {
    pub(crate) fn is_compiled(&self) -> bool {
        match self {
            Self::Vulkan(pipeline) => pipeline.is_compiled(),
            #[cfg(target_os = "windows")]
            Self::Dx12(pipeline) => pipeline.is_compiled(),
            #[cfg(target_os = "macos")]
            Self::Metal(pipeline) => pipeline.is_compiled(),
        }
    }

    pub(crate) fn compile(&self, program: &ComputeProgram) {
        match self {
            Self::Vulkan(pipeline) => pipeline.compile(program),
            #[cfg(target_os = "windows")]
            Self::Dx12(pipeline) => pipeline.compile(program),
            #[cfg(target_os = "macos")]
            Self::Metal(pipeline) => pipeline.compile(program),
        }
    }
}

impl NativeDevice {
    pub(crate) fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: BufferUsages,
    ) -> NativeBuffer {
        match self {
            Self::Vulkan(device) => NativeBuffer::Vulkan(device.create_buffer(label, size, usage)),
            #[cfg(target_os = "windows")]
            Self::Dx12(device) => NativeBuffer::Dx12(device.create_buffer(label, size, usage)),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => NativeBuffer::Metal(device.create_buffer(label, size, usage)),
        }
    }

    pub(crate) fn create_pipeline(&self, program: &ComputeProgram) -> NativePipeline {
        match self {
            Self::Vulkan(device) => NativePipeline::Vulkan(device.create_pipeline(program)),
            #[cfg(target_os = "windows")]
            Self::Dx12(device) => NativePipeline::Dx12(device.create_pipeline(program)),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => NativePipeline::Metal(device.create_pipeline(program)),
        }
    }

    pub(crate) fn create_group(
        &self,
        pipeline: &NativePipeline,
        buffers: &[BoundBuffer],
    ) -> NativeGroup {
        match (self, pipeline) {
            (Self::Vulkan(device), NativePipeline::Vulkan(pipeline)) => {
                NativeGroup::Vulkan(Arc::new(device.create_group(pipeline, buffers)))
            }
            #[cfg(target_os = "windows")]
            (Self::Dx12(_), NativePipeline::Dx12(_)) => NativeGroup::Dx12,
            #[cfg(target_os = "macos")]
            (Self::Metal(_), NativePipeline::Metal(_)) => NativeGroup::Metal,
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            _ => panic!("a pipeline belongs to another backend"),
        }
    }

    pub(crate) fn submit(&self, writes: &[Write], commands: &[Command]) -> u64 {
        match self {
            Self::Vulkan(device) => device.submit(writes, commands),
            #[cfg(target_os = "windows")]
            Self::Dx12(device) => device.submit(writes, commands),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.submit(writes, commands),
        }
    }

    pub(crate) fn wait(&self, index: u64, timeout: Duration) {
        match self {
            Self::Vulkan(device) => device.wait(index, timeout),
            #[cfg(target_os = "windows")]
            Self::Dx12(device) => device.wait(index, timeout),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.wait(index, timeout),
        }
    }

    pub(crate) fn read(&self, buffer: &NativeBuffer, bytes: u64) -> Vec<u8> {
        match (self, buffer) {
            (Self::Vulkan(device), NativeBuffer::Vulkan(buffer)) => device.read(buffer, bytes),
            #[cfg(target_os = "windows")]
            (Self::Dx12(device), NativeBuffer::Dx12(buffer)) => device.read(buffer, bytes),
            #[cfg(target_os = "macos")]
            (Self::Metal(device), NativeBuffer::Metal(buffer)) => device.read(buffer, bytes),
            #[cfg(any(target_os = "windows", target_os = "macos"))]
            _ => panic!("a buffer belongs to another backend"),
        }
    }

    pub(crate) fn assert_alive(&self) {
        match self {
            Self::Vulkan(device) => device.assert_alive(),
            #[cfg(target_os = "windows")]
            Self::Dx12(device) => device.assert_alive(),
            #[cfg(target_os = "macos")]
            Self::Metal(device) => device.assert_alive(),
        }
    }
}
