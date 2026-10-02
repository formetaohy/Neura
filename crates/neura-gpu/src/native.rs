#[cfg(dx12_backend)]
pub(crate) mod dx12;
#[cfg(metal_backend)]
pub(crate) mod metal;
#[cfg(vulkan_backend)]
pub(crate) mod vulkan;

use crate::cache::PipelineCache;
use crate::capability::{AdapterInfo, AdapterPolicy, Backends, BufferUsages, Limits};
use crate::context::{GpuRequest, GpuUnavailable};
use crate::pipeline::{BoundBuffer, ComputeProgram};
use crate::submission::{Command, Write};
use std::sync::Arc;
use std::time::Duration;

pub(crate) const FRAMES_IN_FLIGHT: usize = 4;
pub(crate) const FRAME_TIMEOUT: Duration = Duration::from_secs(30);
pub(crate) const STAGING_BYTES: u64 = 256 << 10;

pub(crate) enum DeviceFailure {
    Missing { offered: Vec<AdapterInfo> },
    Unavailable { reason: String },
}

impl DeviceFailure {
    pub(crate) fn missing(offered: Vec<AdapterInfo>) -> Self {
        Self::Missing { offered }
    }

    pub(crate) fn reason(reason: impl Into<String>) -> Self {
        Self::Unavailable {
            reason: reason.into(),
        }
    }
}

impl From<String> for DeviceFailure {
    fn from(reason: String) -> Self {
        Self::reason(reason)
    }
}

pub(crate) enum NativeDevice {
    #[cfg(vulkan_backend)]
    Vulkan(Arc<vulkan::Device>),
    #[cfg(dx12_backend)]
    Dx12(Arc<dx12::Device>),
    #[cfg(metal_backend)]
    Metal(Arc<metal::Device>),
}

#[derive(Clone)]
pub(crate) enum NativeBuffer {
    #[cfg(vulkan_backend)]
    Vulkan(vulkan::Buffer),
    #[cfg(dx12_backend)]
    Dx12(dx12::Buffer),
    #[cfg(metal_backend)]
    Metal(metal::Buffer),
}

pub(crate) enum NativePipeline {
    #[cfg(vulkan_backend)]
    Vulkan(vulkan::Pipeline),
    #[cfg(dx12_backend)]
    Dx12(dx12::Pipeline),
    #[cfg(metal_backend)]
    Metal(metal::Pipeline),
}

#[derive(Clone)]
pub(crate) enum NativeGroup {
    #[cfg(vulkan_backend)]
    Vulkan(Arc<vulkan::Group>),
    #[cfg(dx12_backend)]
    Dx12,
    #[cfg(metal_backend)]
    Metal,
}

pub(crate) fn open(
    request: &GpuRequest,
    cache: Option<PipelineCache>,
) -> Result<(NativeDevice, AdapterInfo, Limits), GpuUnavailable> {
    let mut reasons = Vec::new();
    let mut offered = Vec::new();
    let mut unsupported = None;
    #[cfg(dx12_backend)]
    if request.backends.contains(Backends::DX12) {
        match dx12::Device::open(request.adapter, cache.clone()) {
            Ok((device, info, limits)) if limits.supports(&Limits::BASELINE) => {
                return Ok((NativeDevice::Dx12(device), info, limits));
            }
            Ok((_, info, limits)) => unsupported = Some((info, limits)),
            Err(DeviceFailure::Missing { offered: found }) => offered.extend(found),
            Err(DeviceFailure::Unavailable { reason }) => reasons.push(format!("D3D12: {reason}")),
        }
    }
    #[cfg(metal_backend)]
    if request.backends.contains(Backends::METAL) {
        match metal::Device::open(request.adapter, cache.clone()) {
            Ok((device, info, limits)) if limits.supports(&Limits::BASELINE) => {
                return Ok((NativeDevice::Metal(device), info, limits));
            }
            Ok((_, info, limits)) => unsupported = Some((info, limits)),
            Err(DeviceFailure::Missing { offered: found }) => offered.extend(found),
            Err(DeviceFailure::Unavailable { reason }) => reasons.push(format!("Metal: {reason}")),
        }
    }
    #[cfg(vulkan_backend)]
    if request.backends.contains(Backends::VULKAN) {
        match vulkan::Device::open(request.adapter, cache.clone()) {
            Ok((device, info, limits)) if limits.supports(&Limits::BASELINE) => {
                return Ok((NativeDevice::Vulkan(device), info, limits));
            }
            Ok((_, info, limits)) => unsupported = Some((info, limits)),
            Err(DeviceFailure::Missing { offered: found }) => offered.extend(found),
            Err(DeviceFailure::Unavailable { reason }) => reasons.push(format!("Vulkan: {reason}")),
        }
    }
    if let Some((info, limits)) = unsupported {
        return Err(GpuUnavailable::UnsupportedLimits { info, limits });
    }
    if let AdapterPolicy::Identity(wanted) = request.adapter {
        return Err(GpuUnavailable::AdapterMissing { wanted, offered });
    }
    Err(GpuUnavailable::NoAdapter {
        reason: if reasons.is_empty() {
            "no requested compute backend is available on this platform".to_owned()
        } else {
            reasons.join("; ")
        },
    })
}

impl NativePipeline {
    pub(crate) fn is_compiled(&self) -> bool {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(pipeline) => pipeline.is_compiled(),
            #[cfg(dx12_backend)]
            Self::Dx12(pipeline) => pipeline.is_compiled(),
            #[cfg(metal_backend)]
            Self::Metal(pipeline) => pipeline.is_compiled(),
        }
    }
}

impl NativeDevice {
    pub(crate) fn compile(&self, pipeline: &NativePipeline, program: &ComputeProgram) {
        match (self, pipeline) {
            #[cfg(vulkan_backend)]
            (Self::Vulkan(device), NativePipeline::Vulkan(pipeline)) => {
                device.compile(pipeline, program)
            }
            #[cfg(dx12_backend)]
            (Self::Dx12(device), NativePipeline::Dx12(pipeline)) => {
                device.compile(pipeline, program)
            }
            #[cfg(metal_backend)]
            (Self::Metal(device), NativePipeline::Metal(pipeline)) => {
                device.compile(pipeline, program)
            }
            #[cfg(multiple_backends)]
            _ => panic!("a pipeline belongs to another backend"),
        }
    }

    pub(crate) fn create_buffer(
        &self,
        label: &str,
        size: u64,
        usage: BufferUsages,
    ) -> NativeBuffer {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => NativeBuffer::Vulkan(device.create_buffer(label, size, usage)),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => NativeBuffer::Dx12(device.create_buffer(label, size, usage)),
            #[cfg(metal_backend)]
            Self::Metal(device) => NativeBuffer::Metal(device.create_buffer(label, size, usage)),
        }
    }

    pub(crate) fn create_pipeline(&self, program: &ComputeProgram) -> NativePipeline {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => NativePipeline::Vulkan(device.create_pipeline(program)),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => NativePipeline::Dx12(device.create_pipeline(program)),
            #[cfg(metal_backend)]
            Self::Metal(device) => NativePipeline::Metal(device.create_pipeline(program)),
        }
    }

    pub(crate) fn create_group(
        &self,
        pipeline: &NativePipeline,
        buffers: &[BoundBuffer],
    ) -> NativeGroup {
        #[cfg(not(vulkan_backend))]
        let _ = buffers;
        match (self, pipeline) {
            #[cfg(vulkan_backend)]
            (Self::Vulkan(device), NativePipeline::Vulkan(pipeline)) => {
                NativeGroup::Vulkan(Arc::new(device.create_group(pipeline, buffers)))
            }
            #[cfg(dx12_backend)]
            (Self::Dx12(_), NativePipeline::Dx12(_)) => NativeGroup::Dx12,
            #[cfg(metal_backend)]
            (Self::Metal(_), NativePipeline::Metal(_)) => NativeGroup::Metal,
            #[cfg(multiple_backends)]
            _ => panic!("a pipeline belongs to another backend"),
        }
    }

    pub(crate) fn submit(&self, writes: &[Write], commands: &[Command]) -> u64 {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => device.submit(writes, commands),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => device.submit(writes, commands),
            #[cfg(metal_backend)]
            Self::Metal(device) => device.submit(writes, commands),
        }
    }

    #[cfg(test)]
    pub(crate) fn frames(&self) -> usize {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => device.frames(),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => device.frames(),
            #[cfg(metal_backend)]
            Self::Metal(device) => device.frames(),
        }
    }

    pub(crate) fn wait(&self, index: u64, timeout: Duration) {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => device.wait(index, timeout),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => device.wait(index, timeout),
            #[cfg(metal_backend)]
            Self::Metal(device) => device.wait(index, timeout),
        }
    }

    pub(crate) fn read(&self, buffer: &NativeBuffer, bytes: u64) -> Vec<u8> {
        match (self, buffer) {
            #[cfg(vulkan_backend)]
            (Self::Vulkan(device), NativeBuffer::Vulkan(buffer)) => device.read(buffer, bytes),
            #[cfg(dx12_backend)]
            (Self::Dx12(device), NativeBuffer::Dx12(buffer)) => device.read(buffer, bytes),
            #[cfg(metal_backend)]
            (Self::Metal(device), NativeBuffer::Metal(buffer)) => device.read(buffer, bytes),
            #[cfg(multiple_backends)]
            _ => panic!("a buffer belongs to another backend"),
        }
    }

    pub(crate) fn assert_alive(&self) {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => device.assert_alive(),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => device.assert_alive(),
            #[cfg(metal_backend)]
            Self::Metal(device) => device.assert_alive(),
        }
    }
}
