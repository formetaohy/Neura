#[cfg(dx12_backend)]
pub(crate) mod dx12;
#[cfg(metal_backend)]
pub(crate) mod metal;
#[cfg(vulkan_backend)]
pub(crate) mod vulkan;

use crate::backend::{Backend, PREFERENCE};
use crate::cache::ArtifactCache;
use crate::capability::{AdapterInfo, AdapterPolicy, BufferUsages, Capability, Limits};
use crate::context::{GpuRequest, GpuUnavailable};
use crate::host::HostDevice;
use crate::pipeline::BoundBuffer;
use crate::submission::{Command, Write};
use neura_shader::ComputeProgram;
use std::sync::Arc;
use std::time::Duration;

pub(crate) const FRAMES_IN_FLIGHT: usize = 4;
pub(crate) const FRAME_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) fn hang_deadline(
    frames: impl IntoIterator<Item = (u64, bool)>,
    index: u64,
    timeout: Duration,
) -> Duration {
    let waits_for_a_first_dispatch = frames
        .into_iter()
        .any(|(frame, first_dispatch)| first_dispatch && frame <= index);
    if waits_for_a_first_dispatch {
        Duration::MAX
    } else {
        timeout
    }
}
pub(crate) const STAGING_BYTES: u64 = 256 << 10;
pub(crate) const TIME_SLOTS: usize = 4096;

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

    fn unusable(self, backend: Backend) -> GpuUnavailable {
        match self {
            Self::Unavailable { reason } => GpuUnavailable::HostUnusable { backend, reason },
            Self::Missing { .. } => unreachable!("a borrowed device enumerates no adapter"),
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

pub(crate) enum NativeQueueGuard<'a> {
    #[cfg(vulkan_backend)]
    Vulkan { _guard: vulkan::QueueGuard<'a> },
    #[cfg(dx12_backend)]
    Dx12 { _guard: dx12::QueueGuard<'a> },
    #[cfg(metal_backend)]
    Metal { _guard: metal::QueueGuard<'a> },
}

pub(crate) fn open(
    request: &GpuRequest,
) -> Result<(NativeDevice, AdapterInfo, Limits, Capability), GpuUnavailable> {
    let order: &[Backend] = match &request.backend {
        Some(backend) if PREFERENCE.contains(backend) => std::slice::from_ref(backend),
        Some(backend) => {
            return Err(GpuUnavailable::BackendMissing {
                wanted: *backend,
                offered: PREFERENCE,
            });
        }
        None => PREFERENCE,
    };
    let mut reasons = Vec::new();
    let mut offered = Vec::new();
    let mut unsupported = None;
    for backend in order {
        match candidate(*backend, request.adapter) {
            Ok((device, info, limits, capability)) if limits.supports(&Limits::BASELINE) => {
                return Ok((device, info, limits, capability));
            }
            Ok((_, info, limits, _)) => unsupported = Some((info, limits)),
            Err(DeviceFailure::Missing { offered: found }) => offered.extend(found),
            Err(DeviceFailure::Unavailable { reason }) => {
                reasons.push(format!("{backend:?}: {reason}"));
            }
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

pub(crate) fn borrow(
    host: HostDevice<'_>,
    request: &GpuRequest,
) -> Result<(NativeDevice, AdapterInfo, Limits, Capability), GpuUnavailable> {
    let backend = host.backend();
    if let Some(wanted) = request.backend
        && wanted != backend
    {
        return Err(GpuUnavailable::BackendMismatch {
            wanted,
            host: backend,
        });
    }
    let (native, info, limits, capability) = match host {
        #[cfg(dx12_backend)]
        HostDevice::Dx12(raw) => {
            dx12::Device::borrow(raw).map(|(device, info, limits, capability)| {
                (NativeDevice::Dx12(device), info, limits, capability)
            })
        }
        #[cfg(metal_backend)]
        HostDevice::Metal(raw) => {
            metal::Device::borrow(raw).map(|(device, info, limits, capability)| {
                (NativeDevice::Metal(device), info, limits, capability)
            })
        }
        #[cfg(vulkan_backend)]
        HostDevice::Vulkan(raw) => {
            vulkan::Device::borrow(raw).map(|(device, info, limits, capability)| {
                (NativeDevice::Vulkan(device), info, limits, capability)
            })
        }
    }
    .map_err(|failure| failure.unusable(backend))?;
    if let AdapterPolicy::Identity(wanted) = request.adapter
        && wanted != info.id
    {
        return Err(GpuUnavailable::AdapterMismatch { wanted, host: info });
    }
    Ok((native, info, limits, capability))
}

fn candidate(
    backend: Backend,
    policy: AdapterPolicy,
) -> Result<(NativeDevice, AdapterInfo, Limits, Capability), DeviceFailure> {
    match backend {
        #[cfg(dx12_backend)]
        Backend::Dx12 => dx12::Device::open(policy).map(|(device, info, limits, capability)| {
            (NativeDevice::Dx12(device), info, limits, capability)
        }),
        #[cfg(metal_backend)]
        Backend::Metal => metal::Device::open(policy).map(|(device, info, limits, capability)| {
            (NativeDevice::Metal(device), info, limits, capability)
        }),
        #[cfg(vulkan_backend)]
        Backend::Vulkan => {
            vulkan::Device::open(policy).map(|(device, info, limits, capability)| {
                (NativeDevice::Vulkan(device), info, limits, capability)
            })
        }
        _ => unreachable!("every backend of this build opens a candidate"),
    }
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
    pub(crate) fn compile(
        &self,
        pipeline: &NativePipeline,
        program: &ComputeProgram,
        artifacts: &ArtifactCache,
    ) {
        match (self, pipeline) {
            #[cfg(vulkan_backend)]
            (Self::Vulkan(device), NativePipeline::Vulkan(pipeline)) => {
                device.compile(pipeline, program, artifacts)
            }
            #[cfg(dx12_backend)]
            (Self::Dx12(device), NativePipeline::Dx12(pipeline)) => {
                device.compile(pipeline, program, artifacts)
            }
            #[cfg(metal_backend)]
            (Self::Metal(device), NativePipeline::Metal(pipeline)) => {
                device.compile(pipeline, program, artifacts)
            }
            #[cfg(multiple_backends)]
            _ => panic!("a pipeline belongs to another backend"),
        }
    }

    pub(crate) fn guard(&self) -> NativeQueueGuard<'_> {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => NativeQueueGuard::Vulkan {
                _guard: device.guard(),
            },
            #[cfg(dx12_backend)]
            Self::Dx12(device) => NativeQueueGuard::Dx12 {
                _guard: device.guard(),
            },
            #[cfg(metal_backend)]
            Self::Metal(device) => NativeQueueGuard::Metal {
                _guard: device.guard(),
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn same(&self, other: &Self) -> bool {
        match (self, other) {
            #[cfg(vulkan_backend)]
            (Self::Vulkan(first), Self::Vulkan(second)) => Arc::ptr_eq(first, second),
            #[cfg(dx12_backend)]
            (Self::Dx12(first), Self::Dx12(second)) => Arc::ptr_eq(first, second),
            #[cfg(metal_backend)]
            (Self::Metal(first), Self::Metal(second)) => Arc::ptr_eq(first, second),
            #[cfg(multiple_backends)]
            _ => false,
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

    pub(crate) fn submit(
        &self,
        writes: &[Write],
        commands: &[Command],
        first_dispatch: bool,
    ) -> u64 {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => device.submit(writes, commands, first_dispatch),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => device.submit(writes, commands, first_dispatch),
            #[cfg(metal_backend)]
            Self::Metal(device) => device.submit(writes, commands, first_dispatch),
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

    pub(crate) fn complete(&self, index: u64) -> bool {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => device.complete(index),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => device.complete(index),
            #[cfg(metal_backend)]
            Self::Metal(device) => device.complete(index),
        }
    }

    pub(crate) fn seconds(&self, index: u64) -> f64 {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(device) => device.seconds(index),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => device.seconds(index),
            #[cfg(metal_backend)]
            Self::Metal(device) => device.seconds(index),
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
