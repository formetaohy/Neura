use crate::library::PipelineLibrary;
use crate::pipeline::{ComputeProgram, PipelineHandle};
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc, Mutex};
use wgpu::{
    Adapter, AdapterInfo, Backend, Backends, Device, DeviceLostReason, DeviceType,
    ExperimentalFeatures, Features, Limits, PowerPreference, Queue,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitsPolicy {
    Minimum,
    Adapter,
}

pub struct GpuRequest {
    pub backends: Backends,
    pub power_preference: PowerPreference,
    pub device_name: Option<String>,
    pub required_features: Features,
    pub limits: LimitsPolicy,
}

impl Default for GpuRequest {
    fn default() -> Self {
        Self {
            backends: Self::NATIVE_BACKENDS,
            power_preference: PowerPreference::HighPerformance,
            device_name: None,
            required_features: Features::empty(),
            limits: LimitsPolicy::Adapter,
        }
    }
}

impl GpuRequest {
    pub const NATIVE_BACKENDS: Backends = Backends::DX12
        .union(Backends::METAL)
        .union(Backends::VULKAN);

    pub fn adapter_named(name: impl Into<String>) -> Self {
        Self {
            device_name: Some(name.into()),
            ..Self::default()
        }
    }

    pub fn minimum_limits(self) -> Self {
        Self {
            limits: LimitsPolicy::Minimum,
            ..self
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuUnavailable {
    NoAdapter {
        backends: Backends,
    },
    DeviceNotFound {
        requested: String,
        available: Vec<String>,
    },
    MissingFeatures {
        missing: Features,
        available: Features,
    },
    DeviceRejected {
        message: String,
    },
}

impl Display for GpuUnavailable {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter { backends } => {
                write!(out, "no adapter exposed by the backends {backends:?}")
            }
            Self::DeviceNotFound {
                requested,
                available,
            } => write!(
                out,
                "no adapter name contains {requested:?}; the device offers {available:?}"
            ),
            Self::MissingFeatures { missing, available } => {
                write!(
                    out,
                    "the adapter lacks {missing:?} and offers {available:?}"
                )
            }
            Self::DeviceRejected { message } => {
                write!(out, "the device request was rejected: {message}")
            }
        }
    }
}

impl std::error::Error for GpuUnavailable {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceLost {
    pub reason: DeviceLostReason,
    pub message: String,
}

impl Display for DeviceLost {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        write!(out, "{:?}: {}", self.reason, self.message)
    }
}

struct Health {
    lost: Mutex<Option<DeviceLost>>,
}

impl Default for Health {
    fn default() -> Self {
        Self {
            lost: Mutex::new(None),
        }
    }
}

pub struct GpuContext {
    info: AdapterInfo,
    device: Device,
    queue: Queue,
    features: Features,
    limits: Limits,
    health: Arc<Health>,
    pipelines: Arc<Mutex<PipelineLibrary>>,
}

impl GpuContext {
    pub const MINIMUM_LIMITS: Limits = Limits {
        max_storage_buffers_per_shader_stage: 8,
        max_storage_buffer_binding_size: 16 * 1024 * 1024,
        max_buffer_size: 64 * 1024 * 1024,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroups_per_dimension: 65_535,
        ..Limits::defaults()
    };

    pub async fn open(request: &GpuRequest) -> Result<Self, GpuUnavailable> {
        let adapter = select_adapter(request).await?;
        let available = adapter.features();
        let missing = request.required_features.difference(available);
        if !missing.is_empty() {
            return Err(GpuUnavailable::MissingFeatures { missing, available });
        }
        let limits = match request.limits {
            LimitsPolicy::Adapter => adapter.limits(),
            LimitsPolicy::Minimum => Self::MINIMUM_LIMITS,
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("neura device"),
                required_features: request.required_features,
                required_limits: limits.clone(),
                experimental_features: ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await
            .map_err(|error| GpuUnavailable::DeviceRejected {
                message: error.to_string(),
            })?;
        Ok(Self::of_device(device, queue, adapter.get_info(), limits))
    }

    pub fn adopt(device: Device, queue: Queue, info: AdapterInfo) -> Self {
        let limits = device.limits();
        Self::of_device(device, queue, info, limits)
    }

    fn of_device(device: Device, queue: Queue, info: AdapterInfo, limits: Limits) -> Self {
        let health = Arc::new(Health::default());
        let callback = health.clone();
        device.set_device_lost_callback(move |reason, message| {
            *callback
                .lost
                .lock()
                .expect("a device loss is never poisoned") = Some(DeviceLost { reason, message });
        });
        let features = device.features();
        Self {
            info,
            device,
            queue,
            features,
            limits,
            health,
            pipelines: Arc::new(Mutex::new(PipelineLibrary::new())),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.info
    }

    pub fn features(&self) -> Features {
        self.features
    }

    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    pub fn binding_alignment(&self) -> u64 {
        self.limits.min_storage_buffer_offset_alignment.max(16) as u64
    }

    pub fn device_lost(&self) -> Option<DeviceLost> {
        self.health
            .lost
            .lock()
            .expect("a device loss is never poisoned")
            .clone()
    }

    pub fn assert_alive(&self) {
        if let Some(lost) = self.device_lost() {
            panic!("the gpu device was lost: {lost}");
        }
    }

    pub fn poll(&self) {
        self.assert_alive();
        self.device
            .poll(wgpu::PollType::Poll)
            .expect("polling the gpu device failed");
        self.assert_alive();
    }

    pub fn declare(&self, program: ComputeProgram) -> PipelineHandle {
        self.pipelines
            .lock()
            .expect("a pipeline library is never poisoned")
            .declare(&self.device, program)
    }

    pub fn declared_kernels(&self) -> usize {
        self.pipelines
            .lock()
            .expect("a pipeline library is never poisoned")
            .declared()
    }
}

async fn select_adapter(request: &GpuRequest) -> Result<Adapter, GpuUnavailable> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: request.backends,
        flags: instance_flags(),
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let adapters = instance.enumerate_adapters(request.backends).await;
    if adapters.is_empty() {
        return Err(GpuUnavailable::NoAdapter {
            backends: request.backends,
        });
    }
    let candidates = match &request.device_name {
        Some(needle) => {
            let wanted = needle.to_lowercase();
            let matched = adapters
                .iter()
                .filter(|adapter| adapter.get_info().name.to_lowercase().contains(&wanted))
                .cloned()
                .collect::<Vec<_>>();
            if matched.is_empty() {
                return Err(GpuUnavailable::DeviceNotFound {
                    requested: needle.clone(),
                    available: adapters
                        .iter()
                        .map(|adapter| adapter.get_info().name)
                        .collect(),
                });
            }
            matched
        }
        None => adapters,
    };
    Ok(prefer(candidates, request.power_preference))
}

fn prefer(mut candidates: Vec<Adapter>, power: PowerPreference) -> Adapter {
    let rank = |adapter: &Adapter| {
        let info = adapter.get_info();
        let device = match power {
            PowerPreference::HighPerformance => match info.device_type {
                DeviceType::DiscreteGpu => 0u8,
                DeviceType::VirtualGpu => 1,
                DeviceType::IntegratedGpu => 2,
                _ => 3,
            },
            PowerPreference::LowPower => match info.device_type {
                DeviceType::IntegratedGpu => 0u8,
                DeviceType::VirtualGpu => 1,
                DeviceType::DiscreteGpu => 2,
                _ => 3,
            },
            PowerPreference::None => 0u8,
        };
        (native_backend_rank(info.backend), device)
    };
    candidates.sort_by_key(|adapter| rank(adapter));
    candidates
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("an adapter candidate list is never empty"))
}

fn native_backend_rank(backend: Backend) -> u8 {
    #[cfg(target_os = "windows")]
    const PRIMARY: Backend = Backend::Dx12;
    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "visionos"))]
    const PRIMARY: Backend = Backend::Metal;
    #[cfg(not(any(
        target_os = "windows",
        target_os = "macos",
        target_os = "ios",
        target_os = "visionos"
    )))]
    const PRIMARY: Backend = Backend::Vulkan;

    u8::from(backend != PRIMARY)
}

fn instance_flags() -> wgpu::InstanceFlags {
    let diagnostics = if cfg!(debug_assertions) {
        wgpu::InstanceFlags::VALIDATION
    } else {
        wgpu::InstanceFlags::empty()
    };
    diagnostics
        .union(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL)
        .with_env()
}
