use crate::backend::Backend;
#[cfg(vulkan_backend)]
use ash::vk;
#[cfg(metal_backend)]
use objc2::rc::Retained;
#[cfg(metal_backend)]
use objc2::runtime::ProtocolObject;
#[cfg(metal_backend)]
use objc2_metal::MTLDevice;
use std::any::Any;
use std::sync::Arc;
use std::time::Duration;
#[cfg(dx12_backend)]
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
#[cfg(dx12_backend)]
use windows::core::Interface;

pub(crate) const HOST_SLICE: Duration = Duration::from_millis(1);

#[cfg(metal_backend)]
pub(crate) fn metal_identity(device: &Retained<ProtocolObject<dyn MTLDevice>>) -> usize {
    Retained::as_ptr(device).cast::<()>() as usize
}

#[cfg(vulkan_backend)]
pub struct VulkanHost<'a> {
    pub entry: &'a ash::Entry,
    pub instance: &'a ash::Instance,
    pub physical: vk::PhysicalDevice,
    pub device: &'a ash::Device,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub cooperative_matrix: bool,
}

#[derive(Clone)]
pub struct HostAdvance(Arc<dyn Fn() + Send + Sync>);

impl HostAdvance {
    pub fn of(advance: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(advance))
    }

    pub(crate) fn advance(&self) {
        (self.0)()
    }
}

pub struct HostDevice<'a> {
    pub(crate) kind: HostKind<'a>,
    pub(crate) advance: Option<HostAdvance>,
    pub(crate) hold: Option<Arc<dyn Any + Send + Sync>>,
}

pub(crate) enum HostKind<'a> {
    #[cfg(vulkan_backend)]
    Vulkan(VulkanHost<'a>),
    #[cfg(dx12_backend)]
    Dx12(&'a ID3D12Device),
    #[cfg(metal_backend)]
    Metal(&'a Retained<ProtocolObject<dyn MTLDevice>>),
}

impl<'a> HostDevice<'a> {
    #[cfg(vulkan_backend)]
    pub fn vulkan(host: VulkanHost<'a>) -> Self {
        Self {
            kind: HostKind::Vulkan(host),
            advance: None,
            hold: None,
        }
    }

    #[cfg(dx12_backend)]
    pub fn dx12(device: &'a ID3D12Device) -> Self {
        Self {
            kind: HostKind::Dx12(device),
            advance: None,
            hold: None,
        }
    }

    #[cfg(metal_backend)]
    pub fn metal(device: &'a Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            kind: HostKind::Metal(device),
            advance: None,
            hold: None,
        }
    }

    pub fn advancing(self, advance: HostAdvance) -> Self {
        Self {
            advance: Some(advance),
            ..self
        }
    }

    pub fn holding(self, hold: Arc<dyn Any + Send + Sync>) -> Self {
        Self {
            hold: Some(hold),
            ..self
        }
    }

    pub(crate) fn advance(&self) -> Option<HostAdvance> {
        self.advance.clone()
    }

    pub(crate) fn backend(&self) -> Backend {
        match &self.kind {
            #[cfg(vulkan_backend)]
            HostKind::Vulkan(_) => Backend::Vulkan,
            #[cfg(dx12_backend)]
            HostKind::Dx12(_) => Backend::Dx12,
            #[cfg(metal_backend)]
            HostKind::Metal(_) => Backend::Metal,
        }
    }

    pub fn identity(&self) -> HostIdentity {
        match &self.kind {
            #[cfg(vulkan_backend)]
            HostKind::Vulkan(host) => HostIdentity::of(
                Backend::Vulkan,
                vk::Handle::as_raw(host.device.handle()),
                vk::Handle::as_raw(host.queue),
            ),
            #[cfg(dx12_backend)]
            HostKind::Dx12(device) => {
                HostIdentity::of(Backend::Dx12, Interface::as_raw(*device) as usize as u64, 0)
            }
            #[cfg(metal_backend)]
            HostKind::Metal(device) => {
                HostIdentity::of(Backend::Metal, metal_identity(device) as u64, 0)
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct HostIdentity {
    backend: Backend,
    device: u64,
    queue: u64,
}

impl HostIdentity {
    pub(crate) fn of(backend: Backend, device: u64, queue: u64) -> Self {
        Self {
            backend,
            device,
            queue,
        }
    }
}
