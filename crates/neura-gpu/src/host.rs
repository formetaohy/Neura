use crate::backend::Backend;
#[cfg(vulkan_backend)]
use ash::vk;
#[cfg(metal_backend)]
use objc2::rc::Retained;
#[cfg(metal_backend)]
use objc2::runtime::ProtocolObject;
#[cfg(metal_backend)]
use objc2_metal::MTLDevice;
#[cfg(dx12_backend)]
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
#[cfg(dx12_backend)]
use windows::core::Interface;

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

pub enum HostDevice<'a> {
    #[cfg(vulkan_backend)]
    Vulkan(VulkanHost<'a>),
    #[cfg(dx12_backend)]
    Dx12(&'a ID3D12Device),
    #[cfg(metal_backend)]
    Metal(&'a Retained<ProtocolObject<dyn MTLDevice>>),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum HostIdentity {
    #[cfg(vulkan_backend)]
    Vulkan { device: u64, queue: u64 },
    #[cfg(dx12_backend)]
    Dx12(usize),
    #[cfg(metal_backend)]
    Metal(u64),
}

impl HostDevice<'_> {
    pub(crate) fn backend(&self) -> Backend {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(_) => Backend::Vulkan,
            #[cfg(dx12_backend)]
            Self::Dx12(_) => Backend::Dx12,
            #[cfg(metal_backend)]
            Self::Metal(_) => Backend::Metal,
        }
    }

    pub(crate) fn identity(&self) -> HostIdentity {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(host) => HostIdentity::Vulkan {
                device: vk::Handle::as_raw(host.device.handle()),
                queue: vk::Handle::as_raw(host.queue),
            },
            #[cfg(dx12_backend)]
            Self::Dx12(device) => HostIdentity::Dx12(Interface::as_raw(*device) as usize),
            #[cfg(metal_backend)]
            Self::Metal(device) => HostIdentity::Metal(device.registryID()),
        }
    }
}
