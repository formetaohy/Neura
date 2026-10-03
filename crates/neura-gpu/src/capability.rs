use bitflags::bitflags;
pub use neura_shader::Backend;
use std::fmt::{self, Display, Formatter};

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Backends: u8 {
        const VULKAN = 1;
        const METAL = 2;
        const DX12 = 4;
    }
}

impl Backends {
    pub const COMPILED: Self = Self::from_bits_truncate(
        Self::bits_of(cfg!(dx12_backend), Self::DX12)
            | Self::bits_of(cfg!(metal_backend), Self::METAL)
            | Self::bits_of(cfg!(vulkan_backend), Self::VULKAN),
    );

    pub const PLATFORM: Self = Self::from_bits_truncate(
        Self::bits_of(cfg!(platform_dx12), Self::DX12)
            | Self::bits_of(cfg!(platform_metal), Self::METAL)
            | Self::bits_of(cfg!(platform_vulkan), Self::VULKAN),
    );

    const fn bits_of(enabled: bool, backend: Self) -> u8 {
        if enabled { backend.bits() } else { 0 }
    }

    pub fn backend(self) -> Backend {
        if self == Self::VULKAN {
            Backend::Vulkan
        } else if self == Self::METAL {
            Backend::Metal
        } else if self == Self::DX12 {
            Backend::Dx12
        } else {
            panic!("one backend names one shader language")
        }
    }
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct BufferUsages: u8 {
        const STORAGE = 1;
        const COPY_SRC = 2;
        const COPY_DST = 4;
        const MAP_READ = 8;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PowerPreference {
    HighPerformance,
    LowPower,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceType {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

impl DeviceType {
    pub(crate) const fn rank(self, preference: PowerPreference) -> u8 {
        match (preference, self) {
            (PowerPreference::HighPerformance, Self::Discrete)
            | (PowerPreference::LowPower, Self::Integrated) => 5,
            (PowerPreference::HighPerformance, Self::Integrated)
            | (PowerPreference::LowPower, Self::Discrete) => 4,
            (_, Self::Virtual) => 3,
            (_, Self::Cpu) => 2,
            (_, Self::Other) => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdapterId {
    Numeric { vendor: u32, device: u32 },
    MetalRegistry(u64),
}

impl Display for AdapterId {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Numeric { vendor, device } => {
                write!(out, "vendor {vendor:#06x} device {device:#06x}")
            }
            Self::MetalRegistry(registry) => write!(out, "Metal registry {registry}"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AdapterPolicy {
    Power(PowerPreference),
    Identity(AdapterId),
}

impl AdapterPolicy {
    pub(crate) fn wants(self, id: AdapterId) -> bool {
        match self {
            Self::Power(_) => true,
            Self::Identity(wanted) => wanted == id,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    pub id: AdapterId,
}

impl Display for AdapterInfo {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        write!(
            out,
            "{} ({}, {:?}, {:?})",
            self.name, self.id, self.device_type, self.backend,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CooperativeMatrix {
    pub subgroup: u32,
    pub rows: u32,
    pub columns: u32,
    pub depth: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Capability {
    pub cooperative_matrix: Option<CooperativeMatrix>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_storage_buffer_binding_size: u64,
    pub max_buffer_size: u64,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_size_x: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_workgroups_per_dimension: u32,
    pub min_storage_buffer_offset_alignment: u64,
}

impl Limits {
    pub const BASELINE: Self = Self {
        max_storage_buffers_per_shader_stage: 8,
        max_storage_buffer_binding_size: 16 << 20,
        max_buffer_size: 64 << 20,
        max_compute_invocations_per_workgroup: 256,
        max_compute_workgroup_size_x: 256,
        max_compute_workgroup_storage_size: 16 << 10,
        max_compute_workgroups_per_dimension: 65_535,
        min_storage_buffer_offset_alignment: 16,
    };

    pub(crate) fn supports(&self, required: &Self) -> bool {
        self.max_storage_buffers_per_shader_stage >= required.max_storage_buffers_per_shader_stage
            && self.max_storage_buffer_binding_size >= required.max_storage_buffer_binding_size
            && self.max_buffer_size >= required.max_buffer_size
            && self.max_compute_invocations_per_workgroup
                >= required.max_compute_invocations_per_workgroup
            && self.max_compute_workgroup_size_x >= required.max_compute_workgroup_size_x
            && self.max_compute_workgroup_storage_size
                >= required.max_compute_workgroup_storage_size
            && self.max_compute_workgroups_per_dimension
                >= required.max_compute_workgroups_per_dimension
    }

    pub(crate) fn minimum(self) -> Self {
        Self {
            min_storage_buffer_offset_alignment: self.min_storage_buffer_offset_alignment,
            ..Self::BASELINE
        }
    }
}
