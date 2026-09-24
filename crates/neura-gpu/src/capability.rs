use bitflags::bitflags;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Backend {
    Vulkan,
    Metal,
    Dx12,
}

bitflags! {
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct Backends: u8 {
        const VULKAN = 1;
        const METAL = 2;
        const DX12 = 4;
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PowerPreference {
    HighPerformance,
    LowPower,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceType {
    Discrete,
    Integrated,
    Virtual,
    Cpu,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdapterId {
    Numeric { vendor: u32, device: u32 },
    MetalRegistry(u64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterInfo {
    pub name: String,
    pub backend: Backend,
    pub device_type: DeviceType,
    pub id: AdapterId,
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
