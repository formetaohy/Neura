pub use neura_shader::Backend;

pub const PREFERENCE: &[Backend] = &[
    #[cfg(dx12_backend)]
    Backend::Dx12,
    #[cfg(metal_backend)]
    Backend::Metal,
    #[cfg(vulkan_backend)]
    Backend::Vulkan,
];
