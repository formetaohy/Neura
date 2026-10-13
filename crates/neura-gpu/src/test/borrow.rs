use crate::{
    AdapterId, AdapterPolicy, Backend, Binding, BufferUsages, Device, GpuBuffer, GpuContext,
    GpuRequest, GpuUnavailable, HostDevice, PREFERENCE, Submission,
};
use neura_compiler::{Read, ReadWrite, kernel};
use std::any::Any;
use std::sync::Arc;

#[kernel(workgroup_size = 64)]
fn double(lid: u32, input: Read<u32>, output: ReadWrite<u32>) {
    output[lid] = input[lid] * 2u32;
}

fn request(backend: Backend) -> GpuRequest {
    GpuRequest {
        backend: Some(backend),
        ..Default::default()
    }
}

fn cached(backend: Backend, name: &str) -> GpuRequest {
    GpuRequest {
        backend: Some(backend),
        artifacts: Some(std::env::temp_dir().join(format!(
            "neura-borrow-cache-{}-{backend:?}-{name}",
            std::process::id(),
        ))),
        ..Default::default()
    }
}

fn stranger(backend: Backend) -> AdapterId {
    match backend {
        Backend::Metal => AdapterId::MetalRegistry(u64::MAX),
        Backend::Dx12 | Backend::Vulkan => AdapterId::Numeric {
            vendor: u32::MAX,
            device: u32::MAX,
        },
    }
}

fn borrow(owner: &Owner, backend: Backend) -> Device {
    Device::borrow(owner.device(), &request(backend))
        .unwrap_or_else(|error| panic!("{backend:?} refused to lend its device: {error}"))
}

fn borrowed_compute(backend: Backend) {
    let owner = Owner::open(backend);
    let device = borrow(&owner, backend);
    assert_eq!(device.adapter_info().backend, backend);
    let context = GpuContext::of_device(device.clone());
    let queue = context.queue().clone();
    let input = GpuBuffer::new(
        &device,
        "borrowed input",
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_DST,
    );
    let output = GpuBuffer::new(
        &device,
        "borrowed output",
        256,
        BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    );
    let readback = GpuBuffer::new(
        &device,
        "borrowed readback",
        256,
        BufferUsages::COPY_DST | BufferUsages::MAP_READ,
    );
    let pipeline = context.declare(double());
    let group = pipeline.bind_group(&[
        Binding {
            index: 0,
            buffer: input.binding(0, 256),
        },
        Binding {
            index: 1,
            buffer: output.binding(0, 256),
        },
    ]);
    let values = (1..=64u32).collect::<Vec<_>>();
    let doubled = values.iter().map(|value| value * 2).collect::<Vec<_>>();
    input.write_at(&queue, 0, bytemuck::cast_slice(&values));
    let mut submission = Submission::new(&device, "borrowed compute");
    submission.dispatch(&pipeline, &group, [1, 1, 1]);
    submission.submit(&queue);
    let mut transfer = Submission::new(&device, "borrowed transfer");
    transfer.copy(&output, 0, &readback, 0, 256);
    let index = transfer.submit(&queue);
    assert_eq!(
        readback.read(&queue, index, 256),
        bytemuck::cast_slice::<u32, u8>(&doubled),
    );
}

#[test]
fn a_borrowed_device_runs_compute() {
    for &backend in PREFERENCE {
        borrowed_compute(backend);
    }
}

#[test]
fn one_request_of_one_host_device_reaches_one_device() {
    for &backend in PREFERENCE {
        let owner = Owner::open(backend);
        let first = borrow(&owner, backend);
        let second = borrow(&owner, backend);
        assert!(
            first.same(&second),
            "{backend:?} built two devices of one host device",
        );
    }
}

#[test]
fn another_request_of_one_host_device_shares_its_queue() {
    for &backend in PREFERENCE {
        let owner = Owner::open(backend);
        let first = borrow(&owner, backend);
        let second = Device::borrow(owner.device(), &cached(backend, "second"))
            .unwrap_or_else(|error| panic!("{backend:?} refused a second request: {error}"));
        assert!(!first.same(&second));
        assert!(
            first.native().same(second.native()),
            "{backend:?} lent two queues for one host device",
        );
    }
}

#[test]
fn a_queue_lock_of_the_host_serializes_every_borrow() {
    for &backend in PREFERENCE {
        let owner = Owner::open(backend);
        let holder = GpuContext::of_device(borrow(&owner, backend));
        let lock = holder.queue().lock();
        let peer = GpuContext::of_device(
            Device::borrow(owner.device(), &cached(backend, "locked"))
                .unwrap_or_else(|error| panic!("{backend:?} refused a second request: {error}")),
        );
        let device = peer.device().clone();
        let buffer = GpuBuffer::new(&device, "a locked clear", 256, BufferUsages::COPY_DST);
        let queue = peer.queue().clone();
        let (started, start) = std::sync::mpsc::channel();
        let (done, finish) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            scope.spawn(move || {
                started.send(()).expect("a started worker");
                let mut submission = Submission::new(&device, "a locked clear");
                submission.clear(&buffer, 0, 256);
                submission.submit(&queue);
                done.send(()).expect("a submitted worker");
            });
            start.recv().expect("a started worker");
            assert!(
                finish
                    .recv_timeout(std::time::Duration::from_millis(100))
                    .is_err(),
                "{backend:?} submitted to a host queue under the lock of the host",
            );
            drop(lock);
            finish
                .recv_timeout(crate::readback::READBACK_TIMEOUT)
                .expect("a submitted worker");
        });
    }
}

#[test]
fn a_host_advance_runs_while_a_guest_waits_for_the_host_queue() {
    for &backend in PREFERENCE {
        let owner = Owner::open(backend);
        let (advanced, ran) = std::sync::mpsc::channel();
        let device = Device::borrow(
            owner.device().advancing(crate::HostAdvance::of(move || {
                let _ = advanced.send(());
            })),
            &cached(backend, "advancing"),
        )
        .unwrap_or_else(|error| panic!("{backend:?} refused a progressing request: {error}"));
        let context = GpuContext::of_device(device.clone());
        let lock = context.queue().lock();
        let buffer = GpuBuffer::new(&device, "an advanced clear", 256, BufferUsages::COPY_DST);
        let queue = context.queue().clone();
        std::thread::scope(|scope| {
            scope.spawn(|| {
                let mut submission = Submission::new(&device, "an advanced clear");
                submission.clear(&buffer, 0, 256);
                submission.submit(&queue);
            });
            ran.recv_timeout(crate::readback::READBACK_TIMEOUT)
                .expect("a host advanced while a guest waited for its queue");
            drop(lock);
        });
    }
}

#[test]
fn a_borrow_holds_what_its_host_hands_over() {
    for &backend in PREFERENCE {
        let owner = Arc::new(Owner::open(backend));
        let watcher = Arc::downgrade(&owner);
        let keep_alive: Arc<dyn Any + Send + Sync> = owner.clone();
        let device = Device::borrow(owner.device().holding(keep_alive), &cached(backend, "held"))
            .unwrap_or_else(|error| panic!("{backend:?} refused a held request: {error}"));
        drop(owner);
        assert!(
            watcher.upgrade().is_some(),
            "{backend:?} dropped what a host handed over while its device was borrowed",
        );
        GpuContext::of_device(device.clone()).drain();
        drop(device);
        assert!(
            watcher.upgrade().is_none(),
            "{backend:?} held what a host handed over after its borrowed device was dropped",
        );
    }
}

#[test]
fn a_borrow_follows_the_identity_of_its_host_device() {
    for &backend in PREFERENCE {
        let first = Owner::open(backend);
        let second = Owner::open(backend);
        let one = borrow(&first, backend);
        let other = borrow(&second, backend);
        let same_host = first.device().identity() == second.device().identity();
        if backend == Backend::Vulkan {
            assert!(
                !same_host,
                "Vulkan opened one device for two hosts of its own",
            );
        }
        assert_eq!(
            one.same(&other),
            same_host,
            "{backend:?} borrowed apart from the identity of its host device",
        );
        assert_eq!(
            one.native().same(other.native()),
            same_host,
            "{backend:?} lent apart from the identity of its host device",
        );
    }
}

#[test]
fn a_borrowed_device_reports_the_adapter_of_the_host() {
    for &backend in PREFERENCE {
        let owner = Owner::open(backend);
        let borrowed = borrow(&owner, backend);
        let opened = Device::open(&request(backend)).expect("a native compute device");
        assert_eq!(borrowed.adapter_info(), opened.adapter_info());
        assert_eq!(borrowed.limits(), opened.limits());
    }
}

#[test]
fn a_borrow_of_another_adapter_is_refused() {
    for &backend in PREFERENCE {
        let owner = Owner::open(backend);
        let wanted = stranger(backend);
        let refused = Device::borrow(
            owner.device(),
            &GpuRequest {
                backend: Some(backend),
                adapter: AdapterPolicy::Identity(wanted),
                ..Default::default()
            },
        );
        match refused {
            Err(GpuUnavailable::AdapterMismatch {
                wanted: reported,
                host,
            }) => {
                assert_eq!(reported, wanted);
                assert_eq!(host.backend, backend);
                assert_ne!(host.id, wanted);
            }
            Err(error) => panic!("another adapter was refused for another reason: {error}"),
            Ok(device) => panic!("another adapter opened {}", device.adapter_info().name),
        }
    }
}

#[cfg(multiple_backends)]
#[test]
fn a_borrow_of_another_backend_is_refused() {
    let wanted = PREFERENCE[0];
    let elsewhere = PREFERENCE[1];
    let owner = Owner::open(elsewhere);
    match Device::borrow(owner.device(), &request(wanted)) {
        Err(GpuUnavailable::BackendMismatch {
            wanted: reported,
            host: lending,
        }) => {
            assert_eq!(reported, wanted);
            assert_eq!(lending, elsewhere);
        }
        Err(error) => panic!("another backend was refused for another reason: {error}"),
        Ok(device) => panic!("another backend opened {}", device.adapter_info().name),
    }
}

enum Owner {
    #[cfg(vulkan_backend)]
    Vulkan(Box<VulkanOwner>),
    #[cfg(dx12_backend)]
    Dx12(windows::Win32::Graphics::Direct3D12::ID3D12Device),
    #[cfg(metal_backend)]
    Metal(objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLDevice>>),
}

impl Owner {
    fn open(backend: Backend) -> Self {
        match backend {
            #[cfg(vulkan_backend)]
            Backend::Vulkan => Self::Vulkan(Box::new(VulkanOwner::open())),
            #[cfg(dx12_backend)]
            Backend::Dx12 => Self::Dx12(dx12_host()),
            #[cfg(metal_backend)]
            Backend::Metal => Self::Metal(
                objc2_metal::MTLCreateSystemDefaultDevice().expect("a Metal device of a host"),
            ),
            _ => unreachable!("every backend of this build lends a device"),
        }
    }

    fn device(&self) -> HostDevice<'_> {
        match self {
            #[cfg(vulkan_backend)]
            Self::Vulkan(owner) => owner.device(),
            #[cfg(dx12_backend)]
            Self::Dx12(device) => HostDevice::dx12(device),
            #[cfg(metal_backend)]
            Self::Metal(device) => HostDevice::metal(device),
        }
    }
}

#[cfg(vulkan_backend)]
struct VulkanOwner {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    physical: ash::vk::PhysicalDevice,
    queue: ash::vk::Queue,
    queue_family: u32,
    cooperative_matrix: bool,
}

#[cfg(vulkan_backend)]
impl VulkanOwner {
    fn open() -> Self {
        use ash::vk;

        let entry = unsafe { ash::Entry::load() }.expect("a Vulkan loader");
        let application = std::ffi::CString::new("a host").expect("a constant application name");
        let loaded = unsafe { entry.try_enumerate_instance_version() }
            .expect("the version of a Vulkan loader")
            .unwrap_or(vk::API_VERSION_1_0);
        let application_info = vk::ApplicationInfo::default()
            .application_name(&application)
            .api_version(loaded.min(vk::API_VERSION_1_3));
        let info = vk::InstanceCreateInfo::default().application_info(&application_info);
        let instance = unsafe { entry.create_instance(&info, None) }.expect("a Vulkan instance");
        let devices = unsafe { instance.enumerate_physical_devices() }.expect("Vulkan devices");
        let (physical, families) = devices
            .iter()
            .filter_map(|physical| {
                let properties = unsafe { instance.get_physical_device_properties(*physical) };
                if vk::api_version_major(properties.api_version) < 1
                    || (vk::api_version_major(properties.api_version) == 1
                        && vk::api_version_minor(properties.api_version) < 1)
                {
                    return None;
                }
                let families =
                    unsafe { instance.get_physical_device_queue_family_properties(*physical) };
                Some((*physical, families))
            })
            .find(|(_, families)| families.iter().any(computing))
            .expect("a Vulkan device of this host computes with timestamps");
        let family = families
            .iter()
            .enumerate()
            .filter(|(_, family)| computing(family))
            .max_by_key(|(_, family)| !family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(index, _)| index as u32)
            .expect("the compute queue family of a Vulkan device");
        let cooperative_matrix = cooperative_matrix(&instance, physical);
        let priority = [1.0f32];
        let mut queues = Vec::new();
        if family != 0 {
            queues.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(0)
                    .queue_priorities(&priority),
            );
        }
        queues.push(
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priority),
        );
        let mut extensions = Vec::new();
        let mut memory_model = vk::PhysicalDeviceVulkan12Features::default();
        let mut matrix = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
        let mut features = vk::PhysicalDeviceFeatures2::default();
        let mut config = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
        if cooperative_matrix {
            extensions.push(ash::khr::cooperative_matrix::NAME.as_ptr());
            memory_model.shader_float16 = vk::TRUE;
            memory_model.vulkan_memory_model = vk::TRUE;
            memory_model.vulkan_memory_model_device_scope = vk::TRUE;
            matrix.cooperative_matrix = vk::TRUE;
            features = features.push_next(&mut memory_model).push_next(&mut matrix);
            config = config.push_next(&mut features);
        }
        let config = config.enabled_extension_names(&extensions);
        let device = unsafe { instance.create_device(physical, &config, None) }
            .expect("a Vulkan compute device of a host");
        let queue = unsafe { device.get_device_queue(family, 0) };
        Self {
            entry,
            instance,
            device,
            physical,
            queue,
            queue_family: family,
            cooperative_matrix,
        }
    }

    fn device(&self) -> HostDevice<'_> {
        HostDevice::vulkan(crate::VulkanHost {
            entry: &self.entry,
            instance: &self.instance,
            physical: self.physical,
            device: &self.device,
            queue: self.queue,
            queue_family: self.queue_family,
            cooperative_matrix: self.cooperative_matrix,
        })
    }
}

#[cfg(vulkan_backend)]
impl Drop for VulkanOwner {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[cfg(vulkan_backend)]
fn computing(family: &ash::vk::QueueFamilyProperties) -> bool {
    family.queue_count > 0
        && family.queue_flags.contains(ash::vk::QueueFlags::COMPUTE)
        && family.timestamp_valid_bits > 0
}

#[cfg(vulkan_backend)]
fn cooperative_matrix(instance: &ash::Instance, physical: ash::vk::PhysicalDevice) -> bool {
    use ash::vk;
    use std::ffi::CStr;

    let available = unsafe { instance.enumerate_device_extension_properties(physical) }
        .expect("the extensions of a Vulkan device");
    let advertised = available.iter().any(|extension| unsafe {
        CStr::from_ptr(extension.extension_name.as_ptr()) == ash::khr::cooperative_matrix::NAME
    });
    if !advertised {
        return false;
    }
    let mut memory_model = vk::PhysicalDeviceVulkan12Features::default();
    let mut matrix = vk::PhysicalDeviceCooperativeMatrixFeaturesKHR::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut memory_model)
        .push_next(&mut matrix);
    unsafe { instance.get_physical_device_features2(physical, &mut features) };
    memory_model.shader_float16 == vk::TRUE
        && memory_model.vulkan_memory_model == vk::TRUE
        && memory_model.vulkan_memory_model_device_scope == vk::TRUE
        && matrix.cooperative_matrix == vk::TRUE
}

#[cfg(vulkan_backend)]
#[test]
fn a_borrowed_vulkan_device_reports_the_matrices_the_host_enabled() {
    let owner = VulkanOwner::open();
    let claimed = owner.cooperative_matrix;
    let device = Device::borrow(owner.device(), &request(Backend::Vulkan))
        .expect("a borrowed Vulkan device");
    assert_eq!(device.capability().cooperative_matrix.is_some(), claimed);
}

#[cfg(vulkan_backend)]
#[test]
fn a_borrowed_vulkan_device_leaves_the_device_of_the_host_alive() {
    use ash::vk;

    let owner = VulkanOwner::open();
    drop(Device::borrow(owner.device(), &request(Backend::Vulkan)).expect("a borrowed device"));
    let buffer = unsafe {
        owner.device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(4096)
                .usage(vk::BufferUsageFlags::STORAGE_BUFFER),
            None,
        )
    }
    .expect("the Vulkan device of the host still allocates");
    unsafe { owner.device.destroy_buffer(buffer, None) };
}

#[cfg(dx12_backend)]
fn dx12_host() -> windows::Win32::Graphics::Direct3D12::ID3D12Device {
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory1, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IDXGIAdapter1, IDXGIFactory1,
        IDXGIFactory6,
    };
    use windows::core::Interface;

    let factory: IDXGIFactory1 = unsafe { CreateDXGIFactory1() }.expect("DXGI");
    let factory: IDXGIFactory6 = factory.cast().expect("a DXGI factory of the sixth kind");
    let adapter: IDXGIAdapter1 =
        unsafe { factory.EnumAdapterByGpuPreference(0, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE) }
            .expect("a DXGI adapter of a host");
    let mut raw: Option<ID3D12Device> = None;
    unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut raw) }
        .expect("a D3D12 device of a host");
    raw.expect("a created D3D12 device")
}
