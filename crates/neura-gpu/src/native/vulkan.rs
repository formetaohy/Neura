use super::{NativeBuffer, NativeGroup, NativePipeline};
use crate::buffer::GpuBuffer;
use crate::capability::{
    AdapterId, AdapterInfo, Backend, BufferUsages, DeviceType, Limits, PowerPreference,
};
use crate::pipeline::{BoundBuffer, ComputeProgram};
use crate::submission::{Command, Write};
use ash::{Entry, Instance, vk};
use std::any::Any;
use std::collections::VecDeque;
use std::ffi::{CStr, CString};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

struct InstanceOwner {
    _entry: Entry,
    raw: Instance,
}

impl Drop for InstanceOwner {
    fn drop(&mut self) {
        unsafe { self.raw.destroy_instance(None) };
    }
}

struct InFlight {
    index: u64,
    command: vk::CommandBuffer,
    fence: vk::Fence,
    _resources: Vec<Arc<dyn Any + Send + Sync>>,
}

#[derive(Default)]
struct QueueState {
    next: u64,
    completed: u64,
    in_flight: VecDeque<InFlight>,
}

pub(crate) struct Device {
    _owner: InstanceOwner,
    raw: ash::Device,
    memory: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    pool: vk::CommandPool,
    state: Mutex<QueueState>,
}

pub(crate) struct BufferResource {
    raw: ash::Device,
    handle: vk::Buffer,
    memory: vk::DeviceMemory,
}

#[derive(Clone)]
pub(crate) struct Buffer {
    resource: Arc<BufferResource>,
}

struct PipelineResource {
    raw: ash::Device,
    layout: vk::PipelineLayout,
    group_layout: vk::DescriptorSetLayout,
    compiled: OnceLock<vk::Pipeline>,
}

pub(crate) struct Pipeline {
    resource: Arc<PipelineResource>,
}

pub(crate) struct Group {
    resource: Arc<GroupResource>,
}

struct GroupResource {
    raw: ash::Device,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    _buffers: Vec<Arc<BufferResource>>,
    _pipeline: Arc<PipelineResource>,
}

fn native_buffer(native: &NativeBuffer) -> &Arc<BufferResource> {
    match native {
        NativeBuffer::Vulkan(buffer) => &buffer.resource,
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        _ => panic!("a Vulkan command cannot use another backend's buffer"),
    }
}

fn buffer(gpu: &GpuBuffer) -> &Arc<BufferResource> {
    native_buffer(gpu.native())
}

fn pipeline(native: &NativePipeline) -> &Arc<PipelineResource> {
    match native {
        NativePipeline::Vulkan(pipeline) => &pipeline.resource,
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        _ => panic!("a Vulkan command cannot use another backend's pipeline"),
    }
}

fn group(native: &NativeGroup) -> &Arc<GroupResource> {
    match native {
        NativeGroup::Vulkan(group) => &group.resource,
        #[cfg(any(target_os = "windows", target_os = "macos"))]
        _ => panic!("a Vulkan command cannot use another backend's bind group"),
    }
}

fn device_type(device: vk::PhysicalDeviceType) -> DeviceType {
    match device {
        vk::PhysicalDeviceType::DISCRETE_GPU => DeviceType::Discrete,
        vk::PhysicalDeviceType::INTEGRATED_GPU => DeviceType::Integrated,
        vk::PhysicalDeviceType::VIRTUAL_GPU => DeviceType::Virtual,
        vk::PhysicalDeviceType::CPU => DeviceType::Cpu,
        _ => DeviceType::Other,
    }
}

fn rank(ty: DeviceType, preference: PowerPreference) -> u8 {
    match (preference, ty) {
        (PowerPreference::HighPerformance, DeviceType::Discrete)
        | (PowerPreference::LowPower, DeviceType::Integrated) => 5,
        (PowerPreference::HighPerformance, DeviceType::Integrated)
        | (PowerPreference::LowPower, DeviceType::Discrete) => 4,
        (_, DeviceType::Virtual) => 3,
        (_, DeviceType::Cpu) => 2,
        _ => 1,
    }
}

fn complete_enumeration<T>(
    mut enumerate: impl FnMut() -> Result<T, vk::Result>,
) -> Result<T, vk::Result> {
    for _ in 0..32 {
        match enumerate() {
            Err(vk::Result::INCOMPLETE) => std::thread::yield_now(),
            result => return result,
        }
    }
    Err(vk::Result::INCOMPLETE)
}

impl Device {
    pub(crate) fn open(
        preference: PowerPreference,
    ) -> Result<(Arc<Self>, AdapterInfo, Limits), String> {
        let entry = unsafe { Entry::load() }.map_err(|error| error.to_string())?;
        let extensions =
            complete_enumeration(|| unsafe { entry.enumerate_instance_extension_properties(None) })
                .map_err(|error| format!("enumerating extensions: {error:?}"))?;
        let portability = extensions.iter().any(|extension| unsafe {
            CStr::from_ptr(extension.extension_name.as_ptr())
                == ash::khr::portability_enumeration::NAME
        });
        let extension_names = if portability {
            vec![ash::khr::portability_enumeration::NAME.as_ptr()]
        } else {
            Vec::new()
        };
        let diagnostics =
            CString::new("VK_LAYER_KHRONOS_validation").expect("a constant layer name");
        let layers = if cfg!(debug_assertions)
            && complete_enumeration(|| unsafe { entry.enumerate_instance_layer_properties() })
                .map_err(|error| format!("enumerating layers: {error:?}"))?
                .iter()
                .any(|layer| unsafe {
                    CStr::from_ptr(layer.layer_name.as_ptr()) == diagnostics.as_c_str()
                }) {
            vec![diagnostics.as_ptr()]
        } else {
            Vec::new()
        };
        let application = CString::new("neura").expect("a constant application name");
        let application_info = vk::ApplicationInfo::default()
            .application_name(&application)
            .api_version(vk::API_VERSION_1_1);
        let flags = if portability {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::empty()
        };
        let info = vk::InstanceCreateInfo::default()
            .application_info(&application_info)
            .flags(flags)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layers);
        let instance = unsafe { entry.create_instance(&info, None) }
            .map_err(|error| format!("creating an instance: {error:?}"))?;
        let owner = InstanceOwner {
            _entry: entry,
            raw: instance,
        };
        let physical = complete_enumeration(|| unsafe { owner.raw.enumerate_physical_devices() })
            .map_err(|error| format!("enumerating devices: {error:?}"))?;
        let mut candidates = physical
            .iter()
            .filter_map(|physical| {
                let props = unsafe { owner.raw.get_physical_device_properties(*physical) };
                if vk::api_version_major(props.api_version) < 1
                    || (vk::api_version_major(props.api_version) == 1
                        && vk::api_version_minor(props.api_version) < 1)
                {
                    return None;
                }
                let queues = unsafe {
                    owner
                        .raw
                        .get_physical_device_queue_family_properties(*physical)
                };
                let family = queues
                    .iter()
                    .enumerate()
                    .filter(|(_, queue)| {
                        queue.queue_count > 0 && queue.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    })
                    .min_by_key(|(_, queue)| queue.queue_flags.contains(vk::QueueFlags::GRAPHICS))?
                    .0 as u32;
                let ty = device_type(props.device_type);
                let memory = unsafe { owner.raw.get_physical_device_memory_properties(*physical) };
                let limits = limits(&props, &memory);
                Some((*physical, props, family, ty, limits, memory))
            })
            .collect::<Vec<_>>();
        candidates.sort_by_key(|(_, _, _, ty, limits, _)| {
            (limits.supports(&Limits::BASELINE), rank(*ty, preference))
        });
        let mut failures = Vec::new();
        for (physical, props, family, ty, limits, memory) in candidates.into_iter().rev() {
            let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            let priority = [1.0];
            let queues = [vk::DeviceQueueCreateInfo::default()
                .queue_family_index(family)
                .queue_priorities(&priority)];
            let available = complete_enumeration(|| unsafe {
                owner.raw.enumerate_device_extension_properties(physical)
            });
            let available = match available {
                Ok(available) => available,
                Err(error) => {
                    failures.push(format!("{name}: enumerating extensions failed: {error:?}"));
                    continue;
                }
            };
            let portability = available.iter().any(|extension| unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr())
                    == ash::khr::portability_subset::NAME
            });
            let device_extensions = if portability {
                vec![ash::khr::portability_subset::NAME.as_ptr()]
            } else {
                Vec::new()
            };
            let config = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&device_extensions);
            let raw = match unsafe { owner.raw.create_device(physical, &config, None) } {
                Ok(raw) => raw,
                Err(error) => {
                    failures.push(format!(
                        "{name}: creating a compute device failed: {error:?}"
                    ));
                    continue;
                }
            };
            let queue = unsafe { raw.get_device_queue(family, 0) };
            let pool = match unsafe {
                raw.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
            } {
                Ok(pool) => pool,
                Err(error) => {
                    failures.push(format!(
                        "{name}: creating a compute command pool failed: {error:?}"
                    ));
                    unsafe { raw.destroy_device(None) };
                    continue;
                }
            };
            let adapter_info = AdapterInfo {
                name,
                backend: Backend::Vulkan,
                device_type: ty,
                id: AdapterId::Numeric {
                    vendor: props.vendor_id,
                    device: props.device_id,
                },
            };
            return Ok((
                Arc::new(Self {
                    _owner: owner,
                    raw,
                    memory,
                    queue,
                    pool,
                    state: Mutex::new(QueueState::default()),
                }),
                adapter_info,
                limits,
            ));
        }
        Err(if failures.is_empty() {
            "no Vulkan 1.1 device with a compute queue".to_owned()
        } else {
            failures.join("; ")
        })
    }

    fn allocate(&self, size: u64, usage: vk::BufferUsageFlags, host: bool) -> Arc<BufferResource> {
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let handle = unsafe { self.raw.create_buffer(&info, None) }
            .unwrap_or_else(|error| panic!("allocating {size} Vulkan buffer bytes: {error:?}"));
        let requirements = unsafe { self.raw.get_buffer_memory_requirements(handle) };
        let wanted = if host {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let memory_type = (0..self.memory.memory_type_count)
            .find(|index| {
                (requirements.memory_type_bits & (1 << index)) != 0
                    && self.memory.memory_types[*index as usize]
                        .property_flags
                        .contains(wanted)
            })
            .or_else(|| {
                (!host)
                    .then(|| {
                        (0..self.memory.memory_type_count)
                            .find(|index| (requirements.memory_type_bits & (1 << index)) != 0)
                    })
                    .flatten()
            })
            .unwrap_or_else(|| {
                panic!(
                    "a Vulkan buffer has no suitable {} memory",
                    if host { "coherent host" } else { "device" }
                )
            });
        let memory = unsafe {
            self.raw.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )
        }
        .unwrap_or_else(|error| panic!("allocating Vulkan device memory: {error:?}"));
        unsafe { self.raw.bind_buffer_memory(handle, memory, 0) }
            .unwrap_or_else(|error| panic!("binding Vulkan device memory: {error:?}"));
        Arc::new(BufferResource {
            raw: self.raw.clone(),
            handle,
            memory,
        })
    }

    pub(crate) fn create_buffer(
        self: &Arc<Self>,
        _label: &str,
        size: u64,
        usage: BufferUsages,
    ) -> Buffer {
        let mut flags = vk::BufferUsageFlags::empty();
        if usage.contains(BufferUsages::STORAGE) {
            flags |= vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if usage.contains(BufferUsages::COPY_SRC) {
            flags |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if usage.contains(BufferUsages::COPY_DST) {
            flags |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        Buffer {
            resource: self.allocate(size, flags, usage.contains(BufferUsages::MAP_READ)),
        }
    }

    pub(crate) fn create_pipeline(self: &Arc<Self>, program: &ComputeProgram) -> Pipeline {
        let bindings = program
            .bindings()
            .iter()
            .map(|spec| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(spec.binding)
                    .descriptor_type(if spec.dynamic_offset {
                        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect::<Vec<_>>();
        let group_layout = unsafe {
            self.raw.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )
        }
        .unwrap_or_else(|error| panic!("creating a Vulkan storage layout: {error:?}"));
        let layouts = [group_layout];
        let layout = unsafe {
            self.raw.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&layouts),
                None,
            )
        }
        .unwrap_or_else(|error| panic!("creating a Vulkan compute layout: {error:?}"));
        Pipeline {
            resource: Arc::new(PipelineResource {
                raw: self.raw.clone(),
                layout,
                group_layout,
                compiled: OnceLock::new(),
            }),
        }
    }

    pub(crate) fn create_group(
        self: &Arc<Self>,
        pipeline: &Pipeline,
        buffers: &[BoundBuffer],
    ) -> Group {
        let regular = buffers.iter().filter(|binding| !binding.dynamic).count() as u32;
        let dynamic = buffers.len() as u32 - regular;
        let sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: regular,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
                descriptor_count: dynamic,
            },
        ];
        let sizes = sizes
            .iter()
            .copied()
            .filter(|size| size.descriptor_count > 0)
            .collect::<Vec<_>>();
        let pool = unsafe {
            self.raw.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )
        }
        .unwrap_or_else(|error| panic!("creating a Vulkan storage descriptor pool: {error:?}"));
        let layouts = [pipeline.resource.group_layout];
        let set = unsafe {
            self.raw.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(pool)
                    .set_layouts(&layouts),
            )
        }
        .unwrap_or_else(|error| panic!("allocating Vulkan storage descriptors: {error:?}"))[0];
        let infos = buffers
            .iter()
            .map(|binding| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buffer(&binding.buffer).handle)
                    .offset(binding.offset)
                    .range(binding.size)
            })
            .collect::<Vec<_>>();
        let writes = buffers
            .iter()
            .zip(&infos)
            .enumerate()
            .map(|(index, (binding, info))| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(index as u32)
                    .descriptor_type(if binding.dynamic {
                        vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
                    } else {
                        vk::DescriptorType::STORAGE_BUFFER
                    })
                    .buffer_info(std::slice::from_ref(info))
            })
            .collect::<Vec<_>>();
        unsafe { self.raw.update_descriptor_sets(&writes, &[]) };
        Group {
            resource: Arc::new(GroupResource {
                raw: self.raw.clone(),
                pool,
                set,
                _buffers: buffers
                    .iter()
                    .map(|binding| buffer(&binding.buffer).clone())
                    .collect(),
                _pipeline: pipeline.resource.clone(),
            }),
        }
    }

    fn barrier(&self, command: vk::CommandBuffer) {
        let memory = [vk::MemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
            .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)];
        unsafe {
            self.raw.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &memory,
                &[],
                &[],
            );
        }
    }

    pub(crate) fn submit(&self, writes: &[Write], commands: &[Command]) -> u64 {
        let mut state = self
            .state
            .lock()
            .expect("the Vulkan queue is never poisoned");
        self.retire(&mut state);
        if writes.is_empty() && commands.is_empty() {
            return state.next;
        }
        let command = unsafe {
            self.raw.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .unwrap_or_else(|error| panic!("allocating a Vulkan compute command buffer: {error:?}"))[0];
        unsafe {
            self.raw.begin_command_buffer(
                command,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .unwrap_or_else(|error| panic!("starting a Vulkan compute submission: {error:?}"));
        let mut resources: Vec<Arc<dyn Any + Send + Sync>> = Vec::new();
        for write in writes {
            let staging = self.allocate(
                write.bytes.len() as u64,
                vk::BufferUsageFlags::TRANSFER_SRC,
                true,
            );
            let mapped = unsafe {
                self.raw.map_memory(
                    staging.memory,
                    0,
                    write.bytes.len() as u64,
                    vk::MemoryMapFlags::empty(),
                )
            }
            .unwrap_or_else(|error| panic!("mapping a Vulkan upload: {error:?}"));
            unsafe {
                std::ptr::copy_nonoverlapping(
                    write.bytes.as_ptr(),
                    mapped.cast::<u8>(),
                    write.bytes.len(),
                );
                self.raw.unmap_memory(staging.memory);
            }
            self.barrier(command);
            let region = [vk::BufferCopy::default()
                .src_offset(0)
                .dst_offset(write.offset)
                .size(write.bytes.len() as u64)];
            unsafe {
                self.raw.cmd_copy_buffer(
                    command,
                    staging.handle,
                    native_buffer(&write.buffer).handle,
                    &region,
                )
            };
            resources.push(staging);
            resources.push(native_buffer(&write.buffer).clone());
        }
        for operation in commands {
            self.barrier(command);
            match operation {
                Command::Copy {
                    source,
                    source_offset,
                    target,
                    target_offset,
                    bytes,
                } => {
                    let region = [vk::BufferCopy::default()
                        .src_offset(*source_offset)
                        .dst_offset(*target_offset)
                        .size(*bytes)];
                    unsafe {
                        self.raw.cmd_copy_buffer(
                            command,
                            buffer(source).handle,
                            buffer(target).handle,
                            &region,
                        )
                    };
                    resources.push(buffer(source).clone());
                    resources.push(buffer(target).clone());
                }
                Command::Clear {
                    buffer: target,
                    offset,
                    bytes,
                } => {
                    unsafe {
                        self.raw
                            .cmd_fill_buffer(command, buffer(target).handle, *offset, *bytes, 0)
                    };
                    resources.push(buffer(target).clone());
                }
                Command::Dispatch {
                    pipeline: handle,
                    group: bindings,
                    offsets,
                    groups,
                } => {
                    let compiled = pipeline(&handle.slot.native);
                    let group = group(&bindings.native);
                    let pipeline_handle = *compiled
                        .compiled
                        .get()
                        .expect("a compute pipeline is compiled before dispatch");
                    unsafe {
                        self.raw.cmd_bind_pipeline(
                            command,
                            vk::PipelineBindPoint::COMPUTE,
                            pipeline_handle,
                        );
                        self.raw.cmd_bind_descriptor_sets(
                            command,
                            vk::PipelineBindPoint::COMPUTE,
                            compiled.layout,
                            0,
                            &[group.set],
                            offsets,
                        );
                        self.raw
                            .cmd_dispatch(command, groups[0], groups[1], groups[2]);
                    }
                    resources.push(compiled.clone());
                    resources.push(group.clone());
                }
            }
        }
        unsafe { self.raw.end_command_buffer(command) }
            .unwrap_or_else(|error| panic!("closing a Vulkan compute submission: {error:?}"));
        let fence = unsafe { self.raw.create_fence(&vk::FenceCreateInfo::default(), None) }
            .unwrap_or_else(|error| panic!("creating a Vulkan completion fence: {error:?}"));
        let buffers = [command];
        let submits = [vk::SubmitInfo::default().command_buffers(&buffers)];
        unsafe { self.raw.queue_submit(self.queue, &submits, fence) }
            .unwrap_or_else(|error| panic!("submitting Vulkan compute work: {error:?}"));
        state.next = state
            .next
            .checked_add(1)
            .expect("compute submission indices fit in u64");
        let index = state.next;
        state.in_flight.push_back(InFlight {
            index,
            command,
            fence,
            _resources: resources,
        });
        index
    }

    fn retire(&self, state: &mut QueueState) {
        while let Some(front) = state.in_flight.front() {
            if !unsafe { self.raw.get_fence_status(front.fence) }
                .unwrap_or_else(|error| panic!("checking Vulkan compute completion: {error:?}"))
            {
                break;
            }
            let finished = state
                .in_flight
                .pop_front()
                .expect("a completed fence was queued");
            unsafe {
                self.raw
                    .free_command_buffers(self.pool, &[finished.command]);
                self.raw.destroy_fence(finished.fence, None);
            }
            state.completed = finished.index;
        }
    }

    pub(crate) fn wait(&self, index: u64, timeout: Duration) {
        let mut state = self
            .state
            .lock()
            .expect("the Vulkan queue is never poisoned");
        assert!(
            index <= state.next,
            "waiting on an unsubmitted Vulkan command"
        );
        if index <= state.completed {
            return;
        }
        let fence = state
            .in_flight
            .iter()
            .find(|item| item.index == index)
            .expect("an unfinished submission owns a fence")
            .fence;
        unsafe {
            self.raw.wait_for_fences(
                &[fence],
                true,
                timeout.as_nanos().min(u64::MAX as u128) as u64,
            )
        }
        .unwrap_or_else(|error| panic!("waiting for Vulkan compute work: {error:?}"));
        self.retire(&mut state);
        assert!(state.completed >= index, "Vulkan compute did not complete");
    }

    pub(crate) fn read(&self, buffer: &Buffer, bytes: u64) -> Vec<u8> {
        let mapped = unsafe {
            self.raw.map_memory(
                buffer.resource.memory,
                0,
                bytes,
                vk::MemoryMapFlags::empty(),
            )
        }
        .unwrap_or_else(|error| panic!("mapping a Vulkan readback: {error:?}"));
        let result =
            unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize) }.to_vec();
        unsafe { self.raw.unmap_memory(buffer.resource.memory) };
        result
    }

    pub(crate) fn assert_alive(&self) {
        let mut state = self
            .state
            .lock()
            .expect("the Vulkan queue is never poisoned");
        self.retire(&mut state);
    }
}

fn limits(
    properties: &vk::PhysicalDeviceProperties,
    memory: &vk::PhysicalDeviceMemoryProperties,
) -> Limits {
    let gpu = &properties.limits;
    let local_bytes = memory.memory_heaps[..memory.memory_heap_count as usize]
        .iter()
        .filter(|heap| heap.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
        .map(|heap| heap.size)
        .max()
        .unwrap_or(0);
    Limits {
        max_storage_buffers_per_shader_stage: gpu.max_per_stage_descriptor_storage_buffers.min(30),
        max_storage_buffer_binding_size: u64::from(gpu.max_storage_buffer_range),
        max_buffer_size: local_bytes.min(1 << 30),
        max_compute_invocations_per_workgroup: gpu.max_compute_work_group_invocations,
        max_compute_workgroup_size_x: gpu.max_compute_work_group_size[0],
        max_compute_workgroup_storage_size: gpu.max_compute_shared_memory_size,
        max_compute_workgroups_per_dimension: gpu
            .max_compute_work_group_count
            .iter()
            .copied()
            .min()
            .unwrap_or(0),
        min_storage_buffer_offset_alignment: gpu.min_storage_buffer_offset_alignment.max(4),
    }
}

impl Pipeline {
    pub(crate) fn is_compiled(&self) -> bool {
        self.resource.compiled.get().is_some()
    }

    pub(crate) fn compile(&self, program: &ComputeProgram) {
        self.resource.compiled.get_or_init(|| {
            let words = program.spirv();
            let module = unsafe {
                self.resource
                    .raw
                    .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
            }
            .unwrap_or_else(|error| {
                panic!(
                    "creating the Vulkan SPIR-V module of {}: {error:?}",
                    program.label()
                )
            });
            let entry =
                CString::new(program.entry()).expect("a compute entry point has no zero byte");
            let stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::COMPUTE)
                .module(module)
                .name(&entry);
            let pipelines = unsafe {
                self.resource.raw.create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .stage(stage)
                        .layout(self.resource.layout)],
                    None,
                )
            };
            unsafe { self.resource.raw.destroy_shader_module(module, None) };
            pipelines.unwrap_or_else(|(_, error)| {
                panic!(
                    "compiling Vulkan compute pipeline {}: {error:?}",
                    program.label()
                )
            })[0]
        });
    }
}

impl Drop for PipelineResource {
    fn drop(&mut self) {
        unsafe {
            if let Some(compiled) = self.compiled.get() {
                self.raw.destroy_pipeline(*compiled, None);
            }
            self.raw.destroy_pipeline_layout(self.layout, None);
            self.raw
                .destroy_descriptor_set_layout(self.group_layout, None);
        }
    }
}

impl Drop for GroupResource {
    fn drop(&mut self) {
        unsafe { self.raw.destroy_descriptor_pool(self.pool, None) };
    }
}

impl Drop for BufferResource {
    fn drop(&mut self) {
        unsafe {
            self.raw.destroy_buffer(self.handle, None);
            self.raw.free_memory(self.memory, None);
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        let mut state = std::mem::take(
            self.state
                .get_mut()
                .expect("the Vulkan queue is never poisoned"),
        );
        if let Some(last) = state.in_flight.back() {
            unsafe {
                self.raw
                    .wait_for_fences(&[last.fence], true, 30_000_000_000)
            }
            .unwrap_or_else(|error| panic!("draining Vulkan on drop: {error:?}"));
        }
        self.retire(&mut state);
        unsafe {
            self.raw.destroy_command_pool(self.pool, None);
            self.raw.destroy_device(None);
        }
    }
}
