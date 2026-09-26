use super::{DeviceFailure, NativeBuffer, NativePipeline};
use crate::buffer::GpuBuffer;
use crate::capability::{
    AdapterId, AdapterInfo, AdapterPolicy, Backend, BufferUsages, DeviceType, Limits,
};
use crate::pipeline::{ComputeProgram, ShaderTranslation};
use crate::submission::{Command, Write};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal as mtl;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
    MTLResource, MTLResourceOptions, MTLSize,
};
use std::any::Any;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const RELEASE_TIMEOUT: Duration = Duration::from_secs(30);

struct InFlight {
    index: u64,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    _resources: Vec<Arc<dyn Any + Send + Sync>>,
}

unsafe impl Send for InFlight {}

#[derive(Default)]
struct QueueState {
    next: u64,
    completed: u64,
    in_flight: VecDeque<InFlight>,
}

pub(crate) struct Device {
    raw: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    state: Mutex<QueueState>,
}

struct BufferResource {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for BufferResource {}
unsafe impl Sync for BufferResource {}

#[derive(Clone)]
pub(crate) struct Buffer {
    resource: Arc<BufferResource>,
}

struct PipelineResource {
    compiled: OnceLock<Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    threads: OnceLock<u32>,
    sizes: OnceLock<Vec<u32>>,
}

pub(crate) struct Pipeline {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    resource: Arc<PipelineResource>,
}

fn native_buffer(native: &NativeBuffer) -> &Arc<BufferResource> {
    let NativeBuffer::Metal(buffer) = native else {
        panic!("a Metal command cannot use another backend's buffer");
    };
    &buffer.resource
}

fn buffer(gpu: &GpuBuffer) -> &Arc<BufferResource> {
    native_buffer(gpu.native())
}

fn pipeline(native: &NativePipeline) -> &Arc<PipelineResource> {
    let NativePipeline::Metal(pipeline) = native else {
        panic!("a Metal command cannot use another backend's pipeline");
    };
    &pipeline.resource
}

fn device_type(device: &ProtocolObject<dyn MTLDevice>) -> DeviceType {
    if device.hasUnifiedMemory() {
        DeviceType::Integrated
    } else {
        DeviceType::Discrete
    }
}

fn describe(device: &ProtocolObject<dyn MTLDevice>) -> AdapterInfo {
    AdapterInfo {
        name: device.name().to_string(),
        backend: Backend::Metal,
        device_type: device_type(device),
        id: AdapterId::MetalRegistry(device.registryID()),
    }
}

impl Device {
    pub(crate) fn open(
        policy: AdapterPolicy,
    ) -> Result<(Arc<Self>, AdapterInfo, Limits), DeviceFailure> {
        let devices = mtl::MTLCopyAllDevices();
        let offered = devices
            .iter()
            .map(|device| describe(&device))
            .collect::<Vec<_>>();
        let mut candidates = devices
            .iter()
            .filter(|device| policy.wants(AdapterId::MetalRegistry(device.registryID())))
            .map(|device| (describe(&device), device))
            .collect::<Vec<_>>();
        if let AdapterPolicy::Power(preference) = policy {
            candidates.sort_by_key(|(info, _)| Reverse(info.device_type.rank(preference)));
        }
        let Some((info, raw)) = candidates.into_iter().next() else {
            return Err(match policy {
                AdapterPolicy::Identity(_) => DeviceFailure::missing(offered),
                AdapterPolicy::Power(_) => DeviceFailure::reason("no Metal compute device"),
            });
        };
        let limits = Limits {
            max_storage_buffers_per_shader_stage: u32::from(
                crate::pipeline::METAL_SIZE_BUFFER_SLOT,
            ),
            max_storage_buffer_binding_size: raw.maxBufferLength().min(1 << 30) as u64,
            max_buffer_size: raw.maxBufferLength().min(1 << 30) as u64,
            max_compute_invocations_per_workgroup: raw.maxThreadsPerThreadgroup().width as u32,
            max_compute_workgroup_size_x: raw.maxThreadsPerThreadgroup().width as u32,
            max_compute_workgroup_storage_size: raw.maxThreadgroupMemoryLength() as u32,
            max_compute_workgroups_per_dimension: 65_535,
            min_storage_buffer_offset_alignment: 256,
        };
        let queue = raw
            .newCommandQueue()
            .ok_or_else(|| DeviceFailure::reason("Metal refused a compute command queue"))?;
        Ok((
            Arc::new(Self {
                raw,
                queue,
                state: Mutex::new(QueueState::default()),
            }),
            info,
            limits,
        ))
    }

    fn allocate(&self, size: u64, shared: bool) -> Arc<BufferResource> {
        let options = if shared {
            MTLResourceOptions::StorageModeShared
        } else {
            MTLResourceOptions::StorageModePrivate
        } | MTLResourceOptions::HazardTrackingModeTracked;
        let raw = self
            .raw
            .newBufferWithLength_options(size as usize, options)
            .unwrap_or_else(|| panic!("allocating {size} Metal compute bytes failed"));
        Arc::new(BufferResource { raw })
    }

    pub(crate) fn create_buffer(
        self: &Arc<Self>,
        label: &str,
        size: u64,
        usage: BufferUsages,
    ) -> Buffer {
        let resource = self.allocate(size, usage.contains(BufferUsages::MAP_READ));
        resource.raw.setLabel(Some(&NSString::from_str(label)));
        Buffer { resource }
    }

    pub(crate) fn create_pipeline(self: &Arc<Self>, _program: &ComputeProgram) -> Pipeline {
        Pipeline {
            device: self.raw.clone(),
            resource: Arc::new(PipelineResource {
                compiled: OnceLock::new(),
                threads: OnceLock::new(),
                sizes: OnceLock::new(),
            }),
        }
    }

    fn blit(
        &self,
        command: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Retained<ProtocolObject<dyn MTLBlitCommandEncoder>> {
        command
            .blitCommandEncoder()
            .expect("a Metal command buffer can encode copies")
    }

    pub(crate) fn submit(&self, writes: &[Write], commands: &[Command]) -> u64 {
        let mut state = self
            .state
            .lock()
            .expect("the Metal compute queue is never poisoned");
        self.retire(&mut state)
            .unwrap_or_else(|error| panic!("{error}"));
        if writes.is_empty() && commands.is_empty() {
            return state.next;
        }
        let command = self
            .queue
            .commandBuffer()
            .expect("a Metal compute command buffer exists");
        let mut resources: Vec<Arc<dyn Any + Send + Sync>> = Vec::new();
        for write in writes {
            let upload = self.allocate(write.bytes.len() as u64, true);
            unsafe {
                ptr::copy_nonoverlapping(
                    write.bytes.as_ptr(),
                    upload.raw.contents().as_ptr().cast::<u8>(),
                    write.bytes.len(),
                );
            }
            let target = native_buffer(&write.buffer);
            let blit = self.blit(&command);
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &upload.raw,
                    0,
                    &target.raw,
                    write.offset as usize,
                    write.bytes.len(),
                );
            }
            blit.endEncoding();
            resources.push(upload);
            resources.push(target.clone());
        }
        for operation in commands {
            match operation {
                Command::Copy {
                    source,
                    source_offset,
                    target,
                    target_offset,
                    bytes,
                } => {
                    let source = buffer(source);
                    let target = buffer(target);
                    let blit = self.blit(&command);
                    unsafe {
                        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &source.raw,
                            *source_offset as usize,
                            &target.raw,
                            *target_offset as usize,
                            *bytes as usize,
                        );
                    }
                    blit.endEncoding();
                    resources.push(source.clone());
                    resources.push(target.clone());
                }
                Command::Clear {
                    buffer: target,
                    offset,
                    bytes,
                } => {
                    let target = buffer(target);
                    let blit = self.blit(&command);
                    blit.fillBuffer_range_value(
                        &target.raw,
                        NSRange {
                            location: *offset as usize,
                            length: *bytes as usize,
                        },
                        0,
                    );
                    blit.endEncoding();
                    resources.push(target.clone());
                }
                Command::Dispatch {
                    pipeline: handle,
                    group,
                    offsets,
                    groups,
                } => {
                    let pipeline = pipeline(&handle.slot.native);
                    let compiled = pipeline
                        .compiled
                        .get()
                        .expect("a Metal pipeline is compiled before dispatch");
                    let threads = *pipeline
                        .threads
                        .get()
                        .expect("a Metal workgroup size is known");
                    let compute = command
                        .computeCommandEncoder()
                        .expect("a Metal command buffer encodes compute");
                    compute.setComputePipelineState(compiled);
                    let mut dynamic = offsets.iter();
                    for (index, binding) in group.buffers.iter().enumerate() {
                        let target = buffer(&binding.buffer);
                        let offset = binding.offset
                            + if binding.dynamic {
                                u64::from(*dynamic.next().expect("one dynamic offset"))
                            } else {
                                0
                            };
                        unsafe {
                            compute.setBuffer_offset_atIndex(
                                Some(&target.raw),
                                offset as usize,
                                index,
                            )
                        };
                        resources.push(target.clone());
                    }
                    let sizes = pipeline
                        .sizes
                        .get()
                        .expect("Metal runtime array bindings are known")
                        .iter()
                        .map(|index| {
                            u32::try_from(group.buffers[*index as usize].size)
                                .expect("Metal binding bytes fit in u32")
                        })
                        .collect::<Vec<_>>();
                    if let Some(first) = sizes.first() {
                        unsafe {
                            compute.setBytes_length_atIndex(
                                std::ptr::NonNull::from(first).cast(),
                                sizes.len() * size_of::<u32>(),
                                usize::from(crate::pipeline::METAL_SIZE_BUFFER_SLOT),
                            )
                        };
                    }
                    compute.dispatchThreadgroups_threadsPerThreadgroup(
                        MTLSize {
                            width: groups[0] as usize,
                            height: groups[1] as usize,
                            depth: groups[2] as usize,
                        },
                        MTLSize {
                            width: threads as usize,
                            height: 1,
                            depth: 1,
                        },
                    );
                    compute.endEncoding();
                    resources.push(pipeline.clone());
                }
            }
        }
        command.commit();
        state.next = state
            .next
            .checked_add(1)
            .expect("compute submission indices fit in u64");
        let index = state.next;
        state.in_flight.push_back(InFlight {
            index,
            command,
            _resources: resources,
        });
        index
    }

    fn retire(&self, state: &mut QueueState) -> Result<(), String> {
        while let Some(front) = state.in_flight.front() {
            match front.command.status() {
                MTLCommandBufferStatus::Completed => {
                    state.completed = state
                        .in_flight
                        .pop_front()
                        .expect("a completed Metal command exists")
                        .index;
                }
                MTLCommandBufferStatus::Error => {
                    return Err(format!("Metal compute failed: {:?}", front.command.error()));
                }
                _ => break,
            }
        }
        Ok(())
    }

    pub(crate) fn wait(&self, index: u64, timeout: Duration) {
        self.await_completion(index, timeout)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    fn await_completion(&self, index: u64, timeout: Duration) -> Result<(), String> {
        let mut state = self
            .state
            .lock()
            .expect("the Metal compute queue is never poisoned");
        if index > state.next {
            return Err("waiting on an unsubmitted Metal command".to_owned());
        }
        if index <= state.completed {
            return Ok(());
        }
        if !state.in_flight.iter().any(|item| item.index == index) {
            return Err("an unfinished Metal command is in flight".to_owned());
        }
        let started = Instant::now();
        loop {
            self.retire(&mut state)?;
            if state.completed >= index {
                return Ok(());
            }
            if started.elapsed() >= timeout {
                return Err("waiting for Metal compute work timed out".to_owned());
            }
            std::thread::sleep(Duration::from_micros(250));
        }
    }

    fn release(&self) {
        let index = self
            .state
            .lock()
            .expect("the Metal compute queue is never poisoned")
            .next;
        let _ = self.await_completion(index, RELEASE_TIMEOUT);
    }

    pub(crate) fn read(&self, buffer: &Buffer, bytes: u64) -> Vec<u8> {
        unsafe {
            std::slice::from_raw_parts(
                buffer.resource.raw.contents().as_ptr().cast::<u8>(),
                bytes as usize,
            )
        }
        .to_vec()
    }

    pub(crate) fn assert_alive(&self) {
        let mut state = self
            .state
            .lock()
            .expect("the Metal compute queue is never poisoned");
        if let Err(error) = self.retire(&mut state) {
            panic!("{error}");
        }
    }
}

impl Pipeline {
    pub(crate) fn is_compiled(&self) -> bool {
        self.resource.compiled.get().is_some()
    }

    pub(crate) fn compile(&self, program: &ComputeProgram) {
        self.resource.compiled.get_or_init(|| {
            let ShaderTranslation::Msl {
                source,
                entry,
                size_bindings,
            } = program.translate(Backend::Metal)
            else {
                panic!("Metal accepts MSL compute programs");
            };
            let library = self
                .device
                .newLibraryWithSource_options_error(&NSString::from_str(&source), None)
                .unwrap_or_else(|error| {
                    panic!("compiling the MSL of {}: {error}", program.label())
                });
            let function = library
                .newFunctionWithName(&NSString::from_str(&entry))
                .unwrap_or_else(|| panic!("{} has no Metal entry named {entry}", program.label()));
            let pipeline = self
                .device
                .newComputePipelineStateWithFunction_error(&function)
                .unwrap_or_else(|error| {
                    panic!(
                        "creating the Metal pipeline of {}: {error}",
                        program.label()
                    )
                });
            let workgroup = program.workgroup_size();
            assert!(
                workgroup > 0 && workgroup as usize <= pipeline.maxTotalThreadsPerThreadgroup(),
                "a Metal compute pipeline cannot schedule its declared workgroup"
            );
            self.resource
                .threads
                .set(workgroup)
                .expect("a pipeline stores its workgroup once");
            self.resource
                .sizes
                .set(size_bindings)
                .expect("a pipeline stores its bindings once");
            pipeline
        });
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        self.release();
    }
}
