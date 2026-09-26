use super::{
    DeviceFailure, FRAME_TIMEOUT, FRAMES_IN_FLIGHT, NativeBuffer, NativePipeline, STAGING_BYTES,
};
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
use std::mem::size_of;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

const RELEASE_TIMEOUT: Duration = Duration::from_secs(30);

struct Frame {
    index: u64,
    command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    staging: Arc<BufferResource>,
    cursor: u64,
    resources: Vec<Arc<dyn Any + Send + Sync>>,
}

unsafe impl Send for Frame {}

impl Frame {
    fn new(device: &Device) -> Self {
        Self {
            index: 0,
            command: None,
            staging: device.allocate(STAGING_BYTES, true),
            cursor: 0,
            resources: Vec::new(),
        }
    }

    fn begin(&mut self) {
        self.cursor = 0;
        self.resources.clear();
    }

    fn stage(&mut self, device: &Device, bytes: &[u8]) -> (Arc<BufferResource>, u64) {
        let size = bytes.len() as u64;
        let (staging, offset) = if size <= STAGING_BYTES - self.cursor {
            let offset = self.cursor;
            self.cursor += size;
            (self.staging.clone(), offset)
        } else {
            let staging = device.allocate(size, true);
            self.resources.push(staging.clone());
            (staging, 0)
        };
        unsafe {
            ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                staging
                    .raw
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(offset as usize),
                bytes.len(),
            );
        }
        (staging, offset)
    }
}

#[derive(Default)]
struct QueueState {
    next: u64,
    completed: u64,
    frames: Vec<Frame>,
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
        let slot = self.acquire(&mut state);
        state.next = state
            .next
            .checked_add(1)
            .expect("compute submission indices fit in u64");
        let index = state.next;
        let frame = &mut state.frames[slot];
        frame.begin();
        frame.index = index;
        let command = self
            .queue
            .commandBuffer()
            .expect("a Metal compute command buffer exists");
        for write in writes {
            let (upload, offset) = frame.stage(self, &write.bytes);
            let target = native_buffer(&write.buffer);
            let blit = self.blit(&command);
            unsafe {
                blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &upload.raw,
                    offset as usize,
                    &target.raw,
                    write.offset as usize,
                    write.bytes.len(),
                );
            }
            blit.endEncoding();
            frame.resources.push(target.clone());
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
                    frame.resources.push(source.clone());
                    frame.resources.push(target.clone());
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
                    frame.resources.push(target.clone());
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
                        frame.resources.push(target.clone());
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
                    frame.resources.push(pipeline.clone());
                }
            }
        }
        command.commit();
        frame.command = Some(command);
        index
    }

    #[cfg(test)]
    pub(crate) fn frames(&self) -> usize {
        self.state
            .lock()
            .expect("the Metal compute queue is never poisoned")
            .frames
            .len()
    }

    fn acquire(&self, state: &mut QueueState) -> usize {
        if let Some(slot) = state
            .frames
            .iter()
            .position(|frame| frame.index <= state.completed)
        {
            return slot;
        }
        if state.frames.len() < FRAMES_IN_FLIGHT {
            state.frames.push(Frame::new(self));
            return state.frames.len() - 1;
        }
        let slot = state
            .frames
            .iter()
            .enumerate()
            .min_by_key(|(_, frame)| frame.index)
            .map(|(slot, _)| slot)
            .expect("a queue that holds a frame recycles one");
        let index = state.frames[slot].index;
        self.await_completion(state, index, FRAME_TIMEOUT)
            .unwrap_or_else(|error| panic!("{error}"));
        slot
    }

    fn retire(&self, state: &mut QueueState) -> Result<(), String> {
        for frame in &state.frames {
            if frame.index <= state.completed {
                continue;
            }
            let Some(command) = &frame.command else {
                state.completed = frame.index;
                continue;
            };
            match command.status() {
                MTLCommandBufferStatus::Completed => {
                    state.completed = state.completed.max(frame.index);
                }
                MTLCommandBufferStatus::Error => {
                    return Err(format!("Metal compute failed: {:?}", command.error()));
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub(crate) fn wait(&self, index: u64, timeout: Duration) {
        let mut state = self
            .state
            .lock()
            .expect("the Metal compute queue is never poisoned");
        self.await_completion(&mut state, index, timeout)
            .unwrap_or_else(|error| panic!("{error}"));
    }

    fn await_completion(
        &self,
        state: &mut QueueState,
        index: u64,
        timeout: Duration,
    ) -> Result<(), String> {
        if index > state.next {
            return Err("waiting on an unsubmitted Metal command".to_owned());
        }
        self.retire(state)?;
        if index <= state.completed {
            return Ok(());
        }
        if !state.frames.iter().any(|frame| frame.index == index) {
            return Err("an unfinished Metal command is in flight".to_owned());
        }
        let started = Instant::now();
        loop {
            self.retire(state)?;
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
        let mut state = self
            .state
            .lock()
            .expect("the Metal compute queue is never poisoned");
        let index = state.next;
        let _ = self.await_completion(&mut state, index, RELEASE_TIMEOUT);
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
