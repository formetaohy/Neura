use super::{NativeBuffer, NativePipeline};
use crate::buffer::GpuBuffer;
use crate::capability::{
    AdapterId, AdapterInfo, Backend, BufferUsages, DeviceType, Limits, PowerPreference,
};
use crate::pipeline::{BindingKind, ComputeProgram, ShaderTranslation};
use crate::submission::{Command, Write};
use libloading::Library;
use std::any::Any;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::mem::{ManuallyDrop, size_of};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::Dxc::*;
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_11_0, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC};
use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, DXGI_ADAPTER_DESC1, IDXGIFactory1};
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{GUID, Interface, PCWSTR};

struct CompletionEvent(HANDLE);

unsafe impl Send for CompletionEvent {}
unsafe impl Sync for CompletionEvent {}

impl Drop for CompletionEvent {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.0) }.expect("a compute completion event closes");
    }
}

struct InFlight {
    index: u64,
    _allocator: ID3D12CommandAllocator,
    _list: ID3D12GraphicsCommandList,
    _resources: Vec<Arc<dyn Any + Send + Sync>>,
}

#[derive(Default)]
struct QueueState {
    next: u64,
    completed: u64,
    in_flight: VecDeque<InFlight>,
}

pub(crate) struct Device {
    raw: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    event: CompletionEvent,
    zero: Arc<BufferResource>,
    state: Mutex<QueueState>,
}

struct BufferResource {
    raw: ID3D12Resource,
    state: Mutex<D3D12_RESOURCE_STATES>,
}

#[derive(Clone)]
pub(crate) struct Buffer {
    resource: Arc<BufferResource>,
}

struct PipelineResource {
    root: ID3D12RootSignature,
    compiled: OnceLock<ID3D12PipelineState>,
}

pub(crate) struct Pipeline {
    device: ID3D12Device,
    resource: Arc<PipelineResource>,
}

fn native_buffer(native: &NativeBuffer) -> &Arc<BufferResource> {
    let NativeBuffer::Dx12(buffer) = native else {
        panic!("a D3D12 command cannot use another backend's buffer");
    };
    &buffer.resource
}

fn buffer(gpu: &GpuBuffer) -> &Arc<BufferResource> {
    native_buffer(gpu.native())
}

fn pipeline(native: &NativePipeline) -> &Arc<PipelineResource> {
    let NativePipeline::Dx12(pipeline) = native else {
        panic!("a D3D12 command cannot use another backend's pipeline");
    };
    &pipeline.resource
}

fn rank(ty: DeviceType, preference: PowerPreference) -> u8 {
    match (preference, ty) {
        (PowerPreference::HighPerformance, DeviceType::Discrete)
        | (PowerPreference::LowPower, DeviceType::Integrated) => 3,
        (PowerPreference::HighPerformance, DeviceType::Integrated)
        | (PowerPreference::LowPower, DeviceType::Discrete) => 2,
        _ => 1,
    }
}

fn adapter_type(desc: &DXGI_ADAPTER_DESC1) -> DeviceType {
    if desc.Flags & 2 != 0 {
        DeviceType::Cpu
    } else if desc.DedicatedVideoMemory > 0 {
        DeviceType::Discrete
    } else {
        DeviceType::Integrated
    }
}

impl Device {
    pub(crate) fn open(
        preference: PowerPreference,
    ) -> Result<(Arc<Self>, AdapterInfo, Limits), String> {
        let _compiler = unsafe { Library::new("dxcompiler.dll") }
            .map_err(|error| format!("loading the D3D12 compute compiler: {error}"))?;
        let factory: IDXGIFactory1 =
            unsafe { CreateDXGIFactory1() }.map_err(|error| format!("creating DXGI: {error}"))?;
        let mut adapters = Vec::new();
        for index in 0.. {
            let adapter = match unsafe { factory.EnumAdapters1(index) } {
                Ok(adapter) => adapter,
                Err(error)
                    if error.code() == windows::Win32::Graphics::Dxgi::DXGI_ERROR_NOT_FOUND =>
                {
                    break;
                }
                Err(error) => return Err(format!("enumerating DXGI adapters: {error}")),
            };
            let desc = unsafe { adapter.GetDesc1() }
                .map_err(|error| format!("querying a DXGI adapter: {error}"))?;
            let mut raw = None;
            if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut raw) }.is_err() {
                continue;
            }
            let raw: ID3D12Device = raw.expect("a successfully created D3D12 device exists");
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_0,
            };
            if unsafe {
                raw.CheckFeatureSupport(
                    D3D12_FEATURE_SHADER_MODEL,
                    (&raw mut model).cast(),
                    size_of::<D3D12_FEATURE_DATA_SHADER_MODEL>() as u32,
                )
            }
            .is_err()
                || model.HighestShaderModel.0 < D3D_SHADER_MODEL_6_0.0
            {
                continue;
            }
            adapters.push((adapter, desc, raw));
        }
        let (_, desc, raw) = adapters
            .into_iter()
            .max_by_key(|(_, desc, _)| rank(adapter_type(desc), preference))
            .ok_or_else(|| "no D3D12 adapter supports compute shader model 6.0".to_owned())?;
        let name = String::from_utf16_lossy(
            &desc.Description[..desc
                .Description
                .iter()
                .position(|unit| *unit == 0)
                .unwrap_or(desc.Description.len())],
        );
        let info = AdapterInfo {
            name,
            backend: Backend::Dx12,
            device_type: adapter_type(&desc),
            id: AdapterId::Numeric {
                vendor: desc.VendorId,
                device: desc.DeviceId,
            },
        };
        let queue: ID3D12CommandQueue = unsafe {
            raw.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: D3D12_COMMAND_LIST_TYPE_COMPUTE,
                ..Default::default()
            })
        }
        .map_err(|error| format!("creating a D3D12 compute queue: {error}"))?;
        let fence: ID3D12Fence = unsafe { raw.CreateFence(0, D3D12_FENCE_FLAG_NONE) }
            .map_err(|error| format!("creating a D3D12 completion fence: {error}"))?;
        let event = CompletionEvent(
            unsafe { CreateEventW(None, false, false, PCWSTR::null()) }
                .map_err(|error| format!("creating a D3D12 completion event: {error}"))?,
        );
        let zero = allocate(&raw, 64 << 10, D3D12_HEAP_TYPE_UPLOAD, false);
        let mut mapped = ptr::null_mut();
        unsafe {
            zero.raw.Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 0 }),
                Some(&raw mut mapped),
            )
        }
        .map_err(|error| format!("mapping the D3D12 zero buffer: {error}"))?;
        unsafe {
            ptr::write_bytes(mapped.cast::<u8>(), 0, 64 << 10);
            zero.raw.Unmap(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: 64 << 10,
                }),
            );
        }
        let limits = Limits {
            max_storage_buffers_per_shader_stage: 30,
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 1 << 30,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_storage_size: 32 << 10,
            max_compute_workgroups_per_dimension: 65_535,
            min_storage_buffer_offset_alignment: 16,
        };
        Ok((
            Arc::new(Self {
                raw,
                queue,
                fence,
                event,
                zero,
                state: Mutex::new(QueueState::default()),
            }),
            info,
            limits,
        ))
    }

    pub(crate) fn create_buffer(
        self: &Arc<Self>,
        _label: &str,
        size: u64,
        usage: BufferUsages,
    ) -> Buffer {
        Buffer {
            resource: allocate(
                &self.raw,
                size,
                if usage.contains(BufferUsages::MAP_READ) {
                    D3D12_HEAP_TYPE_READBACK
                } else {
                    D3D12_HEAP_TYPE_DEFAULT
                },
                usage.contains(BufferUsages::STORAGE),
            ),
        }
    }

    pub(crate) fn create_pipeline(self: &Arc<Self>, program: &ComputeProgram) -> Pipeline {
        let parameters = program
            .bindings()
            .iter()
            .map(|spec| D3D12_ROOT_PARAMETER {
                ParameterType: if spec.kind == BindingKind::ReadWriteStorage {
                    D3D12_ROOT_PARAMETER_TYPE_UAV
                } else {
                    D3D12_ROOT_PARAMETER_TYPE_SRV
                },
                Anonymous: D3D12_ROOT_PARAMETER_0 {
                    Descriptor: D3D12_ROOT_DESCRIPTOR {
                        ShaderRegister: spec.binding,
                        RegisterSpace: 0,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
            })
            .collect::<Vec<_>>();
        let desc = D3D12_ROOT_SIGNATURE_DESC {
            NumParameters: parameters.len() as u32,
            pParameters: parameters.as_ptr(),
            Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
            ..Default::default()
        };
        let mut blob: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        if let Err(error) = unsafe {
            D3D12SerializeRootSignature(
                &desc,
                D3D_ROOT_SIGNATURE_VERSION_1,
                &mut blob,
                Some(&mut errors),
            )
        } {
            let message = errors
                .as_ref()
                .map(|blob| unsafe {
                    String::from_utf8_lossy(std::slice::from_raw_parts(
                        blob.GetBufferPointer().cast::<u8>(),
                        blob.GetBufferSize(),
                    ))
                    .into_owned()
                })
                .unwrap_or_else(|| error.to_string());
            panic!(
                "serializing {}'s D3D12 root signature: {message}",
                program.label()
            );
        }
        let blob = blob.expect("root signature serialization produced data");
        let bytes = unsafe {
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize())
        };
        let root: ID3D12RootSignature = unsafe { self.raw.CreateRootSignature(0, bytes) }
            .unwrap_or_else(|error| {
                panic!(
                    "creating {}'s D3D12 root signature: {error}",
                    program.label()
                )
            });
        Pipeline {
            device: self.raw.clone(),
            resource: Arc::new(PipelineResource {
                root,
                compiled: OnceLock::new(),
            }),
        }
    }

    fn transition(
        &self,
        list: &ID3D12GraphicsCommandList,
        buffer: &BufferResource,
        next: D3D12_RESOURCE_STATES,
    ) {
        let mut state = buffer
            .state
            .lock()
            .expect("a D3D12 resource state is never poisoned");
        if *state == next
            || *state == D3D12_RESOURCE_STATE_GENERIC_READ
                && next == D3D12_RESOURCE_STATE_COPY_SOURCE
        {
            return;
        }
        let mut barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: ManuallyDrop::new(Some(buffer.raw.clone())),
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    StateBefore: *state,
                    StateAfter: next,
                }),
            },
        };
        unsafe {
            list.ResourceBarrier(&[barrier.clone()]);
            ManuallyDrop::drop(&mut (*barrier.Anonymous.Transition).pResource);
        }
        *state = next;
    }

    fn uav_barrier(&self, list: &ID3D12GraphicsCommandList) {
        let barrier = D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                UAV: ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                    pResource: ManuallyDrop::new(None),
                }),
            },
        };
        unsafe { list.ResourceBarrier(&[barrier]) };
    }

    pub(crate) fn submit(&self, writes: &[Write], commands: &[Command]) -> u64 {
        let mut state = self
            .state
            .lock()
            .expect("the D3D12 compute queue is never poisoned");
        self.retire(&mut state);
        if writes.is_empty() && commands.is_empty() {
            return state.next;
        }
        let allocator: ID3D12CommandAllocator = unsafe {
            self.raw
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE)
        }
        .unwrap_or_else(|error| panic!("allocating a D3D12 compute command list: {error}"));
        let list: ID3D12GraphicsCommandList = unsafe {
            self.raw
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, &allocator, None)
        }
        .unwrap_or_else(|error| panic!("creating a D3D12 compute command list: {error}"));
        let mut resources: Vec<Arc<dyn Any + Send + Sync>> = Vec::new();
        for write in writes {
            let upload = allocate(
                &self.raw,
                write.bytes.len() as u64,
                D3D12_HEAP_TYPE_UPLOAD,
                false,
            );
            let mut mapped = ptr::null_mut();
            unsafe {
                upload.raw.Map(
                    0,
                    Some(&D3D12_RANGE { Begin: 0, End: 0 }),
                    Some(&raw mut mapped),
                )
            }
            .unwrap_or_else(|error| panic!("mapping a D3D12 upload: {error}"));
            unsafe {
                ptr::copy_nonoverlapping(
                    write.bytes.as_ptr(),
                    mapped.cast::<u8>(),
                    write.bytes.len(),
                );
                upload.raw.Unmap(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: 0,
                        End: write.bytes.len(),
                    }),
                );
            }
            let target = native_buffer(&write.buffer);
            self.transition(&list, target, D3D12_RESOURCE_STATE_COPY_DEST);
            unsafe {
                list.CopyBufferRegion(
                    &target.raw,
                    write.offset,
                    &upload.raw,
                    0,
                    write.bytes.len() as u64,
                )
            };
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
                    self.transition(&list, source, D3D12_RESOURCE_STATE_COPY_SOURCE);
                    self.transition(&list, target, D3D12_RESOURCE_STATE_COPY_DEST);
                    unsafe {
                        list.CopyBufferRegion(
                            &target.raw,
                            *target_offset,
                            &source.raw,
                            *source_offset,
                            *bytes,
                        )
                    };
                    resources.push(source.clone());
                    resources.push(target.clone());
                }
                Command::Clear {
                    buffer: target,
                    offset,
                    bytes,
                } => {
                    let target = buffer(target);
                    self.transition(&list, target, D3D12_RESOURCE_STATE_COPY_DEST);
                    let mut written = 0;
                    while written < *bytes {
                        let chunk = (64 << 10).min(bytes - written);
                        unsafe {
                            list.CopyBufferRegion(
                                &target.raw,
                                offset + written,
                                &self.zero.raw,
                                0,
                                chunk,
                            )
                        };
                        written += chunk;
                    }
                    resources.push(target.clone());
                }
                Command::Dispatch {
                    pipeline: handle,
                    group,
                    offsets,
                    groups,
                } => {
                    let pipeline = pipeline(&handle.slot.native);
                    unsafe {
                        list.SetComputeRootSignature(&pipeline.root);
                        list.SetPipelineState(
                            pipeline
                                .compiled
                                .get()
                                .expect("a D3D12 pipeline is compiled before dispatch"),
                        );
                    }
                    let mut dynamic = offsets.iter();
                    for (index, binding) in group.buffers.iter().enumerate() {
                        let target = buffer(&binding.buffer);
                        let kind = handle.slot.program.bindings()[index].kind;
                        let state = if kind == BindingKind::ReadWriteStorage {
                            D3D12_RESOURCE_STATE_UNORDERED_ACCESS
                        } else {
                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
                        };
                        self.transition(&list, target, state);
                        let offset = binding.offset
                            + if binding.dynamic {
                                u64::from(*dynamic.next().expect("one dynamic offset"))
                            } else {
                                0
                            };
                        let address = unsafe { target.raw.GetGPUVirtualAddress() } + offset;
                        unsafe {
                            if kind == BindingKind::ReadWriteStorage {
                                list.SetComputeRootUnorderedAccessView(index as u32, address);
                            } else {
                                list.SetComputeRootShaderResourceView(index as u32, address);
                            }
                        }
                        resources.push(target.clone());
                    }
                    unsafe { list.Dispatch(groups[0], groups[1], groups[2]) };
                    self.uav_barrier(&list);
                    resources.push(pipeline.clone());
                }
            }
        }
        unsafe { list.Close() }
            .unwrap_or_else(|error| panic!("closing a D3D12 compute command list: {error}"));
        let command: ID3D12CommandList = list
            .cast()
            .expect("a compute command list is a D3D12 command list");
        unsafe { self.queue.ExecuteCommandLists(&[Some(command)]) };
        state.next = state
            .next
            .checked_add(1)
            .expect("compute submission indices fit in u64");
        unsafe { self.queue.Signal(&self.fence, state.next) }
            .unwrap_or_else(|error| panic!("signaling D3D12 compute completion: {error}"));
        let index = state.next;
        state.in_flight.push_back(InFlight {
            index,
            _allocator: allocator,
            _list: list,
            _resources: resources,
        });
        index
    }

    fn retire(&self, state: &mut QueueState) {
        let completed = unsafe { self.fence.GetCompletedValue() };
        if completed == u64::MAX {
            self.assert_device();
            panic!("D3D12 compute fence failed");
        }
        while state
            .in_flight
            .front()
            .is_some_and(|item| item.index <= completed)
        {
            state.completed = state
                .in_flight
                .pop_front()
                .expect("a finished command was queued")
                .index;
        }
    }

    pub(crate) fn wait(&self, index: u64, timeout: Duration) {
        let mut state = self
            .state
            .lock()
            .expect("the D3D12 compute queue is never poisoned");
        assert!(
            index <= state.next,
            "waiting on an unsubmitted D3D12 command"
        );
        self.retire(&mut state);
        if index <= state.completed {
            return;
        }
        unsafe { self.fence.SetEventOnCompletion(index, self.event.0) }
            .unwrap_or_else(|error| panic!("arming the D3D12 compute fence: {error}"));
        let result = unsafe {
            WaitForSingleObject(
                self.event.0,
                timeout.as_millis().min(u128::from(u32::MAX - 1)) as u32,
            )
        };
        assert_eq!(
            result, WAIT_OBJECT_0,
            "waiting for D3D12 compute work failed or timed out: {result:?}"
        );
        self.assert_device();
        self.retire(&mut state);
        assert!(state.completed >= index, "D3D12 compute did not complete");
    }

    pub(crate) fn read(&self, buffer: &Buffer, bytes: u64) -> Vec<u8> {
        let mut mapped = ptr::null_mut();
        unsafe {
            buffer.resource.raw.Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: bytes as usize,
                }),
                Some(&raw mut mapped),
            )
        }
        .unwrap_or_else(|error| panic!("mapping a D3D12 readback: {error}"));
        let result =
            unsafe { std::slice::from_raw_parts(mapped.cast::<u8>(), bytes as usize) }.to_vec();
        unsafe {
            buffer
                .resource
                .raw
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }))
        };
        result
    }

    fn assert_device(&self) {
        unsafe { self.raw.GetDeviceRemovedReason() }
            .unwrap_or_else(|error| panic!("the D3D12 compute device was lost: {error}"));
    }

    pub(crate) fn assert_alive(&self) {
        self.assert_device();
        let mut state = self
            .state
            .lock()
            .expect("the D3D12 compute queue is never poisoned");
        self.retire(&mut state);
    }
}

fn allocate(
    raw: &ID3D12Device,
    size: u64,
    heap: D3D12_HEAP_TYPE,
    storage: bool,
) -> Arc<BufferResource> {
    let properties = D3D12_HEAP_PROPERTIES {
        Type: heap,
        CreationNodeMask: 1,
        VisibleNodeMask: 1,
        ..Default::default()
    };
    let descriptor = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_UNKNOWN,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: if storage {
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
        } else {
            D3D12_RESOURCE_FLAG_NONE
        },
        ..Default::default()
    };
    let state = if heap == D3D12_HEAP_TYPE_UPLOAD {
        D3D12_RESOURCE_STATE_GENERIC_READ
    } else if heap == D3D12_HEAP_TYPE_READBACK {
        D3D12_RESOURCE_STATE_COPY_DEST
    } else {
        D3D12_RESOURCE_STATE_COMMON
    };
    let mut resource = None;
    unsafe {
        raw.CreateCommittedResource(
            &properties,
            D3D12_HEAP_FLAG_NONE,
            &descriptor,
            state,
            None,
            &mut resource,
        )
    }
    .unwrap_or_else(|error| panic!("allocating {size} D3D12 compute bytes: {error}"));
    Arc::new(BufferResource {
        raw: resource.expect("a committed buffer was created"),
        state: Mutex::new(state),
    })
}

impl Pipeline {
    pub(crate) fn is_compiled(&self) -> bool {
        self.resource.compiled.get().is_some()
    }

    pub(crate) fn compile(&self, program: &ComputeProgram) {
        self.resource.compiled.get_or_init(|| {
            let ShaderTranslation::Hlsl { source, entry } = program.translate(Backend::Dx12) else {
                panic!("D3D12 accepts HLSL compute programs");
            };
            let bytes = dxil(&source, &entry, program.label());
            let state = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: ManuallyDrop::new(Some(self.resource.root.clone())),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            };
            let pipeline = unsafe { self.device.CreateComputePipelineState(&state) }
                .unwrap_or_else(|error| {
                    panic!(
                        "creating the D3D12 compute pipeline of {}: {error}",
                        program.label()
                    )
                });
            let mut state = state;
            unsafe { ManuallyDrop::drop(&mut state.pRootSignature) };
            pipeline
        });
    }
}

fn dxil(source: &str, entry: &str, label: &str) -> Vec<u8> {
    let library = unsafe { Library::new("dxcompiler.dll") }
        .unwrap_or_else(|error| panic!("D3D12 needs dxcompiler.dll to compile {label}: {error}"));
    type Create = unsafe extern "system" fn(
        *const GUID,
        *const GUID,
        *mut *mut c_void,
    ) -> windows::core::HRESULT;
    let create: libloading::Symbol<'_, Create> = unsafe { library.get(b"DxcCreateInstance\0") }
        .unwrap_or_else(|error| panic!("finding DXC in dxcompiler.dll: {error}"));
    let mut raw = ptr::null_mut();
    unsafe { create(&CLSID_DxcCompiler, &IDxcCompiler3::IID, &raw mut raw) }
        .ok()
        .unwrap_or_else(|error| panic!("creating the D3D12 shader compiler: {error}"));
    let compiler = unsafe { IDxcCompiler3::from_raw(raw) };
    let arguments = [
        "-E",
        entry,
        "-T",
        "cs_6_0",
        "-O3",
        "-Wno-parentheses-equality",
        "-WX",
    ];
    let strings = arguments
        .iter()
        .map(|arg| arg.encode_utf16().chain([0]).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let arguments = strings
        .iter()
        .map(|arg| PCWSTR(arg.as_ptr()))
        .collect::<Vec<_>>();
    let result: IDxcResult = unsafe {
        compiler.Compile(
            &DxcBuffer {
                Ptr: source.as_ptr().cast(),
                Size: source.len(),
                Encoding: DXC_CP_UTF8.0,
            },
            Some(&arguments),
            None,
        )
    }
    .unwrap_or_else(|error| panic!("compiling the HLSL of {label}: {error}"));
    let status = unsafe { result.GetStatus() }.expect("DXC reports shader status");
    if let Err(error) = status.ok() {
        let mut errors: Option<IDxcBlobUtf8> = None;
        unsafe { result.GetOutput(DXC_OUT_ERRORS, ptr::null_mut(), &mut errors) }
            .expect("DXC supplies diagnostics");
        let message = errors
            .map(|blob| unsafe {
                std::str::from_utf8(std::slice::from_raw_parts(
                    blob.GetStringPointer().0.cast(),
                    blob.GetStringLength(),
                ))
                .expect("DXC diagnostics are UTF-8")
                .to_owned()
            })
            .unwrap_or_else(|| error.to_string());
        panic!("compiling the D3D12 compute shader {label}: {message}");
    }
    let mut object: Option<IDxcBlob> = None;
    unsafe { result.GetOutput(DXC_OUT_OBJECT, ptr::null_mut(), &mut object) }
        .expect("DXC supplies the compiled shader");
    let object = object.expect("the compiled D3D12 shader exists");
    unsafe {
        std::slice::from_raw_parts(
            object.GetBufferPointer().cast::<u8>(),
            object.GetBufferSize(),
        )
    }
    .to_vec()
}

impl Drop for Device {
    fn drop(&mut self) {
        let index = self
            .state
            .get_mut()
            .expect("the D3D12 compute queue is never poisoned")
            .next;
        self.wait(index, Duration::from_secs(30));
    }
}
