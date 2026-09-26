use crate::buffer::GpuBuffer;
use crate::capability::{AdapterId, AdapterInfo, AdapterPolicy, Backends, Limits, PowerPreference};
use crate::library::PipelineLibrary;
use crate::native::{self, NativeDevice};
use crate::pipeline::{ComputeProgram, PipelineHandle};
use crate::submission::{Command, SubmissionIndex, Write};
use std::fmt::{self, Display, Formatter};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LimitsPolicy {
    Minimum,
    Adapter,
}

pub struct GpuRequest {
    pub backends: Backends,
    pub adapter: AdapterPolicy,
    pub limits: LimitsPolicy,
}

impl Default for GpuRequest {
    fn default() -> Self {
        Self {
            backends: Backends::all(),
            adapter: AdapterPolicy::Power(PowerPreference::HighPerformance),
            limits: LimitsPolicy::Adapter,
        }
    }
}

impl GpuRequest {
    pub fn minimum_limits(self) -> Self {
        Self {
            limits: LimitsPolicy::Minimum,
            ..self
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuUnavailable {
    NoAdapter {
        reason: String,
    },
    AdapterMissing {
        wanted: AdapterId,
        offered: Vec<AdapterInfo>,
    },
    UnsupportedLimits {
        info: AdapterInfo,
        limits: Limits,
    },
}

impl Display for GpuUnavailable {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter { reason } => write!(out, "no native compute device: {reason}"),
            Self::AdapterMissing { wanted, offered } => {
                write!(out, "the requested adapter {wanted} is missing")?;
                if offered.is_empty() {
                    return write!(out, ", and no requested backend offers an adapter");
                }
                for adapter in offered {
                    write!(out, "; {adapter}")?;
                }
                Ok(())
            }
            Self::UnsupportedLimits { info, limits } => write!(
                out,
                "{} ({:?}) cannot provide the compute baseline: {limits:?}",
                info.name, info.backend
            ),
        }
    }
}

impl std::error::Error for GpuUnavailable {}

pub(crate) struct DeviceState {
    pub(crate) writes: Mutex<Vec<Write>>,
    pub(crate) native: NativeDevice,
    pub(crate) info: AdapterInfo,
    pub(crate) limits: Limits,
}

#[derive(Clone)]
pub struct Device {
    pub(crate) state: Arc<DeviceState>,
}

impl Device {
    pub fn open(request: &GpuRequest) -> Result<Self, GpuUnavailable> {
        let (native, info, limits) = native::open(request)?;
        if !limits.supports(&Limits::BASELINE) {
            return Err(GpuUnavailable::UnsupportedLimits { info, limits });
        }
        let limits = match request.limits {
            LimitsPolicy::Minimum => limits.minimum(),
            LimitsPolicy::Adapter => limits,
        };
        Ok(Self {
            state: Arc::new(DeviceState {
                writes: Mutex::new(Vec::new()),
                native,
                info,
                limits,
            }),
        })
    }

    pub fn adapter_info(&self) -> &AdapterInfo {
        &self.state.info
    }

    pub fn limits(&self) -> &Limits {
        &self.state.limits
    }

    pub(crate) fn native(&self) -> &NativeDevice {
        &self.state.native
    }

    pub(crate) fn same(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.state, &other.state)
    }
}

#[derive(Clone)]
pub struct Queue {
    device: Device,
}

impl Queue {
    pub fn of(device: &Device) -> Self {
        Self {
            device: device.clone(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub(crate) fn write(&self, write: Write) {
        self.device
            .state
            .writes
            .lock()
            .expect("the compute queue is never poisoned")
            .push(write);
    }

    pub(crate) fn submit(&self, commands: &[Command]) -> SubmissionIndex {
        let mut writes = self
            .device
            .state
            .writes
            .lock()
            .expect("the compute queue is never poisoned");
        let index = self.device.native().submit(&writes, commands);
        writes.clear();
        SubmissionIndex(index)
    }

    pub fn wait(&self, submission: SubmissionIndex, timeout: Duration) {
        self.device.native().wait(submission.0, timeout);
    }

    pub(crate) fn read(
        &self,
        buffer: &GpuBuffer,
        submission: SubmissionIndex,
        bytes: u64,
    ) -> Vec<u8> {
        let _pending = self
            .device
            .state
            .writes
            .lock()
            .expect("the compute queue is never poisoned");
        self.device
            .native()
            .wait(submission.0, crate::readback::READBACK_TIMEOUT);
        self.device.native().read(buffer.native(), bytes)
    }

    pub fn drain(&self) {
        let index = self.submit(&[]);
        self.wait(index, crate::readback::READBACK_TIMEOUT);
    }
}

pub struct GpuContext {
    device: Device,
    queue: Queue,
    pipelines: Mutex<PipelineLibrary>,
}

impl GpuContext {
    pub const MINIMUM_LIMITS: Limits = Limits::BASELINE;

    pub async fn open(request: &GpuRequest) -> Result<Self, GpuUnavailable> {
        Device::open(request).map(Self::of_device)
    }

    pub fn of_device(device: Device) -> Self {
        Self {
            queue: Queue::of(&device),
            device,
            pipelines: Mutex::new(PipelineLibrary::new()),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn adapter_info(&self) -> &AdapterInfo {
        self.device.adapter_info()
    }

    pub fn limits(&self) -> &Limits {
        self.device.limits()
    }

    pub fn binding_alignment(&self) -> u64 {
        self.limits().min_storage_buffer_offset_alignment.max(16)
    }

    pub fn assert_alive(&self) {
        self.device.native().assert_alive();
    }

    pub fn drain(&self) {
        self.assert_alive();
        self.queue.drain();
        self.assert_alive();
    }

    pub fn declare(&self, program: ComputeProgram) -> PipelineHandle {
        self.assert_alive();
        self.pipelines
            .lock()
            .expect("a pipeline library is never poisoned")
            .declare(&self.device, program)
    }

    pub fn declared_kernels(&self) -> usize {
        self.pipelines
            .lock()
            .expect("a pipeline library is never poisoned")
            .declared()
    }
}
