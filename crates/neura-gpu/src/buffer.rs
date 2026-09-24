use crate::capability::BufferUsages;
use crate::context::{Device, Queue};
use crate::native::NativeBuffer;
use crate::submission::{SubmissionIndex, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_ALLOCATION: AtomicU64 = AtomicU64::new(1);

struct BufferState {
    native: NativeBuffer,
    device: Device,
    size: u64,
    usage: BufferUsages,
    allocation: u64,
}

#[derive(Clone)]
pub struct GpuBuffer {
    inner: Arc<BufferState>,
}

#[derive(Clone, Copy)]
pub struct BufferBinding<'a> {
    pub(crate) buffer: &'a GpuBuffer,
    pub(crate) offset: u64,
    pub(crate) size: u64,
}

impl GpuBuffer {
    pub fn new(device: &Device, label: &str, size: u64, usage: BufferUsages) -> Self {
        assert!(size > 0, "a buffer of {label} must hold bytes");
        assert!(
            size <= device.limits().max_buffer_size,
            "a buffer of {size} bytes exceeds the device limit of {}",
            device.limits().max_buffer_size
        );
        assert!(!usage.is_empty(), "a buffer must have a purpose");
        assert!(
            !usage.contains(BufferUsages::MAP_READ)
                || (usage.contains(BufferUsages::COPY_DST)
                    && !usage.contains(BufferUsages::STORAGE)),
            "a host readback must accept copies and cannot be shader storage",
        );
        Self {
            inner: Arc::new(BufferState {
                native: device.native().create_buffer(label, size, usage),
                device: device.clone(),
                size,
                usage,
                allocation: NEXT_ALLOCATION.fetch_add(1, Ordering::Relaxed),
            }),
        }
    }

    pub fn allocation(&self) -> u64 {
        self.inner.allocation
    }

    pub fn size(&self) -> u64 {
        self.inner.size
    }

    pub fn usage(&self) -> BufferUsages {
        self.inner.usage
    }

    pub(crate) fn device(&self) -> &Device {
        &self.inner.device
    }

    pub(crate) fn native(&self) -> &NativeBuffer {
        &self.inner.native
    }

    pub fn write(&self, queue: &Queue, bytes: &[u8]) {
        self.write_at(queue, 0, bytes);
    }

    pub fn write_at(&self, queue: &Queue, offset: u64, bytes: &[u8]) {
        assert!(
            self.device().same(queue.device()),
            "a queue cannot write another device's buffer"
        );
        assert!(
            self.usage().contains(BufferUsages::COPY_DST),
            "the buffer does not accept host writes"
        );
        assert!(
            offset.is_multiple_of(4) && bytes.len().is_multiple_of(4),
            "a buffer write must cover whole words"
        );
        assert!(
            offset
                .checked_add(bytes.len() as u64)
                .is_some_and(|end| end <= self.size()),
            "writing {} bytes at {offset} exceeds a buffer of {} bytes",
            bytes.len(),
            self.size(),
        );
        if !bytes.is_empty() {
            queue.write(Write {
                buffer: self.native().clone(),
                offset,
                bytes: bytes.to_vec(),
            });
        }
    }

    pub fn binding(&self, offset: u64, size: u64) -> BufferBinding<'_> {
        assert!(offset.is_multiple_of(4) && size > 0 && size.is_multiple_of(4));
        assert!(
            offset
                .checked_add(size)
                .is_some_and(|end| end <= self.size())
        );
        BufferBinding {
            buffer: self,
            offset,
            size,
        }
    }

    pub fn read(&self, queue: &Queue, submission: SubmissionIndex, bytes: u64) -> Vec<u8> {
        assert!(
            self.device().same(queue.device()),
            "a queue cannot read another device's buffer"
        );
        assert!(
            self.usage().contains(BufferUsages::MAP_READ),
            "the buffer is not host readable"
        );
        assert!(
            bytes > 0 && bytes <= self.size(),
            "reading {bytes} bytes exceeds the readback buffer"
        );
        queue.read(self, submission, bytes)
    }
}
