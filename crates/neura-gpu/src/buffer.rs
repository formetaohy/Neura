use std::sync::atomic::{AtomicU64, Ordering};
use wgpu::{
    BindGroupEntry, BindingResource, Buffer, BufferAddress, BufferBinding, BufferDescriptor,
    BufferUsages, Device, Queue,
};

static NEXT_ALLOCATION: AtomicU64 = AtomicU64::new(1);

pub struct GpuBuffer {
    buffer: Buffer,
    size: BufferAddress,
    usage: BufferUsages,
    allocation: u64,
}

impl GpuBuffer {
    pub fn new(device: &Device, label: &str, size: BufferAddress, usage: BufferUsages) -> Self {
        assert!(size > 0, "a buffer of {label} must hold bytes");
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        Self {
            buffer,
            size,
            usage,
            allocation: NEXT_ALLOCATION.fetch_add(1, Ordering::Relaxed),
        }
    }

    pub fn allocation(&self) -> u64 {
        self.allocation
    }

    pub fn size(&self) -> BufferAddress {
        self.size
    }

    pub fn usage(&self) -> BufferUsages {
        self.usage
    }

    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    pub fn write(&self, queue: &Queue, bytes: &[u8]) {
        assert!(
            self.usage.contains(BufferUsages::COPY_DST),
            "writing {} bytes into a buffer that does not accept copies",
            bytes.len(),
        );
        assert!(
            bytes.len() as BufferAddress <= self.size,
            "writing {} bytes into a buffer of {} bytes",
            bytes.len(),
            self.size,
        );
        queue.write_buffer(&self.buffer, 0, bytes);
    }

    pub fn write_at(&self, queue: &Queue, offset: BufferAddress, bytes: &[u8]) {
        assert!(
            self.usage.contains(BufferUsages::COPY_DST),
            "writing {} bytes into a buffer that does not accept copies",
            bytes.len(),
        );
        assert!(
            offset + bytes.len() as BufferAddress <= self.size,
            "writing {} bytes at {offset} into a buffer of {} bytes",
            bytes.len(),
            self.size,
        );
        queue.write_buffer(&self.buffer, offset, bytes);
    }

    pub fn whole(&self) -> BindGroupEntry<'_> {
        BindGroupEntry {
            binding: 0,
            resource: self.resource(0, self.size),
        }
    }

    pub fn resource(&self, offset: BufferAddress, size: BufferAddress) -> BindingResource<'_> {
        assert!(
            offset.is_multiple_of(4),
            "a binding offset must be word aligned"
        );
        assert!(
            size > 0 && size.is_multiple_of(4) && offset + size <= self.size,
            "a binding of {size} bytes at {offset} leaves the buffer of {} bytes",
            self.size,
        );
        BindingResource::Buffer(BufferBinding {
            buffer: &self.buffer,
            offset,
            size: Some(core::num::NonZeroU64::new(size).expect("a binding size is positive")),
        })
    }
}
