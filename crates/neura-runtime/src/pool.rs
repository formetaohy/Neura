use neura_gpu::{BufferUsages, Device, GpuBuffer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub(crate) const POOL_BYTES: u64 = 16 << 20;

struct Held {
    entries: HashMap<(BufferUsages, u64), Vec<GpuBuffer>>,
    bytes: u64,
}

pub(crate) struct Pool {
    device: Device,
    ceiling: u64,
    held: Mutex<Held>,
}

impl Pool {
    pub(crate) fn of(device: &Device, ceiling: u64) -> Arc<Self> {
        Arc::new(Self {
            device: device.clone(),
            ceiling,
            held: Mutex::new(Held {
                entries: HashMap::new(),
                bytes: 0,
            }),
        })
    }

    pub(crate) fn acquire(&self, label: &str, bytes: u64, usage: BufferUsages) -> GpuBuffer {
        let mut held = self.held.lock().expect("a buffer pool is never poisoned");
        if let Some(buffer) = held
            .entries
            .get_mut(&(usage, bytes))
            .and_then(|slot| slot.pop())
        {
            held.bytes -= bytes;
            return buffer;
        }
        GpuBuffer::new(&self.device, label, bytes, usage)
    }

    pub(crate) fn release(&self, buffer: GpuBuffer) {
        let bytes = buffer.size();
        let mut held = self.held.lock().expect("a buffer pool is never poisoned");
        if held.bytes + bytes > self.ceiling {
            return;
        }
        held.bytes += bytes;
        held.entries
            .entry((buffer.usage(), bytes))
            .or_default()
            .push(buffer);
    }
}

pub(crate) struct Recycled {
    buffer: Option<GpuBuffer>,
    pool: Arc<Pool>,
}

impl Recycled {
    pub(crate) fn claim(pool: &Arc<Pool>, label: &str, bytes: u64, usage: BufferUsages) -> Self {
        Self {
            buffer: Some(pool.acquire(label, bytes, usage)),
            pool: pool.clone(),
        }
    }

    pub(crate) fn buffer(&self) -> &GpuBuffer {
        self.buffer
            .as_ref()
            .expect("a recycled buffer lives as long as its holder")
    }
}

impl Drop for Recycled {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}
