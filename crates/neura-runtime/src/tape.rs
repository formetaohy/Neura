use crate::pool::{Pool, Recycled};
use neura_abi::{Kind, StepRecord};
use neura_gpu::{BufferUsages, GpuContext, PipelineHandle};
use neura_precision::Precision;
use neura_profile::Geometry;
use neura_program::Encoding;
use neura_shader::Megakernel;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem::{size_of, size_of_val};
use std::sync::{Arc, Mutex, Weak};

pub(crate) struct DeviceTape {
    signature: Vec<u8>,
    pub(crate) encoding: Encoding,
    pub(crate) kernel: PipelineHandle,
    pub(crate) tasks: Recycled,
    pub(crate) values: Recycled,
    pub(crate) bounds: Recycled,
    pub(crate) steps: Recycled,
    pub(crate) segments: Recycled,
    pool: Arc<Pool>,
}

impl DeviceTape {
    pub(crate) fn build(
        context: &GpuContext,
        pool: &Arc<Pool>,
        encoding: Encoding,
        kernel: Arc<Megakernel>,
        signature: Vec<u8>,
    ) -> Arc<Self> {
        let limits = context.limits();
        for (name, bytes) in [
            ("tape", encoding.tasks().len() as u64),
            ("values", encoding.values().len() as u64),
        ] {
            assert!(
                bytes <= limits.max_storage_buffer_binding_size,
                "a tape of {} tasks binds {bytes} bytes of {name}, and the device binds at most {} bytes of storage",
                encoding.task_count(),
                limits.max_storage_buffer_binding_size,
            );
            assert!(
                bytes <= limits.max_buffer_size,
                "a tape of {} tasks binds {bytes} bytes of {name}, and the device holds buffers of at most {} bytes",
                encoding.task_count(),
                limits.max_buffer_size,
            );
        }
        let kernel = context.declare(kernel.program());
        let tasks_bytes = encoding.tasks().len() as u64;
        let values_bytes = encoding.values().len() as u64;
        let bounds_bytes = encoding.bounds().len() as u64;
        let steps_bytes = (encoding.steps().len() as u64).max(size_of::<StepRecord>() as u64);
        let segments_bytes = size_of_val(encoding.segments()) as u64;
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let tape = Self {
            signature,
            encoding,
            kernel,
            tasks: Recycled::claim(pool, "neura tape", tasks_bytes, storage),
            values: Recycled::claim(pool, "neura values", values_bytes, storage),
            bounds: Recycled::claim(pool, "neura bounds", bounds_bytes, storage),
            steps: Recycled::claim(pool, "neura steps", steps_bytes, storage),
            segments: Recycled::claim(pool, "neura segments", segments_bytes, storage),
            pool: pool.clone(),
        };
        let queue = context.queue();
        tape.tasks.buffer().write(queue, tape.encoding.tasks());
        tape.values.buffer().write(queue, tape.encoding.values());
        tape.bounds.buffer().write(queue, tape.encoding.bounds());
        if !tape.encoding.steps().is_empty() {
            tape.steps.buffer().write(queue, tape.encoding.steps());
        }
        tape.segments
            .buffer()
            .write(queue, bytemuck::cast_slice(tape.encoding.segments()));
        Arc::new(tape)
    }

    pub(crate) fn pool(&self) -> &Arc<Pool> {
        &self.pool
    }

    pub(crate) fn holds(&self, signature: &[u8]) -> bool {
        self.signature == signature
    }
}

#[derive(PartialEq, Eq, Hash)]
struct KernelIdentity {
    kinds: Vec<Kind>,
    geometry: Geometry,
    precision: Precision,
}

pub(crate) struct Tapes {
    kernels: Mutex<HashMap<KernelIdentity, Arc<Megakernel>>>,
    entries: Mutex<HashMap<u64, Vec<Weak<DeviceTape>>>>,
}

impl Tapes {
    pub(crate) fn new() -> Self {
        Self {
            kernels: Mutex::new(HashMap::new()),
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub(crate) fn kernels(&self) -> usize {
        self.kernels
            .lock()
            .expect("a kernel cache is never poisoned")
            .len()
    }

    pub(crate) fn kernel(
        &self,
        kinds: &[Kind],
        geometry: Geometry,
        precision: Precision,
        assemble: impl FnOnce() -> Megakernel,
    ) -> Arc<Megakernel> {
        let identity = KernelIdentity {
            kinds: kinds.to_vec(),
            geometry,
            precision,
        };
        let mut kernels = self
            .kernels
            .lock()
            .expect("a kernel cache is never poisoned");
        kernels
            .entry(identity)
            .or_insert_with(|| Arc::new(assemble()))
            .clone()
    }

    pub(crate) fn resident(&self) -> usize {
        let entries = self.entries.lock().expect("a tape cache is never poisoned");
        entries
            .values()
            .flatten()
            .filter(|tape| tape.strong_count() > 0)
            .count()
    }

    pub(crate) fn of(
        &self,
        signature: Vec<u8>,
        build: impl FnOnce(Vec<u8>) -> Arc<DeviceTape>,
    ) -> Arc<DeviceTape> {
        let key = {
            let mut hasher = DefaultHasher::new();
            signature.hash(&mut hasher);
            hasher.finish()
        };
        let mut entries = self.entries.lock().expect("a tape cache is never poisoned");
        let bucket = entries.entry(key).or_default();
        bucket.retain(|tape| tape.strong_count() > 0);
        if let Some(tape) = bucket.iter().find_map(|tape| {
            let tape = tape.upgrade()?;
            tape.holds(&signature).then_some(tape)
        }) {
            return tape;
        }
        let tape = build(signature);
        bucket.push(Arc::downgrade(&tape));
        tape
    }
}

pub(crate) fn signature(
    encoding: &Encoding,
    profile: neura_profile::Profile,
    precision: Precision,
    alignment: u64,
) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(profile.workgroup().to_le_bytes());
    for tile in encoding.tiles() {
        bytes.extend(tile.rows().to_le_bytes());
        bytes.extend(tile.columns().to_le_bytes());
        bytes.extend(tile.depth().to_le_bytes());
        bytes.extend(tile.thread_rows().to_le_bytes());
        bytes.extend(tile.thread_columns().to_le_bytes());
    }
    bytes.extend((precision as u32).to_le_bytes());
    bytes.extend(alignment.to_le_bytes());
    bytes.extend(encoding.tasks());
    bytes.extend(encoding.values());
    bytes.extend(encoding.bounds());
    bytes.extend(encoding.steps());
    bytes.extend(bytemuck::cast_slice(encoding.segments()));
    bytes
}
