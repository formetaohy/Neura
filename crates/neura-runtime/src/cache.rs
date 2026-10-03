use crate::pool::{Pool, Recycled};
use neura_abi::{Element, Kind, StepRecord};
use neura_gpu::{BufferUsages, GpuContext, PipelineHandle};
use neura_graph::GraphStamp;
use neura_kernel::Kernel;
use neura_plan::{Plan, Product};
use neura_profile::Geometry;
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem::{size_of, size_of_val};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

pub(crate) struct Resident {
    signature: Vec<u8>,
    pub(crate) plan: Arc<Plan>,
    pub(crate) kernel: PipelineHandle,
    pub(crate) steps: Recycled,
    pub(crate) segments: Recycled,
    pub(crate) measures: Option<Recycled>,
    pub(crate) patches: Option<Recycled>,
    pub(crate) patch_list: Option<Recycled>,
    pool: Arc<Pool>,
}

impl Resident {
    pub(crate) fn build(
        context: &GpuContext,
        pool: &Arc<Pool>,
        plan: Arc<Plan>,
        kernel: Arc<Kernel>,
        signature: Vec<u8>,
    ) -> Arc<Self> {
        let limits = context.limits();
        for (name, bytes) in [
            ("tasks", plan.tasks().len() as u64),
            ("values", plan.values().len() as u64),
        ] {
            assert!(
                bytes <= limits.max_storage_buffer_binding_size,
                "a plan of {} tasks binds {bytes} bytes of {name}, and the device binds at most {} bytes of storage",
                plan.task_count(),
                limits.max_storage_buffer_binding_size,
            );
            assert!(
                bytes <= limits.max_buffer_size,
                "a plan of {} tasks binds {bytes} bytes of {name}, and the device holds buffers of at most {} bytes",
                plan.task_count(),
                limits.max_buffer_size,
            );
        }
        let kernel = context.declare(kernel.program());
        let steps_bytes = (plan.steps().len() as u64).max(size_of::<StepRecord>() as u64);
        let segments_bytes = size_of_val(plan.segments()) as u64;
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let carries = plan.carries_authored();
        let owned = |label: &str, bytes: usize| {
            carries.then(|| Recycled::claim(pool, label, (bytes as u64).max(4), storage))
        };
        let measures = owned("neura measures", std::mem::size_of_val(plan.measures()));
        let patches = owned("neura patches", std::mem::size_of_val(plan.patches()));
        let patch_list = owned("neura patch list", std::mem::size_of_val(plan.patch_list()));
        let resident = Self {
            signature,
            plan,
            kernel,
            steps: Recycled::claim(pool, "neura steps", steps_bytes, storage),
            segments: Recycled::claim(pool, "neura segments", segments_bytes, storage),
            measures,
            patches,
            patch_list,
            pool: pool.clone(),
        };
        let queue = context.queue();
        if !resident.plan.steps().is_empty() {
            resident.steps.buffer().write(queue, resident.plan.steps());
        }
        resident
            .segments
            .buffer()
            .write(queue, bytemuck::cast_slice(resident.plan.segments()));
        if let (Some(measures), Some(patches), Some(patch_list)) = (
            resident.measures.as_ref(),
            resident.patches.as_ref(),
            resident.patch_list.as_ref(),
        ) {
            measures
                .buffer()
                .write(queue, bytemuck::cast_slice(resident.plan.measures()));
            patches
                .buffer()
                .write(queue, bytemuck::cast_slice(resident.plan.patches()));
            patch_list
                .buffer()
                .write(queue, bytemuck::cast_slice(resident.plan.patch_list()));
        }
        Arc::new(resident)
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
    elements: Vec<Element>,
    geometry: Geometry,
    authored: bool,
}

#[derive(Clone, PartialEq, Eq, Hash)]
struct PlanIdentity {
    stamp: GraphStamp,
    profile: neura_profile::Profile,
    alignment: u64,
    chosen: Vec<(Product, neura_profile::MatmulTile)>,
}

const ASSEMBLY_CEILING: usize = 8;

pub(crate) struct Assembly {
    pub(crate) signature: Vec<u8>,
    pub(crate) plan: Arc<Plan>,
    pub(crate) kernel: Arc<Kernel>,
}

pub(crate) struct Artifacts {
    kernels: Mutex<HashMap<KernelIdentity, Arc<Kernel>>>,
    assemblies: Mutex<Vec<(PlanIdentity, Arc<Assembly>)>>,
    residents: Mutex<HashMap<u64, Vec<Weak<Resident>>>>,
    built: AtomicUsize,
}

impl Artifacts {
    pub(crate) fn new() -> Self {
        Self {
            kernels: Mutex::new(HashMap::new()),
            assemblies: Mutex::new(Vec::new()),
            residents: Mutex::new(HashMap::new()),
            built: AtomicUsize::new(0),
        }
    }

    pub(crate) fn built(&self) -> usize {
        self.built.load(Ordering::Relaxed)
    }

    pub(crate) fn assemble(
        &self,
        stamp: GraphStamp,
        profile: neura_profile::Profile,
        alignment: u64,
        chosen: &[(Product, neura_profile::MatmulTile)],
        build: impl FnOnce() -> Assembly,
    ) -> Arc<Assembly> {
        let identity = PlanIdentity {
            stamp,
            profile,
            alignment,
            chosen: chosen.to_vec(),
        };
        let mut assemblies = self
            .assemblies
            .lock()
            .expect("an assembly cache is never poisoned");
        if let Some(assembly) = take_assembly(&mut assemblies, &identity) {
            keep_assembly(&mut assemblies, identity, assembly.clone());
            return assembly;
        }
        drop(assemblies);
        let assembly = Arc::new(build());
        self.built.fetch_add(1, Ordering::Relaxed);
        let mut assemblies = self
            .assemblies
            .lock()
            .expect("an assembly cache is never poisoned");
        keep_assembly(&mut assemblies, identity, assembly.clone());
        assembly
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
        elements: &[Element],
        geometry: Geometry,
        authored: bool,
        assemble: impl FnOnce() -> Kernel,
    ) -> Arc<Kernel> {
        let identity = KernelIdentity {
            kinds: kinds.to_vec(),
            elements: elements.to_vec(),
            geometry,
            authored,
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

    pub(crate) fn resident_plans(&self) -> usize {
        let residents = self
            .residents
            .lock()
            .expect("a resident cache is never poisoned");
        residents
            .values()
            .flatten()
            .filter(|resident| resident.strong_count() > 0)
            .count()
    }

    pub(crate) fn resident(
        &self,
        signature: Vec<u8>,
        build: impl FnOnce(Vec<u8>) -> Arc<Resident>,
    ) -> Arc<Resident> {
        let key = {
            let mut hasher = DefaultHasher::new();
            signature.hash(&mut hasher);
            hasher.finish()
        };
        let mut residents = self
            .residents
            .lock()
            .expect("a resident cache is never poisoned");
        let bucket = residents.entry(key).or_default();
        bucket.retain(|resident| resident.strong_count() > 0);
        if let Some(resident) = bucket.iter().find_map(|resident| {
            let resident = resident.upgrade()?;
            resident.holds(&signature).then_some(resident)
        }) {
            return resident;
        }
        let resident = build(signature);
        bucket.push(Arc::downgrade(&resident));
        resident
    }
}

fn take_assembly(
    assemblies: &mut Vec<(PlanIdentity, Arc<Assembly>)>,
    identity: &PlanIdentity,
) -> Option<Arc<Assembly>> {
    let found = assemblies.iter().position(|(kept, _)| kept == identity)?;
    Some(assemblies.remove(found).1)
}

fn keep_assembly(
    assemblies: &mut Vec<(PlanIdentity, Arc<Assembly>)>,
    identity: PlanIdentity,
    assembly: Arc<Assembly>,
) {
    let _ = take_assembly(assemblies, &identity);
    assemblies.push((identity, assembly));
    while assemblies.len() > ASSEMBLY_CEILING {
        assemblies.remove(0);
    }
}

pub(crate) fn signature(plan: &Plan, profile: neura_profile::Profile, alignment: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(profile.workgroup().to_le_bytes());
    for (index, tile) in plan.walked_tiles() {
        bytes.extend(index.to_le_bytes());
        bytes.extend(tile.rows().to_le_bytes());
        bytes.extend(tile.columns().to_le_bytes());
        bytes.extend(tile.depth().to_le_bytes());
        bytes.extend(tile.thread_rows().to_le_bytes());
        bytes.extend(tile.thread_columns().to_le_bytes());
    }
    for tile in plan.attention() {
        bytes.extend(tile.keys().to_le_bytes());
        bytes.extend(tile.width().to_le_bytes());
    }
    bytes.extend(alignment.to_le_bytes());
    bytes.extend(plan.tasks());
    bytes.extend(plan.values());
    bytes.extend(plan.steps());
    bytes.extend(bytemuck::cast_slice(plan.segments()));
    bytes.extend(bytemuck::cast_slice(plan.measures()));
    bytes.extend(bytemuck::cast_slice(plan.patches()));
    bytes.extend(bytemuck::cast_slice(plan.patch_list()));
    bytes
}
