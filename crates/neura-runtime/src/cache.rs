use crate::pool::{Pool, Recycled};
use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;
use neura_abi::{Digest, Element, Kind, StepRecord};
use neura_gpu::{AdapterInfo, BufferUsages, GpuContext, PipelineHandle, WARM_PROGRAMS};
use neura_graph::{Graph, GraphStamp};
use neura_kernel::{Banks, Kernel, Tables};
use neura_plan::{Contiguous, Plan, Product};
use neura_profile::{Geometry, MatmulStrategy, MatmulTile, Profile};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::io::{Read, Write};
use std::mem::size_of;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};

pub(crate) struct Resident {
    signature: Vec<u8>,
    pub(crate) plan: Arc<Plan>,
    pub(crate) kernel: PipelineHandle,
    pub(crate) steps: Recycled,
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
        let storage = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        let resident = Self {
            signature,
            plan,
            kernel,
            steps: Recycled::claim(pool, "neura steps", steps_bytes, storage),
            pool: pool.clone(),
        };
        let queue = context.queue();
        if !resident.plan.steps().is_empty() {
            resident.steps.buffer().write(queue, resident.plan.steps());
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
pub(crate) struct KernelIdentity {
    pub(crate) kinds: Vec<Kind>,
    pub(crate) elements: Vec<Element>,
    pub(crate) matmul_elements: Vec<Element>,
    pub(crate) geometry: Geometry,
    pub(crate) strategies: u32,
    pub(crate) authored: bool,
    pub(crate) banks: Banks,
    pub(crate) paged: bool,
    pub(crate) gated: bool,
    pub(crate) tables: Tables,
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub(crate) struct PlanRequest {
    pub(crate) stamp: GraphStamp,
    pub(crate) profile: neura_profile::Profile,
    pub(crate) alignment: u64,
    pub(crate) chosen: Vec<(Product, neura_profile::MatmulTile)>,
    pub(crate) paged: bool,
    pub(crate) weight_slots: u32,
}

const ASSEMBLY_CEILING: usize = 8;

pub(crate) struct Assembly {
    pub(crate) signature: Vec<u8>,
    pub(crate) plan: Arc<Plan>,
    pub(crate) kernel: Arc<Kernel>,
}

struct WarmKernel {
    kernel: Arc<Kernel>,
    declared: u64,
}

struct KernelCache {
    entries: HashMap<KernelIdentity, WarmKernel>,
    clock: u64,
}

impl KernelCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
        }
    }

    fn get(&mut self, identity: &KernelIdentity) -> Option<Arc<Kernel>> {
        self.clock += 1;
        match self.entries.get_mut(identity) {
            Some(warm) => {
                warm.declared = self.clock;
                Some(warm.kernel.clone())
            }
            None => None,
        }
    }

    fn insert(&mut self, identity: KernelIdentity, kernel: Arc<Kernel>) {
        self.entries.insert(
            identity,
            WarmKernel {
                kernel,
                declared: self.clock,
            },
        );
    }

    fn retain_warm(&mut self) {
        if self.entries.len() <= WARM_PROGRAMS {
            return;
        }
        let mut declared = self
            .entries
            .values()
            .map(|warm| warm.declared)
            .collect::<Vec<u64>>();
        declared.sort_unstable();
        let oldest = declared[self.entries.len() - WARM_PROGRAMS];
        self.entries.retain(|_, warm| warm.declared >= oldest);
    }
}

pub(crate) struct Artifacts {
    kernels: Mutex<KernelCache>,
    assemblies: Mutex<Vec<(PlanRequest, Arc<Assembly>)>>,
    residents: Mutex<HashMap<u64, Vec<Weak<Resident>>>>,
    built: AtomicUsize,
}

impl Artifacts {
    pub(crate) fn new() -> Self {
        Self {
            kernels: Mutex::new(KernelCache::new()),
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
        request: PlanRequest,
        build: impl FnOnce() -> Assembly,
    ) -> Arc<Assembly> {
        let identity = request;
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
            .entries
            .len()
    }

    pub(crate) fn kernel(
        &self,
        identity: KernelIdentity,
        assemble: impl FnOnce() -> Kernel,
    ) -> Arc<Kernel> {
        let mut kernels = self
            .kernels
            .lock()
            .expect("a kernel cache is never poisoned");
        if let Some(kernel) = kernels.get(&identity) {
            return kernel;
        }
        let kernel = Arc::new(assemble());
        kernels.insert(identity, kernel.clone());
        kernels.retain_warm();
        kernel
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
        residents.retain(|_, bucket| {
            bucket.retain(|resident| resident.strong_count() > 0);
            !bucket.is_empty()
        });
        let bucket = residents.entry(key).or_default();
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
    assemblies: &mut Vec<(PlanRequest, Arc<Assembly>)>,
    identity: &PlanRequest,
) -> Option<Arc<Assembly>> {
    let found = assemblies.iter().position(|(kept, _)| kept == identity)?;
    Some(assemblies.remove(found).1)
}

fn keep_assembly(
    assemblies: &mut Vec<(PlanRequest, Arc<Assembly>)>,
    identity: PlanRequest,
    assembly: Arc<Assembly>,
) {
    let _ = take_assembly(assemblies, &identity);
    assemblies.push((identity, assembly));
    while assemblies.len() > ASSEMBLY_CEILING {
        assemblies.remove(0);
    }
}

pub(crate) fn tables(plan: &Plan) -> Tables {
    Tables::of(
        plan.slot_bounds().len() as u32,
        plan.measures().len() as u32,
        plan.patches().len() as u32,
        plan.patch_list().len() as u32,
    )
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
        bytes.extend(tile.slices().to_le_bytes());
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

const PLAN_FORMAT: u32 = 3;
pub(crate) const PLAN_IMAGE_FLOOR: u32 = 1024;

pub(crate) fn plan_key(
    device: &AdapterInfo,
    graph: &Graph,
    profile: Profile,
    chosen: &[(Product, MatmulTile)],
    alignment: u64,
    paged: bool,
    weight_slots: u32,
) -> String {
    let fingerprint = graph.fingerprint();
    let mut digest = Digest::new();
    digest.word(PLAN_FORMAT);
    digest.long(neura_plan::PLAN_BUILD);
    digest.text(&device.name);
    digest.text(&device.id.to_string());
    digest.word(backend(device));
    digest.long(fingerprint);
    let snapshot = graph.snapshot();
    let authored = snapshot.authored();
    digest.length(authored.len());
    for count in authored {
        digest.word(*count);
    }
    digest.long(alignment);
    digest.flag(paged);
    digest.word(weight_slots);
    digest.word(profile.workgroup());
    digest.word(profile.registers());
    digest.long(profile.shared_bytes());
    digest.length(profile.tiles().len());
    for tile in profile.tiles() {
        for word in tile_identity(*tile) {
            digest.word(word);
        }
    }
    digest.length(chosen.len());
    for (product, tile) in chosen {
        digest.word(product.planes());
        digest.word(product.rows());
        digest.word(product.columns());
        digest.word(product.depth());
        digest.word(product.weight().element().code());
        digest.word(contiguous(product.weight().contiguous()));
        for word in tile_identity(*tile) {
            digest.word(word);
        }
    }
    format!(
        "plan-image/{:016x}/{PLAN_FORMAT}-{fingerprint:016x}-{:016x}",
        neura_plan::PLAN_BUILD,
        digest.finish()
    )
}

pub(crate) fn pack_plan(plan: &Plan) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder
        .write_all(&plan.image())
        .expect("a zlib stream writes to memory");
    encoder.finish().expect("a zlib stream closes in memory")
}

pub(crate) fn unpack_plan(
    payload: &[u8],
    graph: &Graph,
    alignment: u64,
    profile: Profile,
    encoding_bytes: u64,
    weight_slots: u32,
) -> Option<Plan> {
    let mut image = Vec::new();
    ZlibDecoder::new(payload).read_to_end(&mut image).ok()?;
    Plan::restore(
        graph,
        alignment,
        profile,
        encoding_bytes,
        weight_slots,
        &image,
    )
}

fn tile_identity(tile: MatmulTile) -> [u32; 8] {
    [
        strategy(tile.strategy()),
        tile.rows(),
        tile.columns(),
        tile.depth(),
        tile.threads(),
        tile.thread_rows(),
        tile.thread_columns(),
        u32::from(tile.stages_a_run()),
    ]
}

fn strategy(strategy: MatmulStrategy) -> u32 {
    match strategy {
        MatmulStrategy::Staged => 0,
        MatmulStrategy::Streamed => 1,
        MatmulStrategy::Cooperative => 2,
        MatmulStrategy::Folded => 3,
    }
}

fn contiguous(contiguous: Contiguous) -> u32 {
    match contiguous {
        Contiguous::Reduction => 0,
        Contiguous::Output => 1,
    }
}

fn backend(device: &AdapterInfo) -> u32 {
    match device.backend {
        neura_gpu::Backend::Vulkan => 0,
        neura_gpu::Backend::Dx12 => 1,
        neura_gpu::Backend::Metal => 2,
    }
}
