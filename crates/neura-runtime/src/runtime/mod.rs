use crate::cache::{self, Artifacts, Assembly, Resident};
use crate::checkpoint::{Checkpoint, TensorData};
use crate::heap::Heap;
use crate::pool::{Pool, Recycled};
use crate::program::{Program, Weights};
use neura_abi::{Kind, Placement, Refusal, WORD_BYTES};
use neura_gpu::{
    BufferUsages, Device, GpuContext, GpuRequest, GpuUnavailable, Queue, Readback, Submission,
    SubmissionIndex,
};
use neura_graph::{Graph, Value};
use neura_kernel::Kernel;
use neura_plan::{Layout, Plan, Product, Region, Span};
use neura_pointwise as op;
use neura_precision::{pack, unpack};
use neura_profile::CooperativeMatrix;
use neura_profile::{Budget, Geometry, MatmulTile, Profile};
use std::sync::Arc;

mod tune;

pub const DEFAULT_READBACK_BYTES: u64 = 1 << 20;
pub const DEFAULT_READBACK_SLOTS: usize = 2;
pub const DEFAULT_HEAP_BYTES: u64 = 16 << 20;
const TUNE_WARMUP: u32 = 2;
const TUNE_ROUNDS: u32 = 8;
const ENTROPY_SEED: u32 = 0x9e37_79b9;

pub struct Readout {
    readback: Arc<Readback>,
    queue: Queue,
    slot: usize,
    submission: SubmissionIndex,
    spans: Vec<(Span, u64, u64)>,
    total: u64,
    refusal: u64,
}

impl Readout {
    pub fn collect(self) -> Vec<Vec<f32>> {
        let bytes = self
            .readback
            .finish(&self.queue, self.slot, self.submission, self.total);
        let refusal = u32::from_ne_bytes(
            bytes[self.refusal as usize..(self.refusal + WORD_BYTES) as usize]
                .try_into()
                .expect("a word was copied back"),
        );
        assert_eq!(refusal, 0, "{}", refusal_message(refusal));
        self.spans
            .iter()
            .map(|(span, offset, length)| {
                let start = *offset as usize;
                unpack(
                    span.element,
                    span.elements as usize,
                    &bytes[start..start + *length as usize],
                )
            })
            .collect()
    }
}

pub struct Run {
    queue: Queue,
    submission: SubmissionIndex,
}

impl Run {
    pub fn seconds(self) -> f64 {
        self.queue.seconds(self.submission)
    }
}

pub struct MemoryRequest {
    pub readback_bytes: u64,
    pub readback_slots: usize,
    pub heap_bytes: u64,
}

impl Default for MemoryRequest {
    fn default() -> Self {
        Self {
            readback_bytes: DEFAULT_READBACK_BYTES,
            readback_slots: DEFAULT_READBACK_SLOTS,
            heap_bytes: DEFAULT_HEAP_BYTES,
        }
    }
}

#[derive(Default)]
pub struct RuntimeRequest {
    pub gpu: GpuRequest,
    pub memory: MemoryRequest,
}

pub struct Runtime {
    context: GpuContext,
    readback: Arc<Readback>,
    heap: Arc<Heap>,
    pool: Arc<Pool>,
    artifacts: Artifacts,
    alignment: u64,
}

impl Runtime {
    pub fn open(request: RuntimeRequest) -> Result<Self, GpuUnavailable> {
        let context = GpuContext::open(&request.gpu)?;
        Ok(Self::from_device(context.device().clone(), request.memory))
    }

    pub fn from_device(device: Device, memory: MemoryRequest) -> Self {
        Self::of_context(GpuContext::of_device(device), memory)
    }

    pub fn heap_bytes(&self) -> u64 {
        self.heap.bytes()
    }

    fn of_context(context: GpuContext, memory: MemoryRequest) -> Self {
        assert!(
            memory.heap_bytes <= context.limits().max_storage_buffer_binding_size,
            "a device heap of {} bytes outruns the {} bytes one storage binding holds",
            memory.heap_bytes,
            context.limits().max_storage_buffer_binding_size,
        );
        Self {
            alignment: context.binding_alignment(),
            readback: Arc::new(Readback::new(
                context.device(),
                memory.readback_bytes,
                memory.readback_slots,
            )),
            heap: Arc::new(Heap::new(&context, memory.heap_bytes)),
            pool: Pool::of(context.device(), crate::pool::POOL_BYTES),
            artifacts: Artifacts::new(),
            context,
        }
    }

    pub fn resident_plans(&self) -> usize {
        self.artifacts.resident_plans()
    }

    pub fn profiles(&self) -> Vec<Profile> {
        Profile::derive(self.budget(), self.cooperative_matrix())
    }

    pub fn capability(&self) -> &neura_gpu::Capability {
        self.context.capability()
    }

    fn cooperative_matrix(&self) -> Option<CooperativeMatrix> {
        let capability = self.context.capability().cooperative_matrix?;
        Some(CooperativeMatrix::new(
            capability.subgroup,
            capability.rows,
            capability.columns,
            capability.depth,
        ))
    }

    pub fn budget(&self) -> Budget {
        let (threads, shared_bytes) = self.workgroup_budget();
        Budget::of(threads, shared_bytes)
    }

    fn workgroup_budget(&self) -> (u32, u64) {
        let limits = self.context.limits();
        (
            limits
                .max_compute_invocations_per_workgroup
                .min(limits.max_compute_workgroup_size_x),
            u64::from(limits.max_compute_workgroup_storage_size),
        )
    }

    pub fn default_profile(&self) -> Profile {
        let profiles = self.profiles();
        profiles
            .iter()
            .copied()
            .min_by_key(|profile| profile.workgroup().abs_diff(Budget::BALANCED_THREADS))
            .expect("the device offers no workgroup the framework can schedule")
    }

    pub fn weights(&self, graph: &Graph) -> Weights {
        self.context.assert_alive();
        let (weights, layout) = self.parameter_store(graph);
        self.seed(&layout, &weights);
        weights
    }

    pub fn load(&self, graph: &Graph, checkpoint: &Checkpoint) -> Weights {
        self.context.assert_alive();
        let (weights, layout) = self.parameter_store(graph);
        self.pour(&layout, &weights, checkpoint);
        weights
    }

    pub fn restore(&self, weights: &Weights, checkpoint: &Checkpoint) {
        self.context.assert_alive();
        assert!(
            weights.lives_on(&self.heap),
            "this weight store lives on the device heap of another runtime",
        );
        let store = checkpoint.pour(weights.region(), weights.state(), weights.words());
        weights
            .buffer()
            .write_at(self.context.queue(), weights.offset(), &store);
    }

    pub fn checkpoint(&self, weights: &Weights) -> Checkpoint {
        self.context.assert_alive();
        assert!(
            weights.lives_on(&self.heap),
            "this weight store lives on the device heap of another runtime",
        );
        let bytes = weights.words() * WORD_BYTES;
        let staging = Recycled::claim(
            &self.pool,
            "neura checkpoint",
            bytes,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura checkpoint");
        submission.copy(
            weights.buffer(),
            weights.offset(),
            staging.buffer(),
            0,
            bytes,
        );
        let submission = submission.submit(self.context.queue());
        let store = staging
            .buffer()
            .read(self.context.queue(), submission, bytes);
        let mut tensors = stored(weights.region(), &store, "parameter");
        if weights.state().tensors() > 0 {
            tensors.extend(stored(weights.state(), &store, "training state"));
        }
        Checkpoint::pack(&tensors)
    }

    fn parameter_store(&self, graph: &Graph) -> (Weights, Layout) {
        let layout = Layout::of(graph, self.alignment);
        let store = self.heap.allocate(layout.words());
        let weights = Weights::new(store, layout.weights().clone(), layout.state().clone());
        (weights, layout)
    }

    fn seed(&self, layout: &Layout, weights: &Weights) {
        let store = weights.allocation();
        let placement = Placement::new(0, store.word());
        let queue = self.context.queue();
        let mut entropy = ENTROPY_SEED;
        for seed in layout.seeds() {
            let values = seed.init().samples(seed.elements(), &mut entropy);
            self.heap.buffer().write_at(
                queue,
                layout.weight_bytes(placement, seed.address()),
                &pack(seed.element(), seed.scale(), &values),
            );
        }
    }

    fn pour(&self, layout: &Layout, weights: &Weights, checkpoint: &Checkpoint) {
        let store = checkpoint.pour(layout.weights(), layout.state(), layout.words());
        weights
            .buffer()
            .write_at(self.context.queue(), weights.offset(), &store);
    }

    pub fn rebind(&self, weights: &Weights, graph: &Graph<'_>) {
        self.context.assert_alive();
        let layout = Layout::of(graph, self.alignment);
        assert_eq!(
            layout.weights(),
            weights.region(),
            "this weight store of {} tensors holds {} bytes where the graph asks for {} tensors and {} bytes; a store rebinds only onto a graph that declares the very same model parameters in the very same order",
            weights.tensors(),
            weights.bytes(),
            layout.weights().tensors(),
            layout.weights().bytes(),
        );
        assert!(
            layout.state().tensors() == 0 || layout.state() == weights.state(),
            "this graph trains with {} tensors of state where the store carries {}; a store rebinds only onto a graph that trains with the state it holds",
            layout.state().tensors(),
            weights.state().tensors(),
        );
    }

    pub fn compile(&self, graph: &Graph, weights: &Weights) -> Program {
        self.compile_with(graph, weights, self.default_profile())
    }

    pub fn precompile(&self, graph: &Graph, profile: Profile) {
        self.assert_profile(profile);
        let plan = Plan::of(graph, self.alignment, profile);
        let kernel = self.kernel(&plan, profile);
        self.context.declare(kernel.program()).compile();
    }

    fn assert_profile(&self, profile: Profile) {
        let (threads, shared_bytes) = self.workgroup_budget();
        assert!(
            profile.fits(threads, shared_bytes),
            "{profile:?} asks the device for {} threads and {} workgroup bytes, while it offers {threads} and {shared_bytes}",
            profile.workgroup(),
            profile.shared_bytes(),
        );
        self.context.assert_alive();
    }

    pub fn compile_with(&self, graph: &Graph, weights: &Weights, profile: Profile) -> Program {
        self.compile_chosen(graph, weights, profile, &[])
    }

    pub fn compile_chosen(
        &self,
        graph: &Graph,
        weights: &Weights,
        profile: Profile,
        chosen: &[(Product, MatmulTile)],
    ) -> Program {
        self.assert_profile(profile);
        assert!(
            weights.lives_on(&self.heap),
            "this weight store lives on the device heap of another runtime",
        );
        let revision = graph.revision();
        let (resident, plan) = self.assemble(graph, profile, chosen);
        assert!(
            resident.plan.task_count() > 0,
            "a program whose plan holds no task has nothing for the device to run",
        );
        assert!(
            resident.plan.weights() == weights.region(),
            "this graph holds {} parameters where the weight store carries {}; one store serves every program of one model",
            resident.plan.weights().tensors(),
            weights.tensors(),
        );
        assert!(
            resident.plan.state().tensors() == 0 || resident.plan.state() == weights.state(),
            "this graph trains with {} tensors of state where the store carries {}; a store serves one training state",
            resident.plan.state().tensors(),
            weights.state().tensors(),
        );
        resident.kernel.compile();
        let tensors = self.heap.allocate(plan.tensor_bytes() / WORD_BYTES);
        Program::of(
            &self.context,
            resident,
            plan,
            tensors,
            weights.clone(),
            revision,
        )
    }

    fn assemble(
        &self,
        graph: &Graph,
        profile: Profile,
        chosen: &[(Product, MatmulTile)],
    ) -> (Arc<Resident>, Arc<Plan>) {
        let assembly =
            self.artifacts
                .assemble(graph.stamp(), profile, self.alignment, chosen, || {
                    self.assembly(graph, profile, chosen)
                });
        let resident = self
            .artifacts
            .resident(assembly.signature.clone(), |signature| {
                Resident::build(
                    &self.context,
                    &self.pool,
                    assembly.plan.clone(),
                    assembly.kernel.clone(),
                    signature,
                )
            });
        (resident, assembly.plan.clone())
    }

    fn assembly(
        &self,
        graph: &Graph,
        profile: Profile,
        chosen: &[(Product, MatmulTile)],
    ) -> Assembly {
        let plan = Arc::new(Plan::chosen(graph, self.alignment, profile, chosen));
        let signature = cache::signature(&plan, profile, self.alignment);
        let kernel = self.kernel(&plan, profile);
        Assembly {
            signature,
            plan,
            kernel,
        }
    }

    fn kernel(&self, plan: &Plan, profile: Profile) -> Arc<Kernel> {
        let kinds = plan.kinds().to_vec();
        let elements = plan.elements().to_vec();
        let walked = plan.walked_tiles().collect::<Vec<_>>();
        let geometry = Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            &walked,
            plan.attention(),
        );
        let authored = plan.carries_authored();
        self.artifacts
            .kernel(&kinds, &elements, geometry.clone(), authored, || {
                Kernel::assemble(&kinds, &elements, geometry, authored)
            })
    }

    fn scratch_weights(&self, graph: &Graph) -> Weights {
        self.context.assert_alive();
        let (scratch, layout) = self.parameter_store(graph);
        self.seed(&layout, &scratch);
        scratch
    }

    pub fn measure(&self, program: &Program) -> f64 {
        for _ in 0..TUNE_WARMUP {
            self.run(program);
        }
        let mut measured = 0.0;
        for _ in 0..TUNE_ROUNDS {
            measured += self.run(program).seconds();
        }
        measured / f64::from(TUNE_ROUNDS)
    }

    pub fn bind(&self, program: &Program, extents: &[u32]) {
        self.assert_owns(program);
        program.assert_current();
        self.context.assert_alive();
        program.bind(extents);
    }

    pub fn run(&self, program: &Program) -> Run {
        self.assert_owns(program);
        program.assert_current();
        self.context.assert_alive();
        assert!(
            program.is_compiled(),
            "a device program compiles at the call that compiles it, and a run only runs what a compile has compiled",
        );
        program.write_extents(self.context.queue());
        if program.dynamic() && program.records_pending() {
            let extents = program.host_extents();
            program.write_records(self.context.queue(), &extents);
            program.records_written();
        }
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura program");
        program
            .progress
            .buffer()
            .write_at(self.context.queue(), 0, &program.header);
        submission.dispatch(
            &program.resident.kernel,
            &program.group,
            [program.workgroups, 1, 1],
        );
        let submission = submission.submit(self.context.queue());
        Run {
            queue: self.context.queue().clone(),
            submission,
        }
    }

    pub fn write(&self, program: &Program, value: Value<'_>, data: &[f32]) {
        self.assert_owns(program);
        program.assert_current();
        let span = program.span(value);
        assert!(
            program.readable(value),
            "value {} is a temporary whose storage a later task of the plan reuses, or a view that walks a layout the storage does not; retain the tensor that owns the storage, and materialize a permuted view before writing it",
            value.id(),
        );
        assert_eq!(
            data.len(),
            span.elements as usize,
            "writing {} numbers into a tensor of {} numbers",
            data.len(),
            span.elements,
        );
        let bytes = pack(span.element, span.scale, data);
        let payload = span.payload_bytes() as usize;
        if payload > 0 {
            program
                .heap()
                .write_at(self.context.queue(), span.offset, &bytes[..payload]);
        }
        let table = &bytes[payload..];
        if !table.is_empty() {
            program.heap().write_at(
                self.context.queue(),
                span.offset + span.table_offset(),
                table,
            );
        }
    }

    pub fn read(&self, program: &Program, value: Value<'_>) -> Vec<f32> {
        let mut values = self.pull(program, &[value]).collect();
        values.pop().expect("one tensor was read")
    }

    pub fn read_many(&self, program: &Program, values: &[Value<'_>]) -> Vec<Vec<f32>> {
        self.pull(program, values).collect()
    }

    pub fn pull(&self, program: &Program, values: &[Value<'_>]) -> Readout {
        self.assert_owns(program);
        program.assert_current();
        self.context.assert_alive();
        assert!(!values.is_empty(), "a pull names at least one tensor");
        let extents = self.walked_extents(program);
        let spans = values
            .iter()
            .map(|value| program.span_with(*value, &extents))
            .collect::<Vec<_>>();
        for value in values {
            assert!(
                program.readable(*value),
                "value {} is a temporary whose storage a later task of the plan reuses, or a view that walks a layout the storage does not; retain the tensor that owns the storage, and materialize a permuted view before pulling it",
                value.id(),
            );
        }
        let total = spans.iter().map(|span| span.image_bytes()).sum::<u64>() + WORD_BYTES;
        assert!(
            total <= self.readback.capacity(),
            "pulling {total} bytes outruns the {} byte readback of this runtime",
            self.readback.capacity(),
        );
        let slot = self.readback.claim();
        let staging = self.readback.staging(slot);
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura pull");
        let mut collected = Vec::with_capacity(spans.len());
        let mut at = 0;
        for span in &spans {
            let payload = span.payload_bytes();
            let table = span.table_bytes();
            if payload > 0 {
                submission.copy(program.heap(), span.offset, staging, at, payload);
            }
            if table > 0 {
                submission.copy(
                    program.heap(),
                    span.offset + span.table_offset(),
                    staging,
                    at + payload,
                    table,
                );
            }
            collected.push((*span, at, payload + table));
            at += payload + table;
        }
        submission.copy(program.refusal.buffer(), 0, staging, at, WORD_BYTES);
        submission.clear(program.refusal.buffer(), 0, WORD_BYTES);
        let submission = submission.submit(self.context.queue());
        Readout {
            readback: self.readback.clone(),
            queue: self.context.queue().clone(),
            slot,
            submission,
            spans: collected,
            total,
            refusal: at,
        }
    }

    fn walked_extents(&self, program: &Program) -> Vec<u32> {
        if !program.carries_authored() {
            return program.host_extents();
        }
        if let Some(extents) = program.cached_extents() {
            return extents;
        }
        let buffer = program.extents_buffer();
        let bytes = buffer.size();
        let staging = Recycled::claim(
            &self.pool,
            "neura extents",
            bytes,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );
        let mut submission = Submission::new(self.context.device(), "neura extents");
        submission.copy(buffer, 0, staging.buffer(), 0, bytes);
        let submission = submission.submit(self.context.queue());
        let words = staging
            .buffer()
            .read(self.context.queue(), submission, bytes);
        let mut extents = Vec::with_capacity(program.slot_count());
        for at in 0..program.slot_count() {
            extents.push(u32::from_ne_bytes(
                words[at * 4..at * 4 + 4]
                    .try_into()
                    .expect("a device extent fills one word"),
            ));
        }
        program.cache_extents(extents.clone());
        extents
    }

    fn assert_owns(&self, program: &Program) {
        assert!(
            program.lives_on(&self.heap),
            "this program runs on the device heap of another runtime",
        );
    }

    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn readback_capacity(&self) -> u64 {
        self.readback.capacity()
    }

    pub fn readback_slots(&self) -> usize {
        self.readback.slots()
    }

    pub fn declared_kernels(&self) -> usize {
        self.context.declared_kernels()
    }

    pub fn assembled_kernels(&self) -> usize {
        self.artifacts.kernels()
    }

    pub fn built_plans(&self) -> usize {
        self.artifacts.built()
    }
}

fn stored<'s>(region: &'s Region, store: &'s [u8], section: &str) -> Vec<TensorData<'s>> {
    region
        .entries()
        .iter()
        .map(|entry| {
            let name = entry.name.as_deref().unwrap_or_else(|| {
                panic!(
                    "the {section} at word {} of {} carries no name, and a checkpoint names every tensor it holds; declare it with a named parameter or a named state",
                    entry.word,
                    entry.element.name(),
                )
            });
            let at = (entry.word * WORD_BYTES) as usize;
            let payload = (entry.element.payload_words(entry.elements) * WORD_BYTES) as usize;
            let quanta = (entry.element.quanta(entry.elements) * WORD_BYTES) as usize;
            TensorData {
                name,
                shape: entry.shape,
                element: entry.element,
                elements: entry.elements,
                scale: entry.scale,
                payload: &store[at..at + payload],
                quanta: &store[at + payload..at + payload + quanta],
            }
        })
        .collect()
}

fn refusal_message(word: u32) -> String {
    let (subject, category, code) = Refusal::read(word);
    if category == Refusal::Element {
        return format!("the device refused element {code} of a tensor");
    }
    if category == Refusal::Empty {
        return "the device refused a coordinate in a dimension of no numbers".to_owned();
    }
    if subject >= Kind::COUNT {
        return format!(
            "the device refused {} {code} of kind {subject}",
            category.name(),
        );
    }
    let kind = Kind::of(subject);
    match category {
        Refusal::Op => format!(
            "the device refused the {} op of the {} task",
            op::name(code),
            kind.name(),
        ),
        Refusal::Partial => format!(
            "the device refused the partial of the {} op over operand {} of the {} task",
            op::name(code / 2),
            code % 2,
            kind.name(),
        ),
        Refusal::Index => format!(
            "the device refused an index outside the rows the {} task names",
            kind.name(),
        ),
        Refusal::Extent => format!(
            "the device refused an extent of {code} rows a device count authors, beyond the bound its graph declares",
        ),
        Refusal::Geometry => format!(
            "the device refused geometry {code} of the {} task",
            kind.name(),
        ),
        Refusal::Origin => format!(
            "the device refused a query origin the {} task has no keys to place",
            kind.name(),
        ),
        Refusal::Task => format!("this device program carries no {} task", kind.name()),
        Refusal::Mask => format!(
            "the device refused a mask of the {} task, whose flag is neither a 1 nor a 0",
            kind.name(),
        ),
        Refusal::Element => unreachable!("an element refusal carries no kind"),
        Refusal::Empty => unreachable!("an empty refusal carries no kind"),
    }
}
