use crate::cache::{self, Artifacts, Assembly, Resident};
use crate::checkpoint::Checkpoint;
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
use neura_plan::{Layout, Plan, Span};
use neura_pointwise as op;
use neura_precision::{pack, unpack};
use neura_profile::CooperativeMatrix;
use neura_profile::{Budget, Geometry, Profile};
use std::marker::PhantomData;
use std::sync::Arc;

pub const DEFAULT_READBACK_BYTES: u64 = 1 << 20;
pub const DEFAULT_HEAP_BYTES: u64 = 16 << 20;
pub const READBACK_SLOTS: u64 = 2;
const TUNE_WARMUP: u32 = 2;
const TUNE_ROUNDS: u32 = 8;
const ENTROPY_SEED: u32 = 0x9e37_79b9;

pub struct Readout<'r> {
    brand: PhantomData<&'r ()>,
    slot: usize,
    submission: SubmissionIndex,
    spans: Vec<(Span, u64, u64)>,
    total: u64,
    refusal: u64,
}

pub struct Run<'r> {
    brand: PhantomData<&'r ()>,
    queue: Queue,
    submission: SubmissionIndex,
}

impl Run<'_> {
    pub fn seconds(self) -> f64 {
        self.queue.seconds(self.submission)
    }
}

pub struct RuntimeRequest {
    pub gpu: GpuRequest,
    pub readback_bytes: u64,
    pub heap_bytes: u64,
}

impl Default for RuntimeRequest {
    fn default() -> Self {
        Self {
            gpu: GpuRequest::default(),
            readback_bytes: DEFAULT_READBACK_BYTES,
            heap_bytes: DEFAULT_HEAP_BYTES,
        }
    }
}

pub struct Runtime {
    context: GpuContext,
    readback: Readback,
    heap: Arc<Heap>,
    pool: Arc<Pool>,
    artifacts: Artifacts,
    alignment: u64,
}

impl Runtime {
    pub async fn open(request: RuntimeRequest) -> Result<Self, GpuUnavailable> {
        let context = GpuContext::open(&request.gpu).await?;
        Ok(Self::of_context(
            context,
            request.readback_bytes,
            request.heap_bytes,
        ))
    }

    pub fn from_device(device: Device, heap_bytes: u64) -> Self {
        Self::of_context(
            GpuContext::of_device(device),
            DEFAULT_READBACK_BYTES,
            heap_bytes,
        )
    }

    pub fn heap_bytes(&self) -> u64 {
        self.heap.bytes()
    }

    fn of_context(context: GpuContext, readback_bytes: u64, heap_bytes: u64) -> Self {
        assert!(
            heap_bytes <= context.limits().max_storage_buffer_binding_size,
            "a device heap of {heap_bytes} bytes outruns the {} bytes one storage binding holds",
            context.limits().max_storage_buffer_binding_size,
        );
        Self {
            alignment: context.binding_alignment(),
            readback: Readback::new(context.device(), readback_bytes, READBACK_SLOTS),
            heap: Arc::new(Heap::new(&context, heap_bytes)),
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

    pub fn weights(&self, graph: &Graph) -> Weights<'_> {
        self.context.assert_alive();
        let (weights, layout) = self.parameter_store(graph);
        self.seed(&layout, &weights);
        weights
    }

    pub fn load(&self, graph: &Graph, checkpoint: &Checkpoint) -> Weights<'_> {
        self.context.assert_alive();
        let (weights, layout) = self.parameter_store(graph);
        self.pour(&layout, &weights, checkpoint);
        weights
    }

    pub fn restore(&self, weights: &Weights<'_>, checkpoint: &Checkpoint) {
        self.context.assert_alive();
        assert!(
            weights.lives_on(&self.heap),
            "this weight store lives on the device heap of another runtime",
        );
        checkpoint.matches(weights.region(), weights.state());
        weights.buffer().write_at(
            self.context.queue(),
            weights.offset(),
            checkpoint.payload(weights.words()),
        );
    }

    pub fn checkpoint(&self, weights: &Weights<'_>) -> Checkpoint {
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
        let payload = staging
            .buffer()
            .read(self.context.queue(), submission, bytes);
        Checkpoint::of(weights.region(), weights.state(), payload)
    }

    fn parameter_store(&self, graph: &Graph) -> (Weights<'_>, Layout) {
        let layout = Layout::of(graph, self.alignment);
        let store = self.heap.allocate(layout.words());
        let weights = Weights::new(store, layout.weights().clone(), layout.state().clone());
        (weights, layout)
    }

    fn seed(&self, layout: &Layout, weights: &Weights<'_>) {
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

    fn pour(&self, layout: &Layout, weights: &Weights<'_>, checkpoint: &Checkpoint) {
        checkpoint.matches(layout.weights(), layout.state());
        weights.buffer().write_at(
            self.context.queue(),
            weights.offset(),
            checkpoint.payload(layout.words()),
        );
    }

    pub fn rebind(&self, weights: &Weights<'_>, graph: &Graph<'_>) {
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

    pub fn compile<'r>(&'r self, graph: &Graph, weights: &Weights<'r>) -> Program<'r> {
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

    pub fn compile_with<'r>(
        &'r self,
        graph: &Graph,
        weights: &Weights<'r>,
        profile: Profile,
    ) -> Program<'r> {
        self.assert_profile(profile);
        assert!(
            weights.lives_on(&self.heap),
            "this weight store lives on the device heap of another runtime",
        );
        let revision = graph.revision();
        let (resident, plan) = self.assemble(graph, profile);
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

    fn assemble(&self, graph: &Graph, profile: Profile) -> (Arc<Resident>, Arc<Plan>) {
        let assembly = self
            .artifacts
            .assemble(graph.stamp(), profile, self.alignment, || {
                self.assembly(graph, profile)
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

    fn assembly(&self, graph: &Graph, profile: Profile) -> Assembly {
        let plan = Arc::new(Plan::of(graph, self.alignment, profile));
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
        self.artifacts
            .kernel(&kinds, &elements, geometry.clone(), || {
                Kernel::assemble(&kinds, &elements, geometry)
            })
    }

    pub fn tune<'r>(&'r self, graph: &Graph, weights: &Weights<'r>) -> Program<'r> {
        let scratch = graph.updates_weights().then(|| self.scratch_weights(graph));
        let mut measured = self.profiles().into_iter().map(|profile| {
            let measuring = scratch.as_ref().unwrap_or(weights);
            (
                profile,
                self.measure(&self.compile_with(graph, measuring, profile)),
            )
        });
        let (mut fastest, mut seconds) = measured
            .next()
            .expect("the device offers no workgroup the framework can schedule");
        for (profile, elapsed) in measured {
            if elapsed < seconds {
                fastest = profile;
                seconds = elapsed;
            }
        }
        self.compile_with(graph, weights, fastest)
    }

    fn scratch_weights(&self, graph: &Graph) -> Weights<'_> {
        self.context.assert_alive();
        let (scratch, layout) = self.parameter_store(graph);
        self.seed(&layout, &scratch);
        scratch
    }

    pub fn measure(&self, program: &Program<'_>) -> f64 {
        for _ in 0..TUNE_WARMUP {
            self.run(program);
        }
        let mut measured = 0.0;
        for _ in 0..TUNE_ROUNDS {
            measured += self.run(program).seconds();
        }
        measured / f64::from(TUNE_ROUNDS)
    }

    pub fn bind(&self, program: &Program<'_>, extents: &[u32]) {
        self.assert_owns(program);
        program.assert_current();
        self.context.assert_alive();
        program.bind(extents);
    }

    pub fn run(&self, program: &Program<'_>) -> Run<'_> {
        self.assert_owns(program);
        program.assert_current();
        self.context.assert_alive();
        assert!(
            program.is_compiled(),
            "a device program compiles at the call that compiles it, and a run only runs what a compile has compiled",
        );
        if program.dynamic() {
            let extents = program.extents().to_vec();
            if program.records_pending() {
                program.write_records(self.context.queue(), &extents);
                program.records_written();
            }
        }
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura program");
        submission.clear(program.refusal.buffer(), 0, program.refusal.buffer().size());
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
            brand: PhantomData,
            queue: self.context.queue().clone(),
            submission,
        }
    }

    pub fn write(&self, program: &Program<'_>, value: Value<'_>, data: &[f32]) {
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
        if !bytes.is_empty() {
            program
                .heap()
                .write_at(self.context.queue(), span.offset, &bytes);
        }
    }

    pub fn read(&self, program: &Program<'_>, value: Value<'_>) -> Vec<f32> {
        let mut values = self.collect(self.pull(program, &[value]));
        values.pop().expect("one tensor was read")
    }

    pub fn read_many(&self, program: &Program<'_>, values: &[Value<'_>]) -> Vec<Vec<f32>> {
        self.collect(self.pull(program, values))
    }

    pub fn pull(&self, program: &Program<'_>, values: &[Value<'_>]) -> Readout<'_> {
        self.assert_owns(program);
        program.assert_current();
        self.context.assert_alive();
        assert!(!values.is_empty(), "a pull names at least one tensor");
        let spans = values
            .iter()
            .map(|value| program.span(*value))
            .collect::<Vec<_>>();
        for value in values {
            assert!(
                program.readable(*value),
                "value {} is a temporary whose storage a later task of the plan reuses, or a view that walks a layout the storage does not; retain the tensor that owns the storage, and materialize a permuted view before pulling it",
                value.id(),
            );
        }
        let total = spans.iter().map(|span| span_bytes(*span)).sum::<u64>() + WORD_BYTES;
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
            let bytes = span_bytes(*span);
            if bytes > 0 {
                submission.copy(program.heap(), span.offset, staging, at, bytes);
            }
            collected.push((*span, at, bytes));
            at += bytes;
        }
        submission.copy(program.refusal.buffer(), 0, staging, at, WORD_BYTES);
        let submission = submission.submit(self.context.queue());
        Readout {
            brand: PhantomData,
            slot,
            submission,
            spans: collected,
            total,
            refusal: at,
        }
    }

    pub fn collect(&self, readout: Readout<'_>) -> Vec<Vec<f32>> {
        self.context.assert_alive();
        let bytes = self.readback.finish(
            self.context.queue(),
            readout.slot,
            readout.submission,
            readout.total,
        );
        self.context.assert_alive();
        let refusal = u32::from_ne_bytes(
            bytes[readout.refusal as usize..(readout.refusal + WORD_BYTES) as usize]
                .try_into()
                .expect("a word was copied back"),
        );
        assert_eq!(refusal, 0, "{}", refusal_message(refusal));
        readout
            .spans
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

    fn assert_owns(&self, program: &Program<'_>) {
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

fn span_bytes(span: Span) -> u64 {
    span.element.storage_words(u64::from(span.elements)) * WORD_BYTES
}

fn refusal_message(word: u32) -> String {
    let (subject, category, code) = Refusal::read(word);
    if category == Refusal::Element {
        return format!("the device refused element {code} of a tensor");
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
        Refusal::Geometry => format!(
            "the device refused geometry {code} of the {} task",
            kind.name(),
        ),
        Refusal::Origin => format!(
            "the device refused a query origin the {} task has no keys to place",
            kind.name(),
        ),
        Refusal::Task => format!("this device program carries no {} task", kind.name()),
        Refusal::Element => unreachable!("an element refusal carries no kind"),
    }
}
