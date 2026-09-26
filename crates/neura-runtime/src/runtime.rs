use crate::checkpoint::Checkpoint;
use crate::heap::Heap;
use crate::pool::{Pool, Recycled};
use crate::program::{Program, Weights};
use crate::tape::{self, DeviceTape, Plan, Tapes};
use neura_abi::{Kind, MAX_DISPATCH_SEGMENTS, Placement, Refusal, WORD_BYTES};
use neura_gpu::{
    BufferUsages, Device, GpuContext, GpuRequest, GpuUnavailable, Readback, Submission,
    SubmissionIndex,
};
use neura_graph::{Graph, Value};
use neura_op as op;
use neura_precision::{pack, unpack};
use neura_profile::{Budget, Geometry, Profile};
use neura_program::{Encoding, Layout, Span};
use neura_shader::Megakernel;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

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
    tapes: Tapes,
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
        assert!(
            MAX_DISPATCH_SEGMENTS <= context.limits().max_compute_workgroups_per_dimension,
            "one dispatch addresses {MAX_DISPATCH_SEGMENTS} segments where the device launches workgroups in steps of {}",
            context.limits().max_compute_workgroups_per_dimension,
        );
        Self {
            alignment: context.binding_alignment(),
            readback: Readback::new(context.device(), readback_bytes, READBACK_SLOTS),
            heap: Arc::new(Heap::new(&context, heap_bytes)),
            pool: Pool::of(context.device(), crate::pool::POOL_BYTES),
            tapes: Tapes::new(),
            context,
        }
    }

    pub fn device_tapes(&self) -> usize {
        self.tapes.resident()
    }

    pub fn profiles(&self) -> Vec<Profile> {
        Profile::derive(self.budget())
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
        checkpoint.matches(weights.region());
        weights
            .buffer()
            .write_at(self.context.queue(), weights.offset(), checkpoint.payload());
    }

    pub fn checkpoint(&self, weights: &Weights<'_>) -> Checkpoint {
        self.context.assert_alive();
        assert!(
            weights.lives_on(&self.heap),
            "this weight store lives on the device heap of another runtime",
        );
        let bytes = weights.region().bytes();
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
        Checkpoint::of(weights.region(), payload)
    }

    fn parameter_store(&self, graph: &Graph) -> (Weights<'_>, Layout) {
        let layout = Layout::of(graph, self.alignment);
        let store = self.heap.allocate(layout.weights().words());
        let weights = Weights::new(store, layout.weights().clone());
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
                &pack(seed.element(), &values),
            );
        }
    }

    fn pour(&self, layout: &Layout, weights: &Weights<'_>, checkpoint: &Checkpoint) {
        checkpoint.matches(layout.weights());
        weights
            .buffer()
            .write_at(self.context.queue(), weights.offset(), checkpoint.payload());
    }

    pub fn rebind(&self, weights: &Weights<'_>, graph: &Graph<'_>) {
        self.context.assert_alive();
        let layout = Layout::of(graph, self.alignment);
        assert_eq!(
            layout.weights(),
            weights.region(),
            "this parameter store of {} tensors holds {} bytes where the graph asks for {} tensors and {} bytes; a store rebinds only onto a graph that declares the very same parameters in the very same order",
            weights.tensors(),
            weights.bytes(),
            layout.weights().tensors(),
            layout.weights().bytes(),
        );
    }

    pub fn compile<'r>(&'r self, graph: &Graph, weights: &Weights<'r>) -> Program<'r> {
        self.compile_with(graph, weights, self.default_profile())
    }

    pub fn compile_with<'r>(
        &'r self,
        graph: &Graph,
        weights: &Weights<'r>,
        profile: Profile,
    ) -> Program<'r> {
        let (threads, shared_bytes) = self.workgroup_budget();
        assert!(
            profile.fits(threads, shared_bytes),
            "{profile:?} asks the device for {} threads and {} workgroup bytes, while it offers {threads} and {shared_bytes}",
            profile.workgroup(),
            profile.shared_bytes(),
        );
        self.context.assert_alive();
        assert!(
            weights.lives_on(&self.heap),
            "this weight store lives on the device heap of another runtime",
        );
        let tape = self.assemble(graph, profile);
        assert!(
            tape.encoding.task_count() > 0,
            "a program whose tape holds no task has nothing for the device to run",
        );
        assert!(
            tape.encoding.weights() == weights.region(),
            "this graph holds {} parameters where the weight store carries {}; one store serves every program of one model",
            tape.encoding.weights().tensors(),
            weights.tensors(),
        );
        tape.kernel.compile();
        let tensors = self
            .heap
            .allocate(tape.encoding.tensor_bytes() / WORD_BYTES);
        Program::of(&self.context, tape, tensors, weights.clone())
    }

    fn assemble(&self, graph: &Graph, profile: Profile) -> Arc<DeviceTape> {
        let plan = self.tapes.plan(graph.stamp(), profile, self.alignment, || {
            self.plan(graph, profile)
        });
        self.tapes.of(plan.signature.clone(), |signature| {
            DeviceTape::build(
                &self.context,
                &self.pool,
                plan.encoding.clone(),
                plan.kernel.clone(),
                signature,
            )
        })
    }

    fn plan(&self, graph: &Graph, profile: Profile) -> Plan {
        let encoding = Arc::new(Encoding::of(graph, self.alignment, profile));
        let signature = tape::signature(&encoding, profile, self.alignment);
        let kinds = encoding.kinds().to_vec();
        let elements = encoding.elements().to_vec();
        let geometry = Geometry::of(
            profile.workgroup(),
            profile.shared_bytes(),
            encoding.tiles(),
            encoding.attention(),
        );
        let kernel = self.tapes.kernel(&kinds, &elements, geometry.clone(), || {
            Megakernel::assemble(&kinds, &elements, geometry)
        });
        Plan {
            signature,
            encoding,
            kernel,
        }
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

    fn measure(&self, program: &Program<'_>) -> f64 {
        for _ in 0..TUNE_WARMUP {
            self.run(program);
        }
        self.context.drain();
        let started = Instant::now();
        for _ in 0..TUNE_ROUNDS {
            self.run(program);
        }
        self.context.drain();
        started.elapsed().as_secs_f64() / f64::from(TUNE_ROUNDS)
    }

    pub fn run(&self, program: &Program<'_>) {
        self.assert_owns(program);
        self.context.assert_alive();
        assert!(
            program.is_compiled(),
            "a device program compiles at the call that compiles it, and a run only runs what a compile has compiled",
        );
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura program");
        submission.clear(program.refusal.buffer(), 0, program.refusal.buffer().size());
        for (index, dispatch) in program.tape.encoding.dispatches().iter().enumerate() {
            let offset = u32::try_from(index as u64 * self.alignment)
                .expect("a dispatch bounds offset fits in the native binding range");
            submission.dispatch(
                &program.tape.kernel,
                &program.group,
                &[offset],
                [dispatch.segments, 1, 1],
            );
        }
        submission.submit(self.context.queue());
    }

    pub fn write(&self, program: &Program<'_>, value: Value<'_>, data: &[f32]) {
        self.assert_owns(program);
        let span = program.span(value);
        assert!(
            program.readable(value),
            "value {} is a temporary whose storage a later task of the tape reuses, or a view that walks a layout the storage does not; retain the tensor that owns the storage, and materialize a permuted view before writing it",
            value.id(),
        );
        assert_eq!(
            data.len(),
            span.elements as usize,
            "writing {} numbers into a tensor of {} numbers",
            data.len(),
            span.elements,
        );
        let bytes = pack(span.element, data);
        program
            .heap()
            .write_at(self.context.queue(), span.offset, &bytes);
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
        self.context.assert_alive();
        assert!(!values.is_empty(), "a pull names at least one tensor");
        let spans = values
            .iter()
            .map(|value| program.span(*value))
            .collect::<Vec<_>>();
        for value in values {
            assert!(
                program.readable(*value),
                "value {} is a temporary whose storage a later task of the tape reuses, or a view that walks a layout the storage does not; retain the tensor that owns the storage, and materialize a permuted view before pulling it",
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
            submission.copy(program.heap(), span.offset, staging, at, bytes);
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
        self.tapes.kernels()
    }

    pub fn built_plans(&self) -> usize {
        self.tapes.built()
    }
}

fn span_bytes(span: Span) -> u64 {
    span.element.words(u64::from(span.elements)) * WORD_BYTES
}

fn refusal_message(word: u32) -> String {
    let refused = word >> 16;
    let code = (word & 0xffff) - 1;
    if refused == neura_abi::REFUSAL_ELEMENT {
        return format!("the device refused element {code} of a tensor");
    }
    if refused >= Kind::COUNT {
        return format!("the device refused kind {refused} with code {code}");
    }
    let kind = Kind::of(refused);
    match kind.refusal() {
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
            "the device refused an index outside the rows of the {} task",
            kind.name(),
        ),
        Refusal::Geometry => format!(
            "the device refused geometry {code} of the {} task",
            kind.name(),
        ),
        Refusal::Code => format!("the device refused code {code} of the {} task", kind.name()),
    }
}
