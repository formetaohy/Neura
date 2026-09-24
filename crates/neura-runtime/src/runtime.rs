use crate::checkpoint::Checkpoint;
use crate::heap::Heap;
use crate::pool::{Pool, Recycled};
use crate::program::{Program, Weights};
use crate::tape::{self, DeviceTape, Tapes};
use neura_abi::{
    Geometry, Kind, MAX_DISPATCH_SEGMENTS, PROFILES, Placement, Precision, Profile, REFUSAL_BYTES,
    WORD_BYTES, op,
};
use neura_gpu::{
    BufferUsages, ComputePassDescriptor, GpuContext, GpuRequest, GpuUnavailable, Readback,
    Submission, wgpu,
};
use neura_program::{Graph, Layout, Span, Store, Value};
use neura_shader::Megakernel;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;
use wgpu::{BufferAsyncError, MapMode, PollType};

pub const DEFAULT_READBACK_BYTES: u64 = 1 << 20;
pub const DEFAULT_HEAP_BYTES: u64 = 16 << 20;
pub const READBACK_SLOTS: u64 = 2;
const TUNE_WARMUP: u32 = 2;
const TUNE_ROUNDS: u32 = 8;
const ENTROPY_SEED: u32 = 0x9e37_79b9;
const CHECKPOINT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

pub struct Readout<'r> {
    brand: PhantomData<&'r ()>,
    slot: usize,
    submission: wgpu::SubmissionIndex,
    completed: mpsc::Receiver<Result<(), BufferAsyncError>>,
    precision: Precision,
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

    pub fn adopt(
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
        heap_bytes: u64,
    ) -> Self {
        Self::of_context(
            GpuContext::adopt(device, queue, info),
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
        let (threads, shared_bytes) = self.workgroup_budget();
        PROFILES
            .iter()
            .copied()
            .filter(|profile| profile.fits(threads, shared_bytes))
            .collect()
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
        self.profiles()
            .into_iter()
            .next_back()
            .expect("the device offers no workgroup the framework can schedule")
    }

    pub fn weights(&self, graph: &Graph, precision: Precision) -> Weights<'_> {
        self.context.assert_alive();
        let (weights, layout) = self.parameter_store(graph, precision);
        self.seed(&layout, &weights);
        weights
    }

    pub fn load(
        &self,
        graph: &Graph,
        checkpoint: &Checkpoint,
        precision: Precision,
    ) -> Weights<'_> {
        self.context.assert_alive();
        let (weights, layout) = self.parameter_store(graph, precision);
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

    pub fn checkpoint<'r>(&self, program: &Program<'r>) -> Checkpoint {
        self.assert_owns(program);
        self.context.assert_alive();
        let weights = program.weights();
        let bytes = weights.region().bytes();
        let total = bytes + REFUSAL_BYTES;
        let staging = Recycled::claim(
            &self.pool,
            "neura checkpoint",
            total,
            BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        );
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura checkpoint");
        submission.copy_buffer_to_buffer(
            weights.buffer().buffer(),
            weights.offset(),
            staging.buffer().buffer(),
            0,
            bytes,
        );
        submission.copy_buffer_to_buffer(
            program.refusal.buffer().buffer(),
            0,
            staging.buffer().buffer(),
            bytes,
            REFUSAL_BYTES,
        );
        let (sender, completed) = mpsc::channel();
        submission.map_buffer_on_submit(
            staging.buffer().buffer(),
            MapMode::Read,
            ..total,
            move |result| {
                let _ = sender.send(result);
            },
        );
        let submission = submission.submit(self.context.queue());
        let payload = self.collect_checkpoint(device, staging, submission, completed, bytes, total);
        Checkpoint::of(weights.region(), payload)
    }

    fn collect_checkpoint(
        &self,
        device: &wgpu::Device,
        staging: Recycled,
        submission: wgpu::SubmissionIndex,
        completed: mpsc::Receiver<Result<(), BufferAsyncError>>,
        bytes: u64,
        total: u64,
    ) -> Vec<u8> {
        device
            .poll(PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(CHECKPOINT_TIMEOUT),
            })
            .unwrap_or_else(|error| {
                panic!("waiting for the checkpoint submission failed: {error}")
            });
        completed
            .recv_timeout(CHECKPOINT_TIMEOUT)
            .unwrap_or_else(|error| panic!("the checkpoint callback never fired: {error}"))
            .unwrap_or_else(|error: BufferAsyncError| {
                panic!("mapping the checkpoint failed: {error}")
            });
        let words = staging
            .buffer()
            .buffer()
            .slice(..total)
            .get_mapped_range()
            .expect("a finished checkpoint is mapped");
        let refusal = u32::from_ne_bytes(
            words[bytes as usize..total as usize]
                .try_into()
                .expect("a fault word was copied back"),
        );
        assert_eq!(refusal, 0, "{}", refusal_message(refusal));
        let payload = words[..bytes as usize].to_vec();
        drop(words);
        staging.buffer().buffer().unmap();
        payload
    }

    fn parameter_store(&self, graph: &Graph, precision: Precision) -> (Weights<'_>, Layout) {
        let layout = graph.layout(self.alignment, precision);
        let store = self.heap.allocate(layout.weights().words());
        let weights = Weights::new(store, layout.weights().clone(), precision);
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
                &weights.precision().pack(&values),
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
        let layout = graph.layout(self.alignment, weights.precision());
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
        let encoding = graph.encode(self.alignment, profile, weights.precision());
        assert!(
            encoding.task_count() > 0,
            "a program whose tape holds no task has nothing for the device to run",
        );
        assert!(
            encoding.weights() == weights.region(),
            "this graph holds {} parameters where the weight store carries {}; one store serves every program of one model",
            encoding.weights().tensors(),
            weights.tensors(),
        );
        assert!(
            !(weights.precision().half() && encoding.updates_weights()),
            "a {:?} weight store carries no in-place parameter update; a model that trains holds its weights in {:?}",
            weights.precision(),
            Precision::Single,
        );
        let signature = tape::signature(&encoding, profile, weights.precision(), self.alignment);
        let kinds = encoding.kinds().to_vec();
        let geometry = Geometry::of(profile);
        let precision = weights.precision();
        let kernel = self
            .tapes
            .kernel(kinds.as_slice(), geometry.clone(), precision, || {
                Megakernel::assemble(&kinds, geometry, precision)
            });
        let tape = self.tapes.of(signature, |signature| {
            DeviceTape::build(&self.context, &self.pool, encoding, kernel, signature)
        });
        let tensors = self
            .heap
            .allocate(tape.encoding.tensor_bytes() / WORD_BYTES);
        Program::of(&self.context, tape, tensors, weights.clone())
    }

    pub fn tune<'r>(&'r self, graph: &Graph, weights: &Weights<'r>) -> Program<'r> {
        let scratch = graph
            .updates_weights()
            .then(|| self.scratch_weights(graph, weights.precision()));
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

    fn scratch_weights(&self, graph: &Graph, precision: Precision) -> Weights<'_> {
        self.context.assert_alive();
        let (scratch, layout) = self.parameter_store(graph, precision);
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
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura program");
        submission.clear_buffer(program.refusal.buffer().buffer(), 0, None);
        let mut pass = submission.begin_compute_pass(&ComputePassDescriptor {
            label: Some("neura tape"),
            timestamp_writes: None,
        });
        pass.set_pipeline(program.tape.kernel.pipeline());
        for (index, dispatch) in program.tape.encoding.dispatches().iter().enumerate() {
            pass.set_bind_group(0, &program.group, &[(index as u64 * self.alignment) as u32]);
            pass.dispatch_workgroups(dispatch.segments, 1, 1);
        }
        drop(pass);
        submission.submit(self.context.queue());
    }

    pub fn write(&self, program: &Program<'_>, value: Value<'_>, data: &[f32]) {
        self.assert_owns(program);
        assert!(
            program.readable(value),
            "value {} is a temporary whose storage a later task of the tape reuses; retain it before the run to write it",
            value.id(),
        );
        let span = program.span(value);
        assert_eq!(
            data.len(),
            span.elements as usize,
            "writing {} numbers into a tensor of {} numbers",
            data.len(),
            span.elements,
        );
        let bytes = match span.store {
            Store::Weights => program.weights.precision().pack(data),
            Store::Tensors => bytemuck::cast_slice(data).to_vec(),
        };
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
        for value in values {
            assert!(
                program.readable(*value),
                "value {} is a temporary whose storage a later task of the tape reuses; retain it before the run to pull it",
                value.id(),
            );
        }
        let spans = values
            .iter()
            .map(|value| program.span(*value))
            .collect::<Vec<_>>();
        let precision = program.weights.precision();
        let total = spans
            .iter()
            .map(|span| span_bytes(*span, precision))
            .sum::<u64>()
            + WORD_BYTES;
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
            let bytes = span_bytes(*span, precision);
            submission.copy_buffer_to_buffer(
                program.heap().buffer(),
                span.offset,
                staging.buffer(),
                at,
                bytes,
            );
            collected.push((*span, at, bytes));
            at += bytes;
        }
        submission.copy_buffer_to_buffer(
            program.refusal.buffer().buffer(),
            0,
            staging.buffer(),
            at,
            WORD_BYTES,
        );
        let (sender, completed) = mpsc::channel();
        submission.map_buffer_on_submit(staging.buffer(), MapMode::Read, ..total, move |result| {
            let _ = sender.send(result);
        });
        let submission = submission.submit(self.context.queue());
        Readout {
            brand: PhantomData,
            slot,
            submission,
            completed,
            precision,
            spans: collected,
            total,
            refusal: at,
        }
    }

    pub fn collect(&self, readout: Readout<'_>) -> Vec<Vec<f32>> {
        self.context.assert_alive();
        let bytes = self.readback.finish(
            self.context.device(),
            readout.slot,
            readout.submission,
            readout.completed,
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
                decode(
                    *span,
                    readout.precision,
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
}

fn span_bytes(span: Span, precision: Precision) -> u64 {
    match span.store {
        Store::Tensors => u64::from(span.elements) * WORD_BYTES,
        Store::Weights => precision.words(u64::from(span.elements)) * WORD_BYTES,
    }
}

fn decode(span: Span, precision: Precision, bytes: &[u8]) -> Vec<f32> {
    match span.store {
        Store::Tensors => bytemuck::cast_slice::<u8, f32>(bytes).to_vec(),
        Store::Weights => precision.unpack(span.elements as usize, bytes),
    }
}

fn refusal_message(word: u32) -> String {
    let refused = word >> 16;
    let code = (word & 0xffff) - 1;
    if refused >= Kind::COUNT {
        return format!("the device refused kind {refused} with code {code}");
    }
    let kind = Kind::of(refused);
    match kind {
        Kind::Binary | Kind::Unary => format!(
            "the device refused the {} op of the {} task",
            op::name(code),
            kind.name(),
        ),
        Kind::Partial => format!(
            "the device refused the partial of the {} op over operand {} of the {} task",
            op::name(code / 2),
            code % 2,
            kind.name(),
        ),
        Kind::OneHot | Kind::Gather | Kind::Scatter => format!(
            "the device refused an index outside the rows of the {} task",
            kind.name(),
        ),
        Kind::Matmul
        | Kind::Argmax
        | Kind::Categorical
        | Kind::SumAxis
        | Kind::Conv2dWeightGrad => {
            format!(
                "the device refused geometry {code} of the {} task",
                kind.name(),
            )
        }
        Kind::Fill
        | Kind::Broadcast
        | Kind::SumChunk
        | Kind::Concat
        | Kind::Accumulate
        | Kind::MatmulFold
        | Kind::Softmax
        | Kind::SoftmaxGrad
        | Kind::LogSoftmax
        | Kind::LogSoftmaxGrad
        | Kind::Conv2d
        | Kind::Conv2dInputGrad => {
            format!("the device refused code {code} of the {} task", kind.name())
        }
    }
}
