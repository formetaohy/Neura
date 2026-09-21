use crate::heap::Heap;
use crate::program::{Program, Span, Weights};
use neura_abi::{
    CURSOR_REFUSED, PROFILES, Placement, Precision, Profile, WORD_BYTES, kind, op, slot_offset,
};
use neura_gpu::{
    ComputePassDescriptor, GpuContext, GpuRequest, GpuUnavailable, Readback, Submission, wgpu,
};
use neura_program::{Graph, Store, Value};
use std::sync::Arc;
use std::time::Instant;

pub const WORKGROUP_BUDGET: u32 = 1024;
pub const DEFAULT_READBACK_BYTES: u64 = 1 << 20;
pub const DEFAULT_HEAP_BYTES: u64 = 16 << 20;
const TUNE_WARMUP: u32 = 2;
const TUNE_ROUNDS: u32 = 8;

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
        Self {
            alignment: context.binding_alignment(),
            readback: Readback::new(context.device(), readback_bytes),
            heap: Arc::new(Heap::new(&context, heap_bytes)),
            context,
        }
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

    pub fn weights(&self, graph: &Graph, precision: Precision) -> Weights {
        self.context.assert_alive();
        let layout = graph.layout(self.alignment, precision);
        let block = self.heap.reserve(layout.weights().words());
        let placement = Placement::new(self.heap.words(), block.word, 0);
        let queue = self.context.queue();
        for (address, data) in layout.uploads() {
            self.heap.buffer().write_at(
                queue,
                layout.weight_bytes(placement, *address),
                &precision.pack(data),
            );
        }
        Weights::new(
            self.heap.clone(),
            block,
            layout.weights().clone(),
            precision,
        )
    }

    pub fn compile(&self, graph: &Graph, weights: &Weights) -> Program {
        self.compile_with(graph, weights, self.default_profile())
    }

    pub fn compile_with(&self, graph: &Graph, weights: &Weights, profile: Profile) -> Program {
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
        let weights_at = weights.offset() / WORD_BYTES;
        let sized = graph.encode(
            self.alignment,
            profile,
            weights.precision(),
            Placement::new(self.heap.words(), weights_at, 0),
        );
        assert!(
            sized.task_count() > 0,
            "a program whose tape holds no task has nothing for the device to run",
        );
        assert!(
            sized.weights() == weights.region(),
            "this graph holds {} parameters where the weight store carries {}; one store serves every program of one model",
            sized.weights().tensors(),
            weights.tensors(),
        );
        assert!(
            !(weights.precision().half() && sized.updates_weights()),
            "a {:?} weight store carries no in-place parameter update; a model that trains holds its weights in {:?}",
            weights.precision(),
            Precision::Single,
        );
        let tensors = self.heap.reserve(sized.tensor_bytes() / WORD_BYTES);
        let placement = Placement::new(self.heap.words(), weights_at, tensors.word);
        let encoding = graph.encode(self.alignment, profile, weights.precision(), placement);
        assert!(
            encoding.tensor_bytes() <= tensors.words * WORD_BYTES,
            "a second plan of {} tensor bytes outruns the {} bytes the first one asked for",
            encoding.tensor_bytes(),
            tensors.words * WORD_BYTES,
        );
        let tensors = self
            .heap
            .shrink(tensors, encoding.tensor_bytes() / WORD_BYTES);
        Program::build(
            &self.context,
            encoding,
            self.heap.clone(),
            tensors,
            weights.clone(),
        )
    }

    pub fn tune(&self, graph: &Graph, weights: &Weights) -> Program {
        let mut measured = self.profiles().into_iter().map(|profile| {
            (
                profile,
                self.measure(&self.compile_with(graph, weights, profile)),
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

    fn measure(&self, program: &Program) -> f64 {
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

    pub fn run(&self, program: &Program) {
        self.context.assert_alive();
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura program");
        submission.clear_buffer(program.cursor.buffer(), 0, None);
        let mut pass = submission.begin_compute_pass(&ComputePassDescriptor {
            label: Some("neura tape"),
            timestamp_writes: None,
        });
        pass.set_pipeline(program.kernel.pipeline());
        let waves = program.encoding.waves();
        let mut first = 0;
        for (index, end) in waves.iter().enumerate() {
            pass.set_bind_group(0, &program.group, &[(index as u64 * self.alignment) as u32]);
            pass.dispatch_workgroups((end - first).min(WORKGROUP_BUDGET), 1, 1);
            first = *end;
        }
        drop(pass);
        submission.submit(self.context.queue());
    }

    pub fn write(&self, program: &Program, value: Value, data: &[f32]) {
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
            .heap
            .buffer()
            .write_at(self.context.queue(), span.offset, &bytes);
    }

    pub fn read(&self, program: &Program, value: Value) -> Vec<f32> {
        let mut values = self.read_many(program, &[value]);
        values.pop().expect("one tensor was read")
    }

    pub fn read_many(&self, program: &Program, values: &[Value]) -> Vec<Vec<f32>> {
        self.context.assert_alive();
        assert!(!values.is_empty(), "a read names at least one tensor");
        for value in values {
            assert!(
                program.readable(*value),
                "value {} is a temporary whose storage a later task of the tape reuses; retain it before the run to read it back",
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
            "reading {total} bytes outruns the {} byte staging buffer of this runtime",
            self.readback.capacity(),
        );
        let device = self.context.device();
        let mut submission = Submission::new(device, "neura read");
        let mut collected = Vec::with_capacity(spans.len());
        let mut at = 0;
        for span in &spans {
            let bytes = span_bytes(*span, precision);
            submission.copy_buffer_to_buffer(
                program.heap.buffer().buffer(),
                span.offset,
                self.readback.staging().buffer(),
                at,
                bytes,
            );
            collected.push((*span, at, bytes));
            at += bytes;
        }
        submission.copy_buffer_to_buffer(
            program.cursor.buffer(),
            slot_offset(CURSOR_REFUSED),
            self.readback.staging().buffer(),
            at,
            WORD_BYTES,
        );
        let bytes = self
            .readback
            .collect(device, self.context.queue(), submission, total);
        let refusal = u32::from_ne_bytes(
            bytes[at as usize..(at + WORD_BYTES) as usize]
                .try_into()
                .expect("a word was copied back"),
        );
        assert_eq!(refusal, 0, "{}", refusal_message(refusal));
        collected
            .iter()
            .map(|(span, offset, length)| {
                let start = *offset as usize;
                decode(*span, precision, &bytes[start..start + *length as usize])
            })
            .collect()
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

    pub fn declared_kernels(&self) -> usize {
        self.context.declared_kernels()
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
    if refused >= kind::COUNT {
        return format!("the device refused kind {refused} with code {code}");
    }
    match refused {
        kind::BINARY | kind::UNARY => format!(
            "the device refused the {} op of the {} task",
            op::name(code),
            kind::name(refused),
        ),
        kind::PARTIAL => format!(
            "the device refused the partial of the {} op over operand {} of the {} task",
            op::name(code / 2),
            code % 2,
            kind::name(refused),
        ),
        _ => format!(
            "the device refused code {code} of the {} task",
            kind::name(refused),
        ),
    }
}
