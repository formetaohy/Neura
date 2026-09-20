use crate::program::Program;
use neura_abi::{
    CURSOR_REFUSED, KIND_COUNT, REFUSED_CHAIN, SCHEDULES, Schedule, WORD_BYTES, slot_offset,
};
use neura_gpu::{
    ComputePassDescriptor, GpuContext, GpuRequest, GpuUnavailable, Readback, Submission, wgpu,
};
use neura_program::{Graph, Value};
use std::time::Instant;

pub const WORKGROUP_BUDGET: u32 = 1024;
pub const DEFAULT_READBACK_BYTES: u64 = 1 << 20;
const TUNE_WARMUP: u32 = 2;
const TUNE_ROUNDS: u32 = 8;

pub struct RuntimeRequest {
    pub gpu: GpuRequest,
    pub readback_bytes: u64,
}

impl Default for RuntimeRequest {
    fn default() -> Self {
        Self {
            gpu: GpuRequest::default(),
            readback_bytes: DEFAULT_READBACK_BYTES,
        }
    }
}

pub struct Runtime {
    context: GpuContext,
    readback: Readback,
    alignment: u64,
}

impl Runtime {
    pub async fn open(request: RuntimeRequest) -> Result<Self, GpuUnavailable> {
        let context = GpuContext::open(&request.gpu).await?;
        Ok(Self::of_context(context, request.readback_bytes))
    }

    pub fn adopt(device: wgpu::Device, queue: wgpu::Queue, info: wgpu::AdapterInfo) -> Self {
        Self::of_context(
            GpuContext::adopt(device, queue, info),
            DEFAULT_READBACK_BYTES,
        )
    }

    fn of_context(context: GpuContext, readback_bytes: u64) -> Self {
        Self {
            alignment: context.binding_alignment(),
            readback: Readback::new(context.device(), readback_bytes),
            context,
        }
    }

    pub fn schedules(&self) -> Vec<Schedule> {
        let (threads, shared_bytes) = self.workgroup_budget();
        SCHEDULES
            .iter()
            .copied()
            .filter(|schedule| schedule.fits(threads, shared_bytes))
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

    pub fn default_schedule(&self) -> Schedule {
        self.schedules()
            .into_iter()
            .next_back()
            .expect("the device offers no workgroup the framework can schedule")
    }

    pub fn compile(&self, graph: &Graph) -> Program {
        self.compile_with(graph, self.default_schedule())
    }

    pub fn compile_with(&self, graph: &Graph, schedule: Schedule) -> Program {
        let (threads, shared_bytes) = self.workgroup_budget();
        assert!(
            schedule.fits(threads, shared_bytes),
            "{schedule:?} asks the device for {} threads and {} workgroup bytes, while it offers {threads} and {shared_bytes}",
            schedule.workgroup(),
            schedule.shared_bytes(),
        );
        self.context.assert_alive();
        let encoding = graph.encode(self.alignment, schedule);
        assert!(
            encoding.task_count() > 0,
            "a program whose tape holds no task has nothing for the device to run",
        );
        Program::build(&self.context, encoding)
    }

    pub fn tune(&self, graph: &Graph) -> Program {
        let mut measured = self
            .schedules()
            .into_iter()
            .map(|schedule| (schedule, self.measure(&self.compile_with(graph, schedule))));
        let (mut fastest, mut seconds) = measured
            .next()
            .expect("the device offers no workgroup the framework can schedule");
        for (schedule, elapsed) in measured {
            if elapsed < seconds {
                fastest = schedule;
                seconds = elapsed;
            }
        }
        self.compile_with(graph, fastest)
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
        program.arena.write_at(
            self.context.queue(),
            span.offset,
            bytemuck::cast_slice(data),
        );
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
        let words = spans
            .iter()
            .map(|span| u64::from(span.elements))
            .sum::<u64>();
        let total = words * WORD_BYTES + WORD_BYTES;
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
            let bytes = u64::from(span.elements) * WORD_BYTES;
            submission.copy_buffer_to_buffer(
                program.arena.buffer(),
                span.offset,
                self.readback.staging().buffer(),
                at,
                bytes,
            );
            collected.push((at, bytes));
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
            .map(|(offset, length)| {
                let start = *offset as usize;
                bytemuck::cast_slice::<u8, f32>(&bytes[start..start + *length as usize]).to_vec()
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

fn refusal_message(word: u32) -> String {
    let kind = word >> 16;
    let code = (word & 0xffff) - 1;
    if kind == REFUSED_CHAIN {
        return format!("the device refused epilogue op {code}");
    }
    if kind < KIND_COUNT {
        format!(
            "the device refused op code {code} of the {} task",
            neura_abi::kind_name(kind),
        )
    } else {
        format!("the device refused task kind {kind} with op code {code}")
    }
}
