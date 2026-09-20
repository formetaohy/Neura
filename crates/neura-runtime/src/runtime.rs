use crate::program::Program;
use neura_abi::{BoundsRecord, CURSOR_BYTES, CURSOR_REFUSED, KIND_COUNT, WORD_BYTES, slot_offset};
use neura_gpu::{
    BindGroupEntry, BufferUsages, ComputePassDescriptor, GpuBuffer, GpuContext, GpuRequest,
    GpuUnavailable, PipelineHandle, Readback, Submission, wgpu,
};
use neura_program::{Graph, Value};
use neura_shader::{ARENA, BOUNDS, CURSOR, Megakernel, TASKS, VALUES};
use std::mem::size_of;

pub const WORKGROUP_BUDGET: u32 = 1024;
pub const DEFAULT_ARENA_BYTES: u64 = 64 << 20;
pub const DEFAULT_READBACK_BYTES: u64 = 1 << 20;
const MINIMUM_ARENA_BYTES: u64 = 4096;

pub struct RuntimeRequest {
    pub gpu: GpuRequest,
    pub arena_bytes: u64,
    pub readback_bytes: u64,
}

impl Default for RuntimeRequest {
    fn default() -> Self {
        Self {
            gpu: GpuRequest::default(),
            arena_bytes: DEFAULT_ARENA_BYTES,
            readback_bytes: DEFAULT_READBACK_BYTES,
        }
    }
}

pub struct Runtime {
    context: GpuContext,
    arena: GpuBuffer,
    kernel: PipelineHandle,
    readback: Readback,
    alignment: u64,
    capacity: u64,
}

impl Runtime {
    pub async fn open(request: RuntimeRequest) -> Result<Self, GpuUnavailable> {
        let context = GpuContext::open(&request.gpu).await?;
        Ok(Self::of_context(
            context,
            request.arena_bytes,
            request.readback_bytes,
        ))
    }

    pub fn adopt(
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
        arena_bytes: u64,
    ) -> Self {
        Self::of_context(
            GpuContext::adopt(device, queue, info),
            arena_bytes,
            DEFAULT_READBACK_BYTES,
        )
    }

    fn of_context(context: GpuContext, arena_bytes: u64, readback_bytes: u64) -> Self {
        assert!(
            arena_bytes >= MINIMUM_ARENA_BYTES && arena_bytes.is_multiple_of(WORD_BYTES),
            "an arena of {arena_bytes} bytes is below the {MINIMUM_ARENA_BYTES} byte floor or off the word grid",
        );
        let limits = context.limits();
        assert!(
            arena_bytes <= limits.max_storage_buffer_binding_size,
            "the device binds at most {} bytes of storage, so an arena of {arena_bytes} bytes cannot be bound",
            limits.max_storage_buffer_binding_size,
        );
        assert!(
            arena_bytes <= limits.max_buffer_size,
            "the device holds buffers of at most {} bytes, so an arena of {arena_bytes} bytes cannot be created",
            limits.max_buffer_size,
        );
        let arena = GpuBuffer::new(
            context.device(),
            "neura arena",
            arena_bytes,
            BufferUsages::STORAGE
                | BufferUsages::COPY_SRC
                | BufferUsages::COPY_DST
                | BufferUsages::VERTEX
                | BufferUsages::INDIRECT,
        );
        let kernel = context.declare(Megakernel::assemble().program());
        Self {
            alignment: context.binding_alignment(),
            readback: Readback::new(context.device(), readback_bytes),
            context,
            arena,
            kernel,
            capacity: arena_bytes,
        }
    }

    pub fn compile(&self, graph: &Graph) -> Program {
        self.context.assert_alive();
        let encoding = graph.encode(self.alignment, self.capacity);
        assert!(
            encoding.task_count() > 0,
            "a program whose tape holds no task has nothing for the device to run",
        );
        let device = self.context.device();
        let tape = GpuBuffer::new(
            device,
            "neura tape",
            encoding.tasks().len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let values = GpuBuffer::new(
            device,
            "neura values",
            encoding.values().len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let bounds = GpuBuffer::new(
            device,
            "neura bounds",
            encoding.bounds().len() as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_DST,
        );
        let cursor = GpuBuffer::new(
            device,
            "neura cursor",
            CURSOR_BYTES,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        );
        let queue = self.context.queue();
        tape.write(queue, encoding.tasks());
        values.write(queue, encoding.values());
        bounds.write(queue, encoding.bounds());
        let mut submission = Submission::new(device, "neura compile");
        submission.clear_buffer(self.arena.buffer(), 0, None);
        submission.submit(queue);
        for (offset, data) in encoding.initial() {
            self.arena
                .write_at(queue, *offset, bytemuck::cast_slice(data));
        }
        let group = self.kernel.bind_group(&[
            BindGroupEntry {
                binding: TASKS,
                resource: tape.resource(0, tape.size()),
            },
            BindGroupEntry {
                binding: VALUES,
                resource: values.resource(0, values.size()),
            },
            BindGroupEntry {
                binding: ARENA,
                resource: self.arena.resource(0, self.arena.size()),
            },
            BindGroupEntry {
                binding: CURSOR,
                resource: cursor.resource(0, cursor.size()),
            },
            BindGroupEntry {
                binding: BOUNDS,
                resource: bounds.resource(0, size_of::<BoundsRecord>() as u64),
            },
        ]);
        Program {
            encoding,
            group,
            cursor,
        }
    }

    pub fn run(&self, program: &Program) {
        self.context.assert_alive();
        let device = self.context.device();
        let queue = self.context.queue();
        let waves = program.encoding.waves();
        let mut submission = Submission::new(device, "neura program");
        submission.clear_buffer(program.cursor.buffer(), 0, None);
        let mut first = 0;
        for (index, end) in waves.iter().enumerate() {
            let grid = (end - first).min(WORKGROUP_BUDGET);
            let offset = (index as u64 * self.alignment) as u32;
            let mut pass = submission.begin_compute_pass(&ComputePassDescriptor {
                label: Some("neura wave"),
                timestamp_writes: None,
            });
            pass.set_pipeline(self.kernel.pipeline());
            pass.set_bind_group(0, &program.group, &[offset]);
            pass.dispatch_workgroups(grid, 1, 1);
            first = *end;
        }
        submission.submit(queue);
    }

    pub fn write(&self, program: &Program, value: Value, data: &[f32]) {
        let span = program.span(value);
        assert_eq!(
            data.len(),
            span.elements as usize,
            "writing {} numbers into a tensor of {} numbers",
            data.len(),
            span.elements,
        );
        self.arena.write_at(
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
                program.encoding.readable(*value),
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
                self.arena.buffer(),
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

    pub fn arena(&self) -> &GpuBuffer {
        &self.arena
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    pub fn readback_capacity(&self) -> u64 {
        self.readback.capacity()
    }

    pub fn context(&self) -> &GpuContext {
        &self.context
    }

    pub fn declared_kernels(&self) -> usize {
        self.context.declared_kernels()
    }
}

fn refusal_message(word: u32) -> String {
    let kind = word >> 16;
    let code = (word & 0xffff) - 1;
    if kind < KIND_COUNT {
        format!(
            "the device refused op code {code} of the {} task",
            neura_abi::kind_name(kind),
        )
    } else {
        format!("the device refused task kind {kind} with op code {code}")
    }
}
