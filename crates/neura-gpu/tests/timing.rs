use neura_gpu::{Backends, BufferUsages, GpuBuffer, GpuContext, GpuRequest, Submission};

fn context(backends: Backends) -> GpuContext {
    pollster::block_on(GpuContext::open(&GpuRequest {
        backends,
        ..Default::default()
    }))
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

fn copy_seconds(context: &GpuContext, bytes: u64) -> f64 {
    let device = context.device();
    let queue = context.queue();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let source = GpuBuffer::new(device, "clock source", bytes, usage);
    let target = GpuBuffer::new(device, "clock target", bytes, usage);
    let mut fastest = f64::MAX;
    for round in 0..12 {
        let mut submission = Submission::new(device, "clock copy");
        submission.copy(&source, 0, &target, 0, bytes);
        let seconds = queue.seconds(submission.submit(queue));
        if round >= 4 {
            fastest = fastest.min(seconds);
        }
    }
    fastest
}

#[test]
fn every_platform_backend_times_the_copy_it_ran() {
    for backends in Backends::PLATFORM {
        let backend = backends.backend();
        let context = context(backends);
        let heavy = copy_seconds(&context, 64 << 20);
        let light = copy_seconds(&context, 4 << 10);
        assert!(
            heavy > 0.0 && light > 0.0,
            "{backend:?} timed a 64 MiB copy at {heavy} seconds and a 4 KiB copy at {light}",
        );
        assert!(
            heavy >= light,
            "{backend:?} timed a 64 MiB copy at {heavy} seconds and a 4 KiB copy at {light}",
        );
    }
}
