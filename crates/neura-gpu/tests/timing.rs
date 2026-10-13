use neura_gpu::{Backend, BufferUsages, GpuBuffer, GpuContext, GpuRequest, PREFERENCE, Submission};

fn context(backend: Backend) -> GpuContext {
    GpuContext::open(&GpuRequest {
        backend: Some(backend),
        ..Default::default()
    })
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

fn pair_seconds(context: &GpuContext) -> (f64, f64) {
    let device = context.device();
    let queue = context.queue();
    let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
    let heavy_source = GpuBuffer::new(device, "heavy source", 64 << 20, usage);
    let heavy_target = GpuBuffer::new(device, "heavy target", 64 << 20, usage);
    let light_source = GpuBuffer::new(device, "light source", 4 << 10, usage);
    let light_target = GpuBuffer::new(device, "light target", 4 << 10, usage);
    let mut heavy = f64::MAX;
    let mut light = f64::MAX;
    for _ in 0..12 {
        let mut submission = Submission::new(device, "heavy copy");
        submission.copy(&heavy_source, 0, &heavy_target, 0, 64 << 20);
        let heavy_index = submission.submit(queue);
        let mut submission = Submission::new(device, "light copy");
        submission.copy(&light_source, 0, &light_target, 0, 4 << 10);
        let light_index = submission.submit(queue);
        heavy = heavy.min(queue.seconds(heavy_index));
        light = light.min(queue.seconds(light_index));
    }
    (heavy, light)
}

#[test]
fn every_platform_backend_times_the_copy_it_ran() {
    for &backend in PREFERENCE {
        let context = context(backend);
        let heavy = copy_seconds(&context, 64 << 20);
        let light = copy_seconds(&context, 4 << 10);
        assert!(
            heavy > 0.0 && light > 0.0,
            "{backend:?} timed a 64 MiB copy at {heavy} seconds and a 4 KiB copy at {light}",
        );
        assert!(
            heavy > 4.0 * light,
            "{backend:?} timed a 64 MiB copy at {heavy} seconds beside the {light} seconds of a 4 KiB copy, and a submission reports the copy it ran",
        );
    }
}

#[test]
fn every_platform_backend_times_a_copy_beside_a_pending_one() {
    for &backend in PREFERENCE {
        let context = context(backend);
        let (heavy, light) = pair_seconds(&context);
        assert!(
            heavy > 4.0 * light,
            "{backend:?} timed a 64 MiB copy beside a pending 4 KiB copy at {heavy} against {light} seconds, and a queue reports the submission that holds each timestamp",
        );
    }
}

#[test]
fn every_platform_backend_reports_a_completed_copy() {
    for &backend in PREFERENCE {
        let context = context(backend);
        let device = context.device();
        let queue = context.queue();
        let usage = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let source = GpuBuffer::new(device, "completion source", 64 << 20, usage);
        let target = GpuBuffer::new(device, "completion target", 64 << 20, usage);
        let mut submission = Submission::new(device, "completion copy");
        submission.copy(&source, 0, &target, 0, 64 << 20);
        let index = submission.submit(queue);
        queue.wait(index, neura_gpu::READBACK_TIMEOUT);
        assert!(
            queue.complete(index),
            "{backend:?} reports a submission it waited on as unfinished",
        );
        assert!(
            queue.complete(index),
            "{backend:?} forgets a submission it has already reported",
        );
    }
}
