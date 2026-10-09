use neura_abi::Element;
use neura_graph::{Graph, Shape};
use neura_runtime::{MemoryRequest, Program, Readout, Run, Runtime, RuntimeRequest, Weights};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::thread;

fn assert_static<T: Send + Sync + 'static>() {}

fn open(memory: MemoryRequest) -> Runtime {
    Runtime::open(RuntimeRequest {
        memory,
        ..Default::default()
    })
    .unwrap_or_else(|error| panic!("no device runs the tests: {error}"))
}

#[test]
fn a_model_is_static_and_rides_to_another_thread_of_the_frame() {
    assert_static::<Runtime>();
    assert_static::<Weights>();
    assert_static::<Program>();
    assert_static::<Readout>();
    assert_static::<Run>();

    let runtime = open(MemoryRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    });
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let out = graph.mul(data, graph.fill(Shape::vector(4), 2.0));
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);

    thread::scope(|scope| {
        let agent = scope.spawn(|| {
            runtime.write(&program, data, &[1.0, 2.0, 3.0, 4.0]);
            runtime.run(&program);
            runtime.read(&program, out)
        });
        let values = agent.join().expect("an agent thread drives its own model");
        assert_eq!(values, vec![2.0, 4.0, 6.0, 8.0]);
    });
}

#[test]
fn a_frame_pulls_every_agent_before_it_collects_any() {
    let slots = 4;
    let runtime = open(MemoryRequest {
        readback_bytes: 1 << 16,
        readback_slots: slots,
        ..Default::default()
    });
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let out = graph.mul(data, graph.fill(Shape::vector(4), 2.0));
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let programs = (0..slots)
        .map(|_| runtime.compile(&graph, &weights))
        .collect::<Vec<_>>();

    let mut readouts = Vec::new();
    for (agent, program) in programs.iter().enumerate() {
        let value = agent as f32 + 1.0;
        runtime.write(program, data, &[value; 4]);
        runtime.run(program);
        readouts.push(runtime.pull(program, &[out]));
    }
    assert_eq!(readouts.len(), runtime.readback_slots());

    for (agent, readout) in readouts.into_iter().enumerate() {
        let value = 2.0 * (agent as f32 + 1.0);
        assert_eq!(
            readout.collect().pop().expect("one tensor came back"),
            vec![value; 4],
        );
    }
}

#[test]
fn a_readout_the_frame_never_collects_returns_its_slot() {
    let slots = 2;
    let runtime = open(MemoryRequest {
        readback_bytes: 1 << 16,
        readback_slots: slots,
        ..Default::default()
    });
    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let out = graph.mul(data, graph.fill(Shape::vector(4), 2.0));
    graph.retain(out);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);

    for round in 0..(slots as u32 * 4) {
        runtime.write(&program, data, &[round as f32; 4]);
        runtime.run(&program);
        let readout = runtime.pull(&program, &[out]);
        if round.is_multiple_of(2) {
            drop(readout);
            continue;
        }
        assert_eq!(
            readout.collect().pop().expect("one tensor came back"),
            vec![2.0 * round as f32; 4],
            "round {round} of a frame that leaves every other readback behind reads the tensor it pulled",
        );
    }
    assert_eq!(runtime.readback_slots(), slots);
}

#[test]
fn a_refusal_a_dropped_readout_carries_reaches_the_next_readback() {
    let runtime = open(MemoryRequest {
        readback_bytes: 1 << 16,
        ..Default::default()
    });
    let graph = Graph::new();
    let table = graph.input(Shape::matrix(4, 2), Element::Single);
    let indices = graph.input(Shape::vector(1), Element::Single);
    let gathered = graph.gather(table, indices);
    graph.retain(gathered);
    let weights = runtime.weights(&graph);
    let program = runtime.compile(&graph, &weights);
    runtime.write(&program, table, &[1.0; 8]);
    runtime.write(&program, indices, &[7.0]);
    runtime.run(&program);

    drop(runtime.pull(&program, &[gathered]));
    let refusal = catch_unwind(AssertUnwindSafe(|| {
        let _ = runtime.pull(&program, &[gathered]).collect();
    }))
    .expect_err("the refusal a dropped readback carried is reported by the next one");
    let message = refusal
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_else(|| "a refusal without a message".to_owned());
    assert!(message.contains("refused an index"), "{message}");
}

#[test]
fn another_runtime_of_the_same_device_refuses_this_model() {
    let first = open(MemoryRequest::default());
    let second = Runtime::from_device(
        first.context().device().clone(),
        MemoryRequest {
            readback_bytes: 1 << 12,
            readback_slots: 3,
            ..Default::default()
        },
    );
    assert_eq!(second.readback_slots(), 3);
    assert_eq!(second.readback_capacity(), 1 << 12);

    let graph = Graph::new();
    let data = graph.input(Shape::vector(4), Element::Single);
    let out = graph.mul(data, graph.fill(Shape::vector(4), 1.0));
    graph.retain(out);
    let weights = first.weights(&graph);
    let program = first.compile(&graph, &weights);

    assert!(
        catch_unwind(AssertUnwindSafe(|| second.run(&program))).is_err(),
        "a second runtime ran the program of the first",
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| second.compile(&graph, &weights))).is_err(),
        "a second runtime compiled the weights of the first",
    );
}
