use neura_abi::{Element, Store, ValueRecord, WORD_BYTES};
use neura_graph::{Graph, Init, Shape};
use neura_plan::{DEFAULT_ENCODING_BYTES, Plan};
use neura_profile::{Budget, Profile};
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

const ALIGNMENT: u64 = 256;

fn narrow() -> Profile {
    Profile::derive(Budget::BASELINE, None)[0]
}

fn chain(rows: u32, free: Option<u32>) -> Graph<'static> {
    let graph = Graph::new();
    let batch = free.map(|bound| graph.free(bound));
    let mut value = match batch {
        Some(batch) => graph.input(
            Shape::matrix(rows, 64).freed(&[(2, batch)]),
            Element::Single,
        ),
        None => graph.input(Shape::matrix(rows, 64), Element::Single),
    };
    for layer in 0..3 {
        let weight = graph.named_parameter(
            &format!("w{layer}"),
            Shape::matrix(64, 64),
            Init::Uniform {
                low: -0.2,
                high: 0.2,
            },
            Element::Single,
        );
        value = graph.relu(graph.matmul(value, weight));
    }
    graph.retain(value);
    graph
}

#[test]
fn a_binding_holds_the_arena_of_the_lengths_it_names() {
    let graph = chain(64, Some(64));
    let plan = Plan::of(&graph, ALIGNMENT, narrow(), DEFAULT_ENCODING_BYTES);
    let bound = plan.encode(&[64]);
    let half = plan.encode(&[32]);
    let one = plan.encode(&[1]);
    let none = plan.encode(&[0]);
    assert_eq!(
        bound.arena_bytes(),
        plan.arena_bytes(),
        "a binding of the bound lengths holds the arena the plan declares for its bound",
    );
    assert!(
        half.arena_bytes() < bound.arena_bytes(),
        "a binding of 32 rows holds {} bytes beside the {} the bound holds",
        half.arena_bytes(),
        bound.arena_bytes(),
    );
    assert!(
        one.arena_bytes() < half.arena_bytes(),
        "a binding of one row holds {} bytes beside the {} 32 rows hold",
        one.arena_bytes(),
        half.arena_bytes(),
    );
    assert!(
        none.arena_bytes() < one.arena_bytes(),
        "a binding of no row holds {} bytes",
        none.arena_bytes(),
    );
}

#[test]
fn every_record_of_a_binding_addresses_that_binding() {
    let graph = chain(64, Some(64));
    let plan = Plan::of(&graph, ALIGNMENT, narrow(), DEFAULT_ENCODING_BYTES);
    for lengths in [[64u32], [7], [1], [0]] {
        let encoding = plan.encode(&lengths);
        let words = encoding.tensor_bytes() / WORD_BYTES;
        let records = encoding.values();
        for id in 0..plan.value_count() {
            let at = id as usize * size_of::<ValueRecord>();
            let record: ValueRecord =
                bytemuck::pod_read_unaligned(&records[at..at + size_of::<ValueRecord>()]);
            if record.store != Store::Tensors.code() || record.storage != id {
                continue;
            }
            if !encoding.holds(id) {
                continue;
            }
            let elements = record.dims.iter().map(|dim| u64::from(*dim)).product();
            let element = Element::of(record.element);
            let reach =
                u64::from(record.base) + element.payload_words(elements) + element.quanta(elements);
            assert!(
                reach <= words,
                "value {id} of a binding of {lengths:?} reaches word {reach} where its tensors hold {words}",
            );
        }
    }
}

#[test]
fn a_static_plan_holds_the_arena_it_derives() {
    let graph = chain(64, None);
    let plan = Plan::of(&graph, ALIGNMENT, narrow(), DEFAULT_ENCODING_BYTES);
    let derived = plan.encode(&[]);
    assert_eq!(derived.values(), plan.values(), "the records differ");
    assert_eq!(derived.tasks(), plan.tasks(), "the task records differ");
    assert_eq!(
        derived.tensor_bytes(),
        plan.tensor_bytes(),
        "a plan of one shape derives the tensors it declares",
    );
}

fn training(rows: u32, free: Option<u32>) -> Graph<'static> {
    let graph = Graph::new();
    let (observations, targets) = match free {
        Some(bound) => {
            let batch = graph.free(bound);
            let shape = Shape::matrix(bound, 64).freed(&[(2, batch)]);
            (
                graph.input(shape, Element::Single),
                graph.input(shape, Element::Single),
            )
        }
        None => (
            graph.input(Shape::matrix(rows, 64), Element::Single),
            graph.input(Shape::matrix(rows, 64), Element::Single),
        ),
    };
    let mut carried = observations;
    let mut parameters = Vec::new();
    for layer in 0..3 {
        let weight = graph.named_parameter(
            &format!("w{layer}"),
            Shape::matrix(64, 64),
            Init::Uniform {
                low: -0.2,
                high: 0.2,
            },
            Element::Single,
        );
        let bias = graph.named_parameter(
            &format!("b{layer}"),
            Shape::matrix(1, 64),
            Init::Zero,
            Element::Single,
        );
        parameters.push(weight);
        parameters.push(bias);
        carried = graph.relu(graph.add(graph.matmul(carried, weight), bias));
    }
    let difference = graph.sub(carried, targets);
    let loss = graph.sum(graph.mul(difference, difference));
    let gradients = graph.backward(loss);
    let rate = graph.fill(Shape::scalar(), -0.005);
    for parameter in parameters {
        graph.add_into(parameter, graph.mul(gradients.of(parameter), rate));
    }
    graph.retain(loss);
    graph
}

#[test]
fn a_free_plan_schedules_its_bound_length_the_way_a_static_plan_does() {
    let free = Plan::of(
        &training(128, Some(128)),
        ALIGNMENT,
        narrow(),
        DEFAULT_ENCODING_BYTES,
    );
    let fixed = Plan::of(
        &training(128, None),
        ALIGNMENT,
        narrow(),
        DEFAULT_ENCODING_BYTES,
    );
    assert_eq!(
        free.task_count(),
        fixed.task_count(),
        "a free plan of a bound of 128 rows schedules {} tasks where the static plan schedules {}",
        free.task_count(),
        fixed.task_count(),
    );
    assert_eq!(
        free.wave_count(),
        fixed.wave_count(),
        "a free plan of a bound of 128 rows gates {} waves where the static plan gates {}; the schedule of a binding walks the lengths that binding names",
        free.wave_count(),
        fixed.wave_count(),
    );
    assert_eq!(
        free.segments().len(),
        fixed.segments().len(),
        "a free plan of a bound of 128 rows dispatches {} segments where the static plan dispatches {}",
        free.segments().len(),
        fixed.segments().len(),
    );
    assert_eq!(
        free.encode(&[128]).wave_count(),
        fixed.wave_count(),
        "a binding of the declared bound gates the waves the static plan of that shape gates",
    );
}

#[test]
fn a_shorter_binding_walks_the_tasks_its_lengths_open() {
    let plan = Plan::of(
        &training(128, Some(128)),
        ALIGNMENT,
        narrow(),
        DEFAULT_ENCODING_BYTES,
    );
    let bound = plan.encode(&[128]);
    let reserved = plan.tensor_bytes();
    let mut tasks = bound.task_count();
    for length in [64u32, 16, 8, 1, 0] {
        let encoding = plan.encode(&[length]);
        assert!(
            encoding.task_count() <= tasks,
            "a binding of {length} rows walks {} tasks where the binding before it walks {tasks}",
            encoding.task_count(),
        );
        assert!(
            encoding.task_count() < bound.task_count(),
            "a binding of {length} rows walks every one of the {} tasks the bound of 128 rows walks",
            bound.task_count(),
        );
        assert!(
            encoding.wave_count() <= bound.wave_count(),
            "a binding of {length} rows gates {} waves where the bound gates {}",
            encoding.wave_count(),
            bound.wave_count(),
        );
        assert!(
            encoding.tensor_bytes() <= reserved,
            "a binding of {length} rows holds {} bytes where the bound holds {reserved}",
            encoding.tensor_bytes(),
        );
        tasks = encoding.task_count();
    }
    assert_eq!(
        bound.task_count(),
        plan.task_count(),
        "the binding of the declared bound walks every task the plan carries",
    );
    let one = plan.encode(&[1]);
    assert!(
        one.task_count() < bound.task_count(),
        "a binding of one row walks {} tasks where the bound of 128 rows walks {}; a binding walks the tasks its own lengths open",
        one.task_count(),
        bound.task_count(),
    );
}

#[test]
fn a_plan_remembers_the_encodings_of_the_shapes_it_has_walked() {
    let graph = training(128, Some(128));
    let plan = Plan::of(&graph, ALIGNMENT, narrow(), DEFAULT_ENCODING_BYTES);
    assert_eq!(
        plan.derived_encodings(),
        0,
        "a plan derives the encoding of the bound lengths at compile, not at a walk",
    );
    let walked = plan.encode(&[64]);
    assert_eq!(plan.derived_encodings(), 1);
    assert_eq!(plan.remembered_encodings(), 1);
    let again = plan.encode(&[64]);
    assert!(
        Arc::ptr_eq(&walked, &again),
        "a plan answers a shape it has walked with the encoding it derived then",
    );
    assert_eq!(
        plan.derived_encodings(),
        1,
        "a shape a plan has walked is not derived again",
    );
    assert!(
        plan.remembered_bytes() <= DEFAULT_ENCODING_BYTES,
        "a plan remembers {} bytes where its budget is {DEFAULT_ENCODING_BYTES}",
        plan.remembered_bytes(),
    );
    let other = plan.encode(&[16]);
    assert!(!Arc::ptr_eq(&walked, &other));
    assert_eq!(plan.remembered_encodings(), 2);
    let bound = plan.encode(&[128]);
    assert!(
        Arc::ptr_eq(&bound, &plan.bound_encoding()),
        "the bound lengths are the encoding the plan itself carries",
    );
    assert_eq!(
        plan.remembered_encodings(),
        2,
        "the bound lengths stand in the plan and take no room of the shapes it remembers",
    );
}

fn encode_millis(plan: &Plan, lengths: &[u32], rounds: u32) -> f64 {
    let mut fastest = f64::MAX;
    for _ in 0..rounds {
        let started = Instant::now();
        let _ = plan.encode(lengths);
        fastest = fastest.min(started.elapsed().as_secs_f64() * 1000.0);
    }
    fastest
}

#[test]
fn a_shape_a_plan_has_walked_costs_no_second_derive() {
    let graph = training(128, Some(128));
    let plan = Plan::of(&graph, ALIGNMENT, narrow(), DEFAULT_ENCODING_BYTES);
    let walked = encode_millis(&plan, &[64], 1);
    let fresh = (0..4u32)
        .map(|at| encode_millis(&plan, &[96 + at], 1))
        .fold(f64::MAX, f64::min);
    let again = encode_millis(&plan, &[64], 64);
    assert!(
        again * 20.0 < fresh,
        "a shape of 64 rows the plan has walked costs {again:.4} ms beside the {fresh:.4} ms of a shape it has not, and a plan answers a shape it has walked without planning it again",
    );
    assert!(
        again * 20.0 < walked,
        "walking a shape the plan has walked costs {again:.4} ms beside the {walked:.4} ms its first walk costs",
    );
}
