use neura_abi::{Element, Store, ValueRecord, WORD_BYTES};
use neura_graph::{Graph, Init, Shape};
use neura_plan::Plan;
use neura_profile::{Budget, Profile};
use std::mem::size_of;

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
    let plan = Plan::of(&graph, ALIGNMENT, narrow());
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
    let plan = Plan::of(&graph, ALIGNMENT, narrow());
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
    let plan = Plan::of(&graph, ALIGNMENT, narrow());
    let derived = plan.encode(&[]);
    assert_eq!(derived.values(), plan.values(), "the records differ");
    assert_eq!(derived.tasks(), plan.tasks(), "the task records differ");
    assert_eq!(
        derived.tensor_bytes(),
        plan.tensor_bytes(),
        "a plan of one shape derives the tensors it declares",
    );
}
