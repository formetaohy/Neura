use neura_abi::Element;
use neura_graph::{Graph, Init, Shape};
use neura_plan::Plan;
use neura_profile::{Budget, Profile};
use std::time::Instant;

const ALIGNMENT: u64 = 256;

fn narrow() -> Profile {
    Profile::derive(Budget::BASELINE, None)[0]
}

fn chain(rounds: usize) -> Graph<'static> {
    let graph = Graph::new();
    let state = graph.state(Shape::of([1, 1, 1, 4096]), Init::Zero, Element::Single);
    let one = graph.fill(Shape::of([1, 1, 1, 4096]), 0.5);
    for _ in 0..rounds {
        graph.add_into(state, one);
    }
    graph.retain(state);
    graph
}

fn planning_millis(graph: &Graph, profile: Profile) -> f64 {
    let mut fastest = f64::MAX;
    for _ in 0..3 {
        let started = Instant::now();
        let plan = Plan::of(graph, ALIGNMENT, profile);
        fastest = fastest.min(started.elapsed().as_secs_f64() * 1000.0);
        drop(plan);
    }
    fastest
}

#[test]
fn a_chain_of_updates_plans_beside_its_own_length() {
    let short = chain(512);
    let long = chain(2048);
    let short_millis = planning_millis(&short, narrow()).max(0.05);
    let long_millis = planning_millis(&long, narrow());
    assert!(
        long_millis < 8.0 * short_millis,
        "a chain of 2048 in-place updates planned in {long_millis:.1} ms beside {short_millis:.1} ms for 512, and an analysis that walks every pair of accesses grows sixteen times",
    );
}
