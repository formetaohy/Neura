use neura_abi::{Element, Placement};
use neura_graph::{AttentionOptions, Graph, Init, Shape};
use neura_plan::{DEFAULT_ENCODING_BYTES, Plan};
use neura_profile::{Budget, Profile};

#[test]
fn an_image_restores_every_walk_the_plan_delivers_at_each_binding() {
    assert_ne!(
        neura_plan::PLAN_BUILD,
        0,
        "a plan build identity names the sources it hashes",
    );
    let profile = Profile::ladder(Budget::BASELINE, None)
        .into_iter()
        .find(|profile| profile.workgroup() == Budget::BALANCED_THREADS)
        .expect("the baseline budget holds a balanced profile");
    let graph = Graph::new();
    let rows = graph.free(4);
    let left = graph.input(Shape::of([4, 1, 8, 8]).freed(&[(0, rows)]), Element::Single);
    let right = graph.parameter(Shape::of([1, 1, 8, 8]), Init::Zero, Element::Single);
    let product = graph.matmul(left, right);
    let query = graph.parameter(Shape::of([1, 1, 4, 16]), Init::Zero, Element::Single);
    let key = graph.parameter(Shape::of([1, 1, 16, 16]), Init::Zero, Element::Single);
    let value = graph.parameter(Shape::of([1, 1, 16, 16]), Init::Zero, Element::Single);
    let attended = graph.attention(
        query,
        key,
        value,
        AttentionOptions {
            bias: None,
            scale: None,
            causal: false,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(product);
    graph.retain(attended);
    let plan = Plan::chosen(&graph, 256, profile, &[], DEFAULT_ENCODING_BYTES, 0);
    let again = Plan::chosen(&graph, 256, profile, &[], DEFAULT_ENCODING_BYTES, 0);
    assert_eq!(
        plan.image(),
        again.image(),
        "a plan image is what every compile of its graph writes",
    );
    let image = plan.image();
    for at in [8usize, 16usize] {
        let mut changed = image.clone();
        changed[at] ^= 1;
        assert!(
            Plan::restore(&graph, 256, profile, DEFAULT_ENCODING_BYTES, 0, &changed).is_none(),
            "a plan image built by another planner or for another graph is refused",
        );
    }
    let restored = Plan::restore(&graph, 256, profile, DEFAULT_ENCODING_BYTES, 0, &image)
        .expect("a plan image restores the plan it names");

    assert_eq!(plan.slot_bounds(), restored.slot_bounds());
    assert_eq!(plan.tasks(), restored.tasks());
    assert_eq!(plan.values(), restored.values());
    assert_eq!(plan.segments(), restored.segments());
    assert_eq!(plan.wave_segments(), restored.wave_segments());
    assert_eq!(plan.kinds(), restored.kinds());
    assert_eq!(plan.attention(), restored.attention());
    assert_eq!(plan.convolution(), restored.convolution());
    assert_eq!(plan.convolution_weight(), restored.convolution_weight());
    assert_eq!(plan.convolution_input(), restored.convolution_input());
    assert_eq!(plan.products(), restored.products());
    assert_eq!(
        plan.walked_tiles().collect::<Vec<_>>(),
        restored.walked_tiles().collect::<Vec<_>>(),
    );
    assert_eq!(plan.tensor_bytes(), restored.tensor_bytes());
    assert_eq!(plan.arena_bytes(), restored.arena_bytes());
    for value in [left, right, product, query, key, value, attended] {
        assert_eq!(
            plan.span(value, Placement::new(0, 0)),
            restored.span(value, Placement::new(0, 0)),
        );
    }

    for lengths in [plan.slot_bounds().to_vec(), vec![2]] {
        let bound = plan.encode(&lengths);
        let restored_bound = restored.encode(&lengths);
        assert_eq!(bound.tasks(), restored_bound.tasks());
        assert_eq!(bound.values(), restored_bound.values());
        assert_eq!(bound.segments(), restored_bound.segments());
        assert_eq!(bound.waves(), restored_bound.waves());
        assert_eq!(bound.dependencies(), restored_bound.dependencies());
        assert_eq!(bound.successors(), restored_bound.successors());
        assert_eq!(bound.roots(), restored_bound.roots());
        assert_eq!(bound.quanta(), restored_bound.quanta());
        assert_eq!(bound.lengths(), restored_bound.lengths());
        assert_eq!(bound.arena_bytes(), restored_bound.arena_bytes());
        assert_eq!(bound.tensor_bytes(), restored_bound.tensor_bytes());
    }
}
