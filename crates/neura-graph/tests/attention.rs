use neura_abi::{EXACT_WALK_LIMIT, Element};
use neura_graph::{AttentionOptions, Graph, Init, Shape};

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

fn windowed(reach: u32, causal: bool) -> Graph<'static> {
    let graph = Graph::new();
    let tensor = || graph.parameter(Shape::of([1, 1, 4, 2]), Init::Zero, Element::Single);
    let (query, key, value) = (tensor(), tensor(), tensor());
    graph.attention(
        query,
        key,
        value,
        AttentionOptions {
            scale: 0.5,
            causal,
            origin: None,
            segments: None,
            reach: Some(reach),
        },
    );
    graph
}

#[test]
fn an_attention_that_reaches_back_over_no_key_is_refused() {
    assert!(refuses(|| {
        let _ = windowed(0, true);
    }));
}

#[test]
fn an_attention_that_reaches_back_over_keys_without_a_mask_is_refused() {
    assert!(refuses(|| {
        let _ = windowed(4, false);
    }));
}

#[test]
fn an_attention_that_reaches_back_over_keys_a_device_counts_endlessly_is_refused() {
    assert!(refuses(|| {
        let _ = windowed(EXACT_WALK_LIMIT + 1, true);
    }));
}

#[test]
fn an_attention_that_reaches_back_over_keys_hands_them_to_every_task() {
    let graph = windowed(2, true);
    let snapshot = graph.snapshot();
    let task = &snapshot.tasks()[0];
    assert_eq!(task.reach, 2);
    assert_eq!(task.slot, 1);
}
