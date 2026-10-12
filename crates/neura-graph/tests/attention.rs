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
            bias: None,
            scale: Some(graph.knob(0.5)),
            causal,
            origin: None,
            segments: None,
            reach: Some(reach),
            query_segments: None,
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

#[test]
fn an_attention_carries_the_scale_it_weighs_with() {
    let graph = Graph::new();
    let tensor = || graph.parameter(Shape::of([1, 1, 4, 2]), Init::Zero, Element::Single);
    let (query, key, value) = (tensor(), tensor(), tensor());
    let scale = graph.knob(0.5);
    let options = |scale| AttentionOptions {
        bias: None,
        scale,
        causal: true,
        origin: None,
        segments: None,
        reach: None,
        query_segments: None,
    };
    let weighed = graph.attention(query, key, value, options(Some(scale)));
    let standard = graph.attention(query, key, value, options(None));
    let snapshot = graph.snapshot();
    let task = |out: neura_graph::Value<'static>| {
        snapshot
            .tasks()
            .iter()
            .find(|task| task.out == out.id())
            .expect("an attention is one task")
    };
    assert_eq!(task(weighed).knob, scale.id());
    assert_eq!(task(standard).knob, neura_abi::NO_VALUE);
}

#[test]
fn an_attention_refuses_a_scale_of_other_than_one_number() {
    let graph = Graph::new();
    let tensor = || graph.parameter(Shape::of([1, 1, 4, 2]), Init::Zero, Element::Single);
    let (query, key, value) = (tensor(), tensor(), tensor());
    let shares = graph.parameter(Shape::vector(2), Init::Zero, Element::Single);
    assert!(refuses(|| {
        let _ = graph.attention(
            query,
            key,
            value,
            AttentionOptions {
                bias: None,
                scale: Some(shares),
                causal: true,
                origin: None,
                segments: None,
                reach: None,
                query_segments: None,
            },
        );
    }));
}

fn biased() -> (
    Graph<'static>,
    neura_graph::Value<'static>,
    neura_graph::Value<'static>,
) {
    let graph = Graph::new();
    let tensor = || graph.parameter(Shape::of([1, 1, 4, 2]), Init::Zero, Element::Single);
    let (query, key, value) = (tensor(), tensor(), tensor());
    let bias = graph.resident(Shape::of([1, 1, 1, 4]), Element::Single);
    let out = graph.attention(
        query,
        key,
        value,
        AttentionOptions {
            bias: Some(bias),
            scale: Some(graph.knob(0.5)),
            causal: false,
            origin: None,
            segments: None,
            reach: None,
            query_segments: None,
        },
    );
    graph.retain(out);
    (graph, bias, out)
}

#[test]
fn an_attention_carries_the_bias_it_shifts_every_score_by() {
    let (graph, bias, _) = biased();
    let snapshot = graph.snapshot();
    assert_eq!(snapshot.tasks()[0].kind, neura_abi::Kind::Attention);
    assert_eq!(snapshot.tasks()[0].bias, bias.id());
}

#[test]
fn every_gradient_of_a_biased_attention_shifts_the_scores_of_one_bias() {
    let (graph, bias, out) = biased();
    graph.backward(graph.sum(out));
    let snapshot = graph.snapshot();
    let mut walked = 0;
    for task in snapshot.tasks() {
        if !matches!(
            task.kind,
            neura_abi::Kind::Attention
                | neura_abi::Kind::AttentionQueryGrad
                | neura_abi::Kind::AttentionKeyGrad
                | neura_abi::Kind::AttentionValueGrad
        ) {
            continue;
        }
        assert_eq!(task.bias, bias.id());
        walked += 1;
    }
    assert_eq!(walked, 4);
}
