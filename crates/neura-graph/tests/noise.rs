use neura_abi::{Distribution, Element, Kind, noise};
use neura_graph::{Graph, Init, Shape};
use neura_pointwise as op;

fn refuses(action: impl FnOnce()) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).is_err()
}

#[test]
fn a_draw_asks_for_one_seed_and_hands_back_a_tensor_of_its_own() {
    let graph = Graph::new();
    let seed = graph.input(Shape::scalar(), Element::Single);
    let uniform = graph.uniform(Shape::matrix(4, 8), seed);
    let normal = graph.normal(Shape::matrix(4, 8), seed);
    assert_eq!(graph.shape(uniform), Shape::matrix(4, 8));
    assert_eq!(graph.element(uniform), Element::Single);
    assert_eq!(graph.element(normal), Element::Single);
    assert!(
        !graph.trains(uniform) && !graph.trains(normal),
        "a draw of a seed learns from the seed it names",
    );
    let snapshot = graph.snapshot();
    assert!(
        !snapshot.values()[uniform.id() as usize].requires_grad
            && !snapshot.values()[normal.id() as usize].requires_grad,
        "a draw of a seed is a leaf nothing descends from",
    );
    let tasks = snapshot.tasks();
    assert_eq!(tasks.len(), 2, "a draw is one task");
    assert_eq!(tasks[0].kind, Kind::Noise);
    assert_eq!(tasks[0].op, noise::UNIFORM);
    assert_eq!(tasks[0].inputs[0], seed.id());
    assert_eq!(tasks[1].kind, Kind::Noise);
    assert_eq!(tasks[1].op, noise::NORMAL);
    assert_eq!(tasks[1].inputs[0], seed.id());
    assert_eq!(Distribution::of(noise::UNIFORM), Distribution::Uniform);
    assert_eq!(Distribution::of(noise::NORMAL), Distribution::Normal);
    assert_eq!(Distribution::Uniform.name(), "uniform");
    assert_eq!(Distribution::Normal.symbol(), "noise::NORMAL");
}

#[test]
fn a_draw_refuses_the_seeds_it_cannot_read() {
    let graph = Graph::new();
    let vector = graph.input(Shape::vector(4), Element::Single);
    assert!(
        refuses(|| {
            let _ = graph.uniform(Shape::matrix(2, 2), vector);
        }),
        "a draw of a seed of four numbers takes no seed",
    );
    assert!(refuses(|| {
        let _ = graph.normal(Shape::matrix(2, 2), vector);
    }),);
}

#[test]
fn a_dropout_scales_the_numbers_it_keeps_and_zeroes_the_rest() {
    let graph = Graph::new();
    let seed = graph.input(Shape::scalar(), Element::Single);
    let data = graph.gradient_input(Shape::matrix(2, 3), Element::Single);
    let dropped = graph.dropout(data, 0.5, seed);
    assert_eq!(graph.shape(dropped), Shape::matrix(2, 3));
    let snapshot = graph.snapshot();
    assert!(
        snapshot.values()[dropped.id() as usize].requires_grad,
        "a dropout of a tracked tensor learns",
    );
    let tasks = snapshot.tasks();
    let draw = tasks
        .iter()
        .find(|task| task.kind == Kind::Noise)
        .expect("a dropout draws the mask it keeps");
    assert_eq!(draw.op, noise::UNIFORM);
    assert_eq!(draw.inputs[0], seed.id());
    let mask = tasks
        .iter()
        .find(|task| task.kind == Kind::Binary && task.op == op::LESS)
        .expect("a dropout compares every draw against the share it keeps");
    assert_eq!(mask.inputs[0], draw.out);
    let threshold = tasks
        .iter()
        .find(|task| task.kind == Kind::Fill && task.param == 0.5)
        .expect("a dropout compares every draw against the share it keeps");
    assert_eq!(mask.inputs[1], threshold.out);
    let picked = tasks
        .iter()
        .find(|task| task.kind == Kind::Select)
        .expect("a dropout picks the numbers its mask keeps");
    assert_eq!(picked.inputs[0], mask.out);
    assert_eq!(picked.inputs[1], data.id());
}

#[test]
fn a_dropout_keeps_between_none_and_all_of_the_numbers() {
    let graph = Graph::new();
    let seed = graph.input(Shape::scalar(), Element::Single);
    let data = graph.parameter(Shape::matrix(2, 3), Init::Zero, Element::Single);
    for keep in [0.0f32, -0.5, 1.5] {
        assert!(
            refuses(|| {
                let _ = graph.dropout(data, keep, seed);
            }),
            "a dropout that keeps {keep} of the numbers it draws is no dropout",
        );
    }
    assert_eq!(graph.dropout(data, 1.0, seed).shape(), Shape::matrix(2, 3));
}
